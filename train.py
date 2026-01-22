import os
import json
from typing import Optional, List, Literal, Tuple, Dict, cast, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
import fire
from tqdm import tqdm

from sae_lens import SAE
from transformers import PreTrainedModel, PreTrainedTokenizer

from corrsteer.utils import (
  load_model_tokenizer,
  load_dataloaders,
  get_device,
  fix_seed,
  generate_options,
  get_logit_processor,
  get_dims,
  get_eos_positions,
  build_prompt,
  extract_answer,
  load_sae,
)
from corrsteer.config import dataset_config, model_config, calculate_reward
from corrsteer.steer import get_steering_hook
from eval import EvalController, FixedFeaturePolicyNetwork

class FeatureExample(BaseModel):
  prompt: str
  baseline: str
  steered: str
  oracle: str
  think: Optional[str] = None

class ActivationCaptureHook:
  """Capture activations (SAE-encoded or raw residual stream)."""

  def __init__(self, sae: Optional[SAE] = None, mask: str = 'generation', raw: bool = False):
    self.sae: Optional[SAE] = sae
    self.mask: str = mask
    self.raw: bool = raw
    self.buffer: Optional[torch.Tensor] = None  # (B, T, dict_size) or (B, T, hidden_dim)

  def __call__(self, _: nn.Module, inputs: tuple[torch.Tensor]) -> None:
    residual: torch.Tensor = inputs[0]
    if self.mask == 'all':
      tokens: torch.Tensor = residual
    else:
      tokens: torch.Tensor = residual[:, -1:, :]
    
    if self.raw:
      # Use raw residual stream directly
      activations = tokens.detach()
    else:
      # Use SAE encoding
      tokens = tokens.to(self.sae.dtype)
      batch_size, seq_len, hidden_dim = tokens.shape
      tokens_2d: torch.Tensor = tokens.view(-1, hidden_dim)
      encoded_2d: torch.Tensor = self.sae.encode(tokens_2d)
      dict_size: int = encoded_2d.shape[-1]
      activations = encoded_2d.view(batch_size, seq_len, dict_size)
    
    if self.buffer is None:
      self.buffer = activations
    else:
      self.buffer = torch.cat([self.buffer, activations], dim=1)
    return None


class StreamingCorrelationAccumulator:
  """Streaming Pearson r per feature and mean positive activation (O(1) memory)."""

  def __init__(self, dict_size: int, real: bool = False, pos_only: bool = False, neg_only: bool = False, mi_edges: Optional[torch.Tensor] = None, fisher_enabled: bool = False):
    self.dict_size = dict_size
    self.real = real
    self.pos_only = pos_only
    self.neg_only = neg_only
    self.sum_x = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_xx = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_xy = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_y = 0.0
    self.sum_yy = 0.0
    self.n = 0
    self.sum_x_pos = torch.zeros(dict_size, dtype=torch.float64)
    self.count_pos = 0
    self.count_active = torch.zeros(dict_size, dtype=torch.int64)
    self.count_active_correct = torch.zeros(dict_size, dtype=torch.int64)
    self.count_active_incorrect = torch.zeros(dict_size, dtype=torch.int64)
    self.count_correct = 0
    self.count_incorrect = 0
    self.sum_x_success = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_x_failure = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_xx_success = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_xx_failure = torch.zeros(dict_size, dtype=torch.float64)
    self.count_success = 0
    self.count_failure = 0
    self.sum_x_pos_active = torch.zeros(dict_size, dtype=torch.float64)
    self.count_pos_active = torch.zeros(dict_size, dtype=torch.int64)
    self.sum_x_success_active = torch.zeros(dict_size, dtype=torch.float64)
    self.sum_x_failure_active = torch.zeros(dict_size, dtype=torch.float64)
    self.count_success_active = torch.zeros(dict_size, dtype=torch.int64)
    self.count_failure_active = torch.zeros(dict_size, dtype=torch.int64)
    # Exact MI (nonlinear) via fixed log-binned histograms
    self.mi_enabled = mi_edges is not None
    if self.mi_enabled:
      # mi_edges: ascending boundaries for bucketize; num_bins = len(edges)+1
      self.mi_edges = mi_edges.to(dtype=torch.float32)
      self.mi_num_bins = int(self.mi_edges.numel()) + 1
      self.mi_counts_success = torch.zeros((dict_size, self.mi_num_bins), dtype=torch.int64)
      self.mi_counts_failure = torch.zeros((dict_size, self.mi_num_bins), dtype=torch.int64)
    self.fisher_enabled = fisher_enabled

  @torch.no_grad()
  def update_corr(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
    if batch_x.numel() == 0:
      return
    batch_x = batch_x.to(dtype=torch.float64, device=self.sum_x.device)
    batch_y = batch_y.to(dtype=torch.float64, device=self.sum_x.device)
    B = batch_x.shape[0]
    self.n += int(B)
    self.sum_x += batch_x.sum(dim=0)
    self.sum_xx += (batch_x * batch_x).sum(dim=0)
    y_col = batch_y.view(-1, 1)
    self.sum_xy += (batch_x * y_col).sum(dim=0)
    self.sum_y += float(batch_y.sum())
    self.sum_yy += float((batch_y * batch_y).sum())
    # Note: MI counts updated separately in update_mi
    active_mask = batch_x > 0.0
    self.count_active += active_mask.sum(dim=0).to(dtype=torch.int64)
    correct_mask = batch_y > 0.0
    incorrect_mask = batch_y == 0.0
    if correct_mask.any():
      self.count_correct += int(correct_mask.sum().item())
      active_correct_mask = active_mask[correct_mask]
      self.count_active_correct += active_correct_mask.sum(dim=0).to(dtype=torch.int64)
    if incorrect_mask.any():
      self.count_incorrect += int(incorrect_mask.sum().item())
      active_incorrect_mask = active_mask[incorrect_mask]
      self.count_active_incorrect += active_incorrect_mask.sum(dim=0).to(dtype=torch.int64)

  @torch.no_grad()
  def update_coeff(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
    if batch_x.numel() == 0:
      return
    batch_x = batch_x.to(dtype=torch.float64, device=self.sum_x.device)
    batch_y = batch_y.to(dtype=torch.float64, device=self.sum_x.device)
    pos_mask = batch_y > 0.0
    if pos_mask.any():
      self.count_pos += int(pos_mask.sum().item())
      self.sum_x_pos += batch_x[pos_mask].sum(dim=0)
    success_mask = batch_y > 0.0
    failure_mask = batch_y == 0.0
    if success_mask.any():
      self.count_success += int(success_mask.sum().item())
      self.sum_x_success += batch_x[success_mask].sum(dim=0)
      self.sum_xx_success += (batch_x[success_mask] * batch_x[success_mask]).sum(dim=0)
    if failure_mask.any():
      self.count_failure += int(failure_mask.sum().item())
      self.sum_x_failure += batch_x[failure_mask].sum(dim=0)
      self.sum_xx_failure += (batch_x[failure_mask] * batch_x[failure_mask]).sum(dim=0)
    if self.real:
      pos_mask_expanded = pos_mask.view(-1, 1)
      pos_active_features = (batch_x > 0.0) & pos_mask_expanded
      if pos_active_features.any():
        self.sum_x_pos_active += (batch_x * pos_active_features.float()).sum(dim=0)
        self.count_pos_active += pos_active_features.sum(dim=0).to(dtype=torch.int64)
      success_mask_expanded = success_mask.view(-1, 1)
      success_active_features = (batch_x > 0.0) & success_mask_expanded
      if success_active_features.any():
        self.sum_x_success_active += (batch_x * success_active_features.float()).sum(dim=0)
        self.count_success_active += success_active_features.sum(dim=0).to(dtype=torch.int64)
      failure_mask_expanded = failure_mask.view(-1, 1)
      failure_active_features = (batch_x > 0.0) & failure_mask_expanded
      if failure_active_features.any():
        self.sum_x_failure_active += (batch_x * failure_active_features.float()).sum(dim=0)
        self.count_failure_active += failure_active_features.sum(dim=0).to(dtype=torch.int64)

  @torch.no_grad()
  def update_mi(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
    if not self.mi_enabled:
      return
    if batch_x.numel() == 0:
      return
    # batch_x: (B, D), batch_y: (B,)
    batch_x = batch_x.to(dtype=torch.float32)
    B, D = batch_x.shape
    # Bin indices per value using fixed edges; shape (B, D)
    edges = self.mi_edges.to(device=batch_x.device)
    bin_idx = torch.bucketize(batch_x, edges)
    num_bins = self.mi_num_bins
    # One-hot encode bins: (B, D, num_bins)
    one_hot = F.one_hot(bin_idx.to(torch.int64), num_classes=num_bins).to(dtype=torch.int32)
    # Masks for success/failure
    by = batch_y.to(device=one_hot.device)
    success_mask = (by > 0.0).view(B, 1, 1).to(dtype=one_hot.dtype)
    failure_mask = (by == 0.0).view(B, 1, 1).to(dtype=one_hot.dtype)
    # Sum over batch to get per-feature counts per bin
    counts_success = (one_hot * success_mask).sum(dim=0).to(dtype=torch.int64)  # (D, num_bins)
    counts_failure = (one_hot * failure_mask).sum(dim=0).to(dtype=torch.int64)  # (D, num_bins)
    # Accumulate
    self.mi_counts_success += counts_success.to(device=self.mi_counts_success.device)
    self.mi_counts_failure += counts_failure.to(device=self.mi_counts_failure.device)

  @torch.no_grad()
  def correlations(self) -> torch.Tensor:
    if self.n == 0:
      return torch.zeros(self.dict_size, dtype=torch.float64)
    n = float(self.n)
    mean_y = self.sum_y / n
    var_y = self.sum_yy / n - mean_y * mean_y
    mean_x = self.sum_x / n
    var_x = self.sum_xx / n - mean_x * mean_x
    cov_xy = self.sum_xy / n - mean_x * mean_y
    denom = torch.sqrt(var_x.clamp(min=1e-12) * max(var_y, 1e-12))
    r = torch.where(denom > 0, cov_xy / denom, torch.zeros_like(cov_xy))
    return r

  @torch.no_grad()
  def mean_coefficient(self, feature_idx: int) -> float:
    if self.count_pos == 0:
      return 0.0
    return float(self.sum_x_pos[feature_idx] / self.count_pos)
  
  @torch.no_grad()
  def frequency_percent(self, feature_idx: int) -> float:
    if self.n == 0:
      return 0.0
    return float(self.count_active[feature_idx] / self.n * 100.0)
  
  @torch.no_grad()
  def frequency_percent_correct(self, feature_idx: int) -> float:
    if self.n == 0:
      return 0.0
    return float(self.count_active_correct[feature_idx] / self.n * 100.0)
  
  @torch.no_grad()
  def frequency_percent_incorrect(self, feature_idx: int) -> float:
    if self.n == 0:
      return 0.0
    return float(self.count_active_incorrect[feature_idx] / self.n * 100.0)
  
  @torch.no_grad()
  def mean_coefficient_success(self, feature_idx: int) -> float:
    if self.count_success == 0:
      return 0.0
    return float(self.sum_x_success[feature_idx] / self.count_success)
  
  @torch.no_grad()
  def mean_coefficient_failure(self, feature_idx: int) -> float:
    if self.count_failure == 0:
      return 0.0
    return float(self.sum_x_failure[feature_idx] / self.count_failure)
  
  @torch.no_grad()
  def mean_coefficient_real(self, feature_idx: int) -> float:
    if self.count_pos_active[feature_idx] == 0:
      return 0.0
    return float(self.sum_x_pos_active[feature_idx] / self.count_pos_active[feature_idx])
  
  @torch.no_grad()
  def mean_coefficient_success_real(self, feature_idx: int) -> float:
    if self.count_success_active[feature_idx] == 0:
      return 0.0
    return float(self.sum_x_success_active[feature_idx] / self.count_success_active[feature_idx])
  
  @torch.no_grad()
  def mean_coefficient_failure_real(self, feature_idx: int) -> float:
    if self.count_failure_active[feature_idx] == 0:
      return 0.0
    return float(self.sum_x_failure_active[feature_idx] / self.count_failure_active[feature_idx])

  @torch.no_grad()
  def fisher_scores(self) -> torch.Tensor:
    """Fisher Information approximation via squared activation ratio (DSG Section 3.2)."""
    eps = 1e-12
    c1 = max(self.count_success, 1)
    c0 = max(self.count_failure, 1)
    success_score = self.sum_xx_success / c1
    failure_score = self.sum_xx_failure / c0
    imp_ratio = success_score / torch.clamp(failure_score, min=eps)
    return imp_ratio.to(dtype=torch.float64)


  @torch.no_grad()
  def mi_scores(self) -> torch.Tensor:
    """Exact MI from empirical histograms (discrete bins over activations)."""
    if not self.mi_enabled:
      # Fallback to Gaussian approximation if MI not enabled
      r = self.correlations().clamp(min=-0.999999, max=0.999999)
      return (-0.5 * torch.log1p(-(r * r)) ).to(dtype=torch.float64)
    eps = 1e-12
    counts1 = self.mi_counts_success.to(dtype=torch.float64)  # (D, B)
    counts0 = self.mi_counts_failure.to(dtype=torch.float64)  # (D, B)
    n1 = counts1.sum(dim=1)  # (D,)
    n0 = counts0.sum(dim=1)  # (D,)
    N = n1 + n0  # (D,)
    # Avoid divide by zero: mask features with N==0 -> MI=0
    valid = N > 0
    # p(y)
    p1 = torch.zeros_like(N)
    p0 = torch.zeros_like(N)
    p1[valid] = n1[valid] / N[valid]
    p0[valid] = n0[valid] / N[valid]
    # p(z)
    pz = torch.zeros_like(counts1)
    pz[valid] = (counts1[valid] + counts0[valid]) / N[valid].unsqueeze(1)
    # p(z,y)
    pz1 = torch.zeros_like(counts1)
    pz0 = torch.zeros_like(counts0)
    pz1[valid] = counts1[valid] / N[valid].unsqueeze(1)
    pz0[valid] = counts0[valid] / N[valid].unsqueeze(1)
    # Compute MI per feature: sum_b sum_y p(z,y) log p(z,y)/(p(z)p(y))
    def safe_term(pzy, pz, py):
      num = pzy.clamp(min=eps)
      den = (pz.clamp(min=eps) * py.clamp(min=eps).unsqueeze(1))
      return num * torch.log(num / den)
    term1 = safe_term(pz1, pz, p1)
    term0 = safe_term(pz0, pz, p0)
    I = (term1 + term0).sum(dim=1)
    I[~valid] = 0.0
    return I.to(dtype=torch.float64)

  @torch.no_grad()
  def contrastive_diff(self) -> torch.Tensor:
    """Absolute difference of means between success and failure."""
    c1 = max(self.count_success, 1)
    c0 = max(self.count_failure, 1)
    mu1 = self.sum_x_success / c1
    mu0 = self.sum_x_failure / c0
    return (mu1 - mu0).abs().to(dtype=torch.float64)

  @torch.no_grad()
  def top_features_all(self, k: int = 1) -> Tuple[List[Tuple[int, float, float, float, dict]], List[Tuple[int, float, float, float, dict]]]:
    """Always return both positive and negative features for storage"""
    r = self.correlations()
    _, top_pos_indices = torch.topk(r, k)
    _, top_neg_indices = torch.topk(r, k, largest=False)
    
    top_positive = []
    for idx in top_pos_indices:
      idx = int(idx.item())
      if self.real:
        coeff = self.mean_coefficient_real(idx)
      else:
        coeff = float(self.sum_x_pos[idx] / max(self.count_pos, 1))
      freq = self.frequency_percent(idx)
      stats = {
        "freq_correct": self.frequency_percent_correct(idx),
        "freq_incorrect": self.frequency_percent_incorrect(idx),
        "coeff_success": self.mean_coefficient_success_real(idx) if self.real else self.mean_coefficient_success(idx),
        "coeff_failure": self.mean_coefficient_failure_real(idx) if self.real else self.mean_coefficient_failure(idx)
      }
      top_positive.append((idx, coeff, float(r[idx].item()), freq, stats))
    
    top_negative = []
    for idx in top_neg_indices:
      idx = int(idx.item())
      if self.real:
        coeff = self.mean_coefficient_real(idx)
      else:
        coeff = float(self.sum_x_pos[idx] / max(self.count_pos, 1))
      freq = self.frequency_percent(idx)
      stats = {
        "freq_correct": self.frequency_percent_correct(idx),
        "freq_incorrect": self.frequency_percent_incorrect(idx),
        "coeff_success": self.mean_coefficient_success_real(idx) if self.real else self.mean_coefficient_success(idx),
        "coeff_failure": self.mean_coefficient_failure_real(idx) if self.real else self.mean_coefficient_failure(idx)
      }
      top_negative.append((idx, coeff, float(r[idx].item()), freq, stats))
    return (top_positive, top_negative)

  @torch.no_grad()
  def top_features(self, k: int = 1) -> Tuple[List[Tuple[int, float, float, float, dict]], List[Tuple[int, float, float, float, dict]]]:
    r = self.correlations()
    
    if self.pos_only:
      pos_mask = r > 0
      if pos_mask.any():
        filtered_r = torch.where(pos_mask, r, torch.tensor(-float('inf')))
        _, top_pos_indices = torch.topk(filtered_r, min(k, pos_mask.sum().item()))
      else:
        _, top_pos_indices = torch.topk(r, k)
      top_neg_indices = torch.tensor([])
    elif self.neg_only:
      neg_mask = r < 0
      if neg_mask.any():
        filtered_r = torch.where(neg_mask, r, torch.tensor(float('inf')))
        _, top_neg_indices = torch.topk(filtered_r, min(k, neg_mask.sum().item()), largest=False)
      else:
        _, top_neg_indices = torch.topk(r, k, largest=False)
      top_pos_indices = torch.tensor([])
    else:
      _, top_pos_indices = torch.topk(r, k)
      _, top_neg_indices = torch.topk(r, k, largest=False)
    
    top_positive = []
    if len(top_pos_indices) > 0:
      for idx in top_pos_indices:
        idx = int(idx.item())
        if self.real:
          coeff = self.mean_coefficient_real(idx)
        else:
          coeff = float(self.sum_x_pos[idx] / max(self.count_pos, 1))
        freq = self.frequency_percent(idx)
        stats = {
          "freq_correct": self.frequency_percent_correct(idx),
          "freq_incorrect": self.frequency_percent_incorrect(idx),
          "coeff_success": self.mean_coefficient_success_real(idx) if self.real else self.mean_coefficient_success(idx),
          "coeff_failure": self.mean_coefficient_failure_real(idx) if self.real else self.mean_coefficient_failure(idx)
        }
        
        top_positive.append((idx, coeff, float(r[idx].item()), freq, stats))
    top_negative = []
    if len(top_neg_indices) > 0:
      for idx in top_neg_indices:
        idx = int(idx.item())
        if self.real:
          coeff = self.mean_coefficient_real(idx)
        else:
          coeff = float(self.sum_x_pos[idx] / max(self.count_pos, 1))
        freq = self.frequency_percent(idx)
        stats = {
          "freq_correct": self.frequency_percent_correct(idx),
          "freq_incorrect": self.frequency_percent_incorrect(idx),
          "coeff_success": self.mean_coefficient_success_real(idx) if self.real else self.mean_coefficient_success(idx),
          "coeff_failure": self.mean_coefficient_failure_real(idx) if self.real else self.mean_coefficient_failure(idx)
        }
        top_negative.append((idx, coeff, float(r[idx].item()), freq, stats))
    return (top_positive, top_negative)


class CorrConfig(BaseModel):
  num_samples: int = 4000
  validate_every: int = 0
  batch_size: Optional[int] = None
  model: str = "gemma2b"
  task: str = "mmlu"
  layer: Union[int, Literal["global", "foreach"], List[int]] = 20
  seed: int = 42
  dtype: str = "bfloat16"
  output_dir: str = "checkpoints"
  scale: float = 1.0
  pool: Literal["max", "mean"] = "max"
  steer_pool: Literal["max", "mean"] = "max"
  few: Optional[int] = None
  category: Optional[str] = None
  filter_value: Optional[str] = None
  mask: Literal['generation', 'all'] = 'generation'
  topk: int = 20
  decode: bool = False
  select_token: bool = False
  limit: int = 200
  cot: bool = False
  stable: bool = False
  validate: bool = False
  report_every: int = 100
  real: bool = False
  pos: bool = True
  neg: bool = False
  eval: bool = False
  example: bool = False
  raw: bool = False
  reverse: bool = False
  # Selection/coeff method toggles
  mi: bool = False
  fisher: bool = False
  caa: bool = False
  caacoeff: bool = False
  # Ablation controls
  shuffle_labels: bool = False  # Permute labels for selection bias test
  random_features: bool = False  # Select random features instead of top correlated

  def __post_init__(self):
    if self.pos and self.neg:
      raise ValueError("Cannot use both --pos and --neg flags simultaneously")
    methods = int(self.mi) + int(self.fisher) + int(self.caa)
    if methods > 1:
      raise ValueError("Use only one of --mi, --fisher, or --caa")


class CorrSteerController:
  def __init__(self):
    self.device = get_device()
    self.tokenizer: PreTrainedTokenizer
    self.llm: PreTrainedModel
    self.sae: SAE
    self.config: CorrConfig
    self.snapshots: Dict[str, dict] = {}
    self.dtype = None
    self.baseline_results: List[str] = []
    self.baseline_activations: Dict[int, torch.Tensor] = {}
    self.validation_samples: List = []
    self.feature_examples: Dict[int, List[FeatureExample]] = {}

  def _pool_activations(self, encoded_steps: torch.Tensor, eos_positions: torch.Tensor, pool: str) -> torch.Tensor:
    """Reduce (B, T, D) to (B, D) up to EOS per sample by max/mean pooling."""
    B, T, D = encoded_steps.shape
    pooled = torch.zeros(B, D, dtype=encoded_steps.dtype, device=encoded_steps.device)
    for i in range(B):
      valid_len = int(eos_positions[i].item())
      valid_len = max(0, min(valid_len, T))
      if valid_len == 0:
        continue
      slice_i = encoded_steps[i, :valid_len, :]
      if pool == "max":
        pooled[i] = slice_i.max(dim=0).values
      else:
        pooled[i] = slice_i.mean(dim=0)
    return pooled

  def _compute_rewards(self, generated_texts: List[str], ground_truths: List[str]) -> List[float]:
    rewards: List[float] = []
    for pred, gt in zip(generated_texts, ground_truths):
      _, predicted_label = extract_answer(pred, self.config.task)
      reward = calculate_reward(predicted_label, gt, self.config.task, self.tokenizer)
      if getattr(self.config, "reverse", False):
        reward = 1.0 - float(reward)
      rewards.append(float(reward))
    return rewards

  def _validate_baseline(self, train_loader, val_loader, max_new_tokens: int, layers: List[int]) -> float:
    """Validate baseline and collect activations for all layers"""
    correct = 0
    total = 0
    self.baseline_results = []
    self.validation_samples = []
    self.baseline_activations = {}
    option_tokens = generate_options(self.tokenizer, self.config.task) if self.config.select_token else None
    processor = get_logit_processor(option_tokens) if self.config.select_token else None
    few_shots = None
    if self.config.few is not None and self.config.few > 0:
      few_shots = train_loader.get_last_samples(self.config.few)
    for sample in val_loader:
      prompt, gt = build_prompt(sample, self.config.task, cot=self.config.cot, few_shots=few_shots)
      inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      with torch.no_grad():
        generated_ids = cast(torch.Tensor, self.llm.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
        do_sample=False, logits_processor=processor,
        pad_token_id=self.tokenizer.pad_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
      ))
      offset = len(input_ids[0])
      generated_text = self.tokenizer.decode(generated_ids[0, offset:], skip_special_tokens=True)
      _, predicted_label = extract_answer(generated_text, self.config.task)
      reward = calculate_reward(predicted_label, gt, self.config.task, self.tokenizer)
      if getattr(self.config, "reverse", False):
        reward = 1.0 - float(reward)
      correct += reward
      total += 1
      self.baseline_results.append(predicted_label)
      self.validation_samples.append((sample, prompt, gt))
      del generated_ids
      torch.cuda.empty_cache()
    return correct / total * 100.0

  def _validate_feature(self, layer: int, feat_idx: int, coeff: float, train_loader, val_loader, max_new_tokens: int, collect_examples: bool = False) -> float:
    """Validate feature and optionally collect examples using stored baseline activations"""
    sae = self.saes[layer]
    _, dict_size = get_dims(self.llm, sae)
    policy_net = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    hook = get_steering_hook(policy_net, sae, decode=self.config.decode, lastk=1, multiple=1, mask=self.config.mask)
    option_tokens = generate_options(self.tokenizer, self.config.task) if self.config.select_token else None
    processor = get_logit_processor(option_tokens) if self.config.select_token else None
    few_shots = None
    if self.config.few is not None and self.config.few > 0:
      few_shots = train_loader.get_last_samples(self.config.few)
    
    correct = 0
    total = 0
    sample_idx = 0
    for sample in val_loader:
      prompt, gt = build_prompt(sample, self.config.task, cot=self.config.cot, few_shots=few_shots)
      inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      handle = self.llm.model.layers[layer].register_forward_pre_hook(hook)
      with torch.no_grad():
        generated_ids = cast(torch.Tensor, self.llm.generate(
          input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
          do_sample=False, logits_processor=processor,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
        ))
      handle.remove()
      offset = len(input_ids[0])
      generated_text = self.tokenizer.decode(generated_ids[0, offset:], skip_special_tokens=True)
      if self.config.cot:
        think, predicted_label = extract_answer(generated_text, self.config.task)
      else:
        think = None
        _, predicted_label = extract_answer(generated_text, self.config.task)
      reward = calculate_reward(predicted_label, gt, self.config.task, self.tokenizer)
      if getattr(self.config, "reverse", False):
        reward = 1.0 - float(reward)
      correct += reward
      total += 1
      if collect_examples:
        baseline_result = self.baseline_results[sample_idx] if sample_idx < len(self.baseline_results) else ""
        if baseline_result != predicted_label:
          example = FeatureExample(
            prompt=prompt,
            baseline=baseline_result,
            steered=predicted_label,
            oracle=gt,
            think=think
          )
          if feat_idx not in self.feature_examples:
            self.feature_examples[feat_idx] = []
          self.feature_examples[feat_idx].append(example)
      sample_idx += 1
      del generated_ids
      torch.cuda.empty_cache()
    return correct / total * 100.0

  def _select_best_feature_by_correlation(self, layer_result: dict) -> dict:
    """Select feature with highest absolute correlation without validation"""
    pos_feat = layer_result["top_positive"][0]
    neg_feat = layer_result["top_negative"][0]
    pos_abs_corr = abs(pos_feat["correlation"])
    neg_abs_corr = abs(neg_feat["correlation"])
    
    if pos_abs_corr >= neg_abs_corr:
      return {"feature_index": pos_feat["feature_index"], "coefficient": pos_feat["coefficient"], 
              "correlation": pos_feat["correlation"], "frequency": pos_feat["frequency"]}
    else:
      return {"feature_index": neg_feat["feature_index"], "coefficient": -abs(neg_feat["coefficient"]), 
              "correlation": neg_feat["correlation"], "frequency": neg_feat["frequency"]}

  def _validate_layer_features(self, layer: int, layer_result: dict, train_loader, val_loader, max_new_tokens: int) -> tuple[float, dict]:
    has_positive = layer_result["top_positive"] and len(layer_result["top_positive"]) > 0
    has_negative = layer_result["top_negative"] and len(layer_result["top_negative"]) > 0
    if self.config.pos and not has_positive:
      raise ValueError(f"Layer {layer} missing positive features for --pos validation")
    if self.config.neg and not has_negative:
      raise ValueError(f"Layer {layer} missing negative features for --neg validation")
    if not self.config.pos and not self.config.neg and (not has_positive or not has_negative):
      raise ValueError(f"Layer {layer} missing positive or negative features for validation")
    pos_feat_idx = layer_result["top_positive"][0]["feature_index"] if has_positive else None
    pos_coeff = layer_result["top_positive"][0]["coefficient"] if has_positive else None
    neg_feat_idx = layer_result["top_negative"][0]["feature_index"] if has_negative else None
    neg_base_coeff = layer_result["top_negative"][0]["coefficient"] if has_negative else None
    neg_coeff = -abs(neg_base_coeff) if has_negative else None
    if isinstance(self.config.layer, int) and self.config.stable and has_positive and has_negative:
      pos_accuracy = self._validate_feature(layer, pos_feat_idx, pos_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
      neg_accuracy = self._validate_feature(layer, neg_feat_idx, neg_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
      combined_accuracy = (pos_accuracy + neg_accuracy) / 2
      return combined_accuracy, {
        "positive": {"feature_index": pos_feat_idx, "coefficient": pos_coeff, "correlation": layer_result["top_positive"][0]["correlation"], "frequency": layer_result["top_positive"][0]["frequency"]},
        "negative": {"feature_index": neg_feat_idx, "coefficient": neg_coeff, "correlation": layer_result["top_negative"][0]["correlation"], "frequency": layer_result["top_negative"][0]["frequency"]}
      }
    elif self.config.pos and has_positive:
      pos_accuracy = self._validate_feature(layer, pos_feat_idx, pos_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
      return pos_accuracy, {"feature_index": pos_feat_idx, "coefficient": pos_coeff, "correlation": layer_result["top_positive"][0]["correlation"], "frequency": layer_result["top_positive"][0]["frequency"]}
    elif self.config.neg and has_negative:
      neg_accuracy = self._validate_feature(layer, neg_feat_idx, neg_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
      return neg_accuracy, {"feature_index": neg_feat_idx, "coefficient": neg_coeff, "correlation": layer_result["top_negative"][0]["correlation"], "frequency": layer_result["top_negative"][0]["frequency"]}
    else:
      pos_accuracy = self._validate_feature(layer, pos_feat_idx, pos_coeff, train_loader, val_loader, max_new_tokens, collect_examples=False) if has_positive else -1
      neg_accuracy = self._validate_feature(layer, neg_feat_idx, neg_coeff, train_loader, val_loader, max_new_tokens, collect_examples=False) if has_negative else -1
      if pos_accuracy >= neg_accuracy and has_positive:
        self._validate_feature(layer, pos_feat_idx, pos_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
        return pos_accuracy, {"feature_index": pos_feat_idx, "coefficient": pos_coeff, "correlation": layer_result["top_positive"][0]["correlation"], "frequency": layer_result["top_positive"][0]["frequency"]}
      elif has_negative:
        self._validate_feature(layer, neg_feat_idx, neg_coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
        return neg_accuracy, {"feature_index": neg_feat_idx, "coefficient": neg_coeff, "correlation": layer_result["top_negative"][0]["correlation"], "frequency": layer_result["top_negative"][0]["frequency"]}
      else:
        raise ValueError(f"No valid features to validate for layer {layer}")

  def _run_collection(self, train_loader, batch_size: int, max_new_tokens: int) -> None:
    total_processed = 0
    batch_samples: List = []
    progress_bar = tqdm(total=self.config.num_samples, desc="Collecting correlations", unit="samples")
    
    for sample in tqdm(train_loader, desc="Processing batches", leave=False):
      batch_samples.append(sample)
      if len(batch_samples) < batch_size:
        continue
      batch = batch_samples
      batch_samples = []
      if total_processed >= self.config.num_samples:
        break
      prompts, gts = zip(*[build_prompt(s, self.config.task, cot=False, few_shots=None) for s in batch])
      inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      capture_hooks = {layer: ActivationCaptureHook(self.saes[layer] if not self.config.raw else None, self.config.mask, self.config.raw) for layer in self.layers}
      handles = {layer: self.llm.model.layers[layer].register_forward_pre_hook(capture_hooks[layer]) for layer in self.layers}
      with torch.no_grad():
        generated_ids = cast(torch.Tensor, self.llm.generate(
          input_ids,
          attention_mask=attention_mask,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
        ))
      for handle in handles.values():
        handle.remove()
      eos_positions = get_eos_positions(generated_ids, input_ids, self.tokenizer, self.config.task)
      offsets = [len(input_ids[i]) for i in range(len(input_ids))]
      sliced = [generated_ids[i, offset:] for i, offset in enumerate(offsets)]
      generated_texts = self.tokenizer.batch_decode(sliced, skip_special_tokens=True)
      rewards = self._compute_rewards(generated_texts, list(gts))
      batch_y = torch.tensor(rewards, dtype=torch.float32, device=self.device)
      # Selection bias control: shuffle labels to test if correlation is spurious
      if self.config.shuffle_labels:
        perm = torch.randperm(batch_y.size(0), device=self.device)
        batch_y = batch_y[perm]
      for layer in self.layers:
        capture_hook = capture_hooks[layer]
        if capture_hook.buffer is None:
          continue
        encoded_steps = capture_hook.buffer
        pooled_corr = self._pool_activations(encoded_steps, eos_positions, self.config.pool)
        pooled_coeff = self._pool_activations(encoded_steps, eos_positions, self.config.steer_pool)
        self.accumulators[layer].update_corr(pooled_corr, batch_y)
        self.accumulators[layer].update_coeff(pooled_coeff, batch_y)
        if getattr(self.config, "mi", False):
          self.accumulators[layer].update_mi(pooled_corr, batch_y)
      total_processed += len(batch)
      progress_bar.update(len(batch))
      del generated_ids
      torch.cuda.empty_cache()
      if total_processed % self.config.report_every == 0:
        print(f"\n=== Sample {total_processed} - Top Features ===")
        for layer in self.layers:
          top_positive, top_negative = self.accumulators[layer].top_features(1)
          if len(top_positive) > 0 and len(top_negative) > 0:
            top_pos_idx, _, top_pos_r, top_pos_freq, _ = top_positive[0]
            top_neg_idx, _, top_neg_r, top_neg_freq, _ = top_negative[0]
            mean_coeff = self.accumulators[layer].mean_coefficient(top_pos_idx)
            print(f"Layer {layer}: pos {top_pos_idx} r={top_pos_r:.4f}, neg {top_neg_idx} r={top_neg_r:.4f}, coeff={mean_coeff:.4f}")
          elif len(top_positive) > 0:
            top_pos_idx, _, top_pos_r, top_pos_freq, _ = top_positive[0]
            mean_coeff = self.accumulators[layer].mean_coefficient(top_pos_idx)
            print(f"Layer {layer}: pos {top_pos_idx} r={top_pos_r:.4f}, neg None, coeff={mean_coeff:.4f}")
          elif len(top_negative) > 0:
            top_neg_idx, _, top_neg_r, top_neg_freq, _ = top_negative[0]
            print(f"Layer {layer}: pos None, neg {top_neg_idx} r={top_neg_r:.4f}")
          else:
            print(f"Layer {layer}: pos None, neg None")
        if self.config.eval and total_processed >= self.config.num_samples:
          self.snapshots[str(total_processed)] = {}
          for layer in self.layers:
            s_top_positive, _ = self.accumulators[layer].top_features(1)
            if len(s_top_positive) > 0:
              s_idx, s_coeff, s_corr, s_freq, s_stats = s_top_positive[0]
              self.snapshots[str(total_processed)][str(layer)] = {
                "selected": {
                  "feature_index": int(s_idx),
                  "coefficient": float(s_coeff * self.config.scale),
                  "correlation": float(s_corr),
                  "frequency": float(s_freq),
                  "stats": s_stats,
                }
              }
        print("=" * 40)
    if batch_samples and total_processed < self.config.num_samples:
      batch = batch_samples
      prompts, gts = zip(*[build_prompt(s, self.config.task, cot=False, few_shots=None) for s in batch])
      inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      capture_hooks = {layer: ActivationCaptureHook(self.saes[layer] if not self.config.raw else None, self.config.mask, self.config.raw) for layer in self.layers}
      handles = {layer: self.llm.model.layers[layer].register_forward_pre_hook(capture_hooks[layer]) for layer in self.layers}
      with torch.no_grad():
        generated_ids = cast(torch.Tensor, self.llm.generate(
          input_ids,
          attention_mask=attention_mask,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
        ))
      for handle in handles.values():
        handle.remove()
      eos_positions = get_eos_positions(generated_ids, input_ids, self.tokenizer, self.config.task)
      offsets = [len(input_ids[i]) for i in range(len(input_ids))]
      sliced = [generated_ids[i, offset:] for i, offset in enumerate(offsets)]
      generated_texts = self.tokenizer.batch_decode(sliced, skip_special_tokens=True)
      rewards = self._compute_rewards(generated_texts, list(gts))
      batch_y = torch.tensor(rewards, dtype=torch.float32, device=self.device)
      # Selection bias control: shuffle labels to test if correlation is spurious
      if self.config.shuffle_labels:
        perm = torch.randperm(batch_y.size(0), device=self.device)
        batch_y = batch_y[perm]
      for layer in self.layers:
        capture_hook = capture_hooks[layer]
        if capture_hook.buffer is not None:
          encoded_steps = capture_hook.buffer
          pooled_corr = self._pool_activations(encoded_steps, eos_positions, self.config.pool)
          pooled_coeff = self._pool_activations(encoded_steps, eos_positions, self.config.steer_pool)
          self.accumulators[layer].update_corr(pooled_corr, batch_y)
          self.accumulators[layer].update_coeff(pooled_coeff, batch_y)
          if getattr(self.config, "mi", False):
            self.accumulators[layer].update_mi(pooled_corr, batch_y)
      total_processed += len(batch)
      progress_bar.update(len(batch))
      del generated_ids
      torch.cuda.empty_cache()
      if total_processed % self.config.report_every == 0:
        print(f"\n=== Sample {total_processed} - Top Features ===")
        for layer in self.layers:
          top_positive, top_negative = self.accumulators[layer].top_features(1)
          pos_str = f"pos {top_positive[0][0]} r={top_positive[0][2]:.4f}" if len(top_positive) > 0 else "pos None"
          neg_str = f"neg {top_negative[0][0]} r={top_negative[0][2]:.4f}" if len(top_negative) > 0 else "neg None"
          if len(top_positive) > 0:
            mean_coeff = self.accumulators[layer].mean_coefficient(top_positive[0][0])
            print(f"Layer {layer}: {pos_str}, {neg_str}, coeff={mean_coeff:.4f}")
          else:
            print(f"Layer {layer}: {pos_str}, {neg_str}")
        print("=" * 40)
    progress_bar.close()

  def train(self, **kwargs) -> Dict[str, object]:
    cfg = CorrConfig(**kwargs)
    self.config = cfg
    cfg.output_dir = f"{cfg.output_dir}_{cfg.seed}"
    fix_seed(cfg.seed)
    if cfg.layer == "foreach":
      cfg.validate = True
      self.config.validate = True
    dtype = getattr(torch, cfg.dtype) if isinstance(cfg.dtype, str) else cfg.dtype
    tokenizer, llm = load_model_tokenizer(cfg.model, dtype=dtype)
    self.tokenizer = tokenizer
    self.llm = llm
    train_loader, val_loader, _ = load_dataloaders(
      cfg.task,
      seed=cfg.seed,
      category=cfg.category,
      filter_value=cfg.filter_value,
      val_limit=cfg.limit,
    )
    batch_size = cfg.batch_size if cfg.batch_size else model_config[cfg.model].batch_size
    max_new_tokens = dataset_config[cfg.task].max_new_tokens
    if isinstance(cfg.layer, int):
      layers = [cfg.layer]
    elif cfg.layer in ["global", "foreach"]:
      layers = list(range(1, llm.config.num_hidden_layers))
    elif isinstance(cfg.layer, list):
      layers = cfg.layer
    else:
      raise ValueError(f"Invalid layer: {cfg.layer}")

    print(f"Training correlations for layers: {layers}")
    self.layers = layers
    self.saes = {}
    self.accumulators = {}
    for layer in layers:
      if cfg.raw:
        # Use raw residual stream - no SAE needed
        self.saes[layer] = None
        hidden_dim = llm.config.hidden_size
        feature_dim = hidden_dim
      else:
        # Use SAE encoding
        sae, _, _ = load_sae(cfg.model, layer, self.device)
        _, dict_size = get_dims(llm, sae)
        self.saes[layer] = sae
        feature_dim = dict_size
      mi_edges = None
      if getattr(cfg, "mi", False):
        try:
          import math
          edges = []
          per_decade = 4
          for d in range(-6, 1):
            for k in range(1, per_decade):
              edges.append(10 ** (d + k / per_decade))
          edges.append(1.0)
          mi_edges = torch.tensor(edges, dtype=torch.float32)
        except Exception:
          mi_edges = torch.tensor([1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float32)
      fisher_enabled = getattr(cfg, "fisher", False)
      self.accumulators[layer] = StreamingCorrelationAccumulator(feature_dim, real=cfg.real, pos_only=cfg.pos, neg_only=cfg.neg, mi_edges=mi_edges, fisher_enabled=fisher_enabled)
    self._run_collection(train_loader, batch_size, max_new_tokens)
    layer_results = {}
    for layer in layers:
      accumulator = self.accumulators[layer]
      # Build ranking scores based on method
      if cfg.mi:
        scores = accumulator.mi_scores()
      elif cfg.fisher:
        scores = accumulator.fisher_scores()
      elif cfg.caa:
        scores = accumulator.contrastive_diff()
      else:
        scores = accumulator.correlations().abs()
      r = accumulator.correlations()
      # Determine masks based on desired polarity
      if cfg.pos:
        mask_pos = r > 0
        mask_neg = torch.zeros_like(mask_pos)
      elif cfg.neg:
        mask_pos = torch.zeros_like(r, dtype=torch.bool)
        mask_neg = r < 0
      else:
        mask_pos = r > 0
        mask_neg = r < 0

      # Helper to assemble top-k tuples using chosen coefficient strategy
      def build_topk(mask: torch.Tensor, k: int) -> List[Tuple[int, float, float, float, dict]]:
        if mask.any():
          masked_scores = torch.where(mask, scores, torch.tensor(-float('inf')))
          _, top_idx = torch.topk(masked_scores, min(k, int(mask.sum().item())))
        else:
          top_idx = torch.tensor([])
        results: List[Tuple[int, float, float, float, dict]] = []
        for idx_t in top_idx:
          i = int(idx_t.item())
          # Coefficient: positive mean or contrastive diff
          if cfg.real:
            base_coeff = accumulator.mean_coefficient_real(i)
          else:
            if cfg.caacoeff:
              pos_mean = accumulator.mean_coefficient_success(i)
              neg_mean = accumulator.mean_coefficient_failure(i)
              base_coeff = float(pos_mean - neg_mean)
            else:
              base_coeff = accumulator.mean_coefficient(i)
          freq = accumulator.frequency_percent(i)
          stat = {
            "freq_correct": accumulator.frequency_percent_correct(i),
            "freq_incorrect": accumulator.frequency_percent_incorrect(i),
            "coeff_success": accumulator.mean_coefficient_success_real(i) if cfg.real else accumulator.mean_coefficient_success(i),
            "coeff_failure": accumulator.mean_coefficient_failure_real(i) if cfg.real else accumulator.mean_coefficient_failure(i),
          }
          corr_val = float(r[i].item())
          results.append((i, float(base_coeff), corr_val, freq, stat))
        return results

      top_positive = build_topk(mask_pos, cfg.topk)
      top_negative = build_topk(mask_neg, cfg.topk)

      # Selection bias control: replace with random features
      if cfg.random_features and len(top_positive) > 0:
        import random as py_random
        # Get all positive-correlated feature indices
        pos_indices = mask_pos.nonzero(as_tuple=True)[0].tolist()
        if len(pos_indices) >= len(top_positive):
          random_indices = py_random.sample(pos_indices, len(top_positive))
          top_positive = []
          for i in random_indices:
            base_coeff = accumulator.mean_coefficient(i)
            freq = accumulator.frequency_percent(i)
            stat = {
              "freq_correct": accumulator.frequency_percent_correct(i),
              "freq_incorrect": accumulator.frequency_percent_incorrect(i),
              "coeff_success": accumulator.mean_coefficient_success(i),
              "coeff_failure": accumulator.mean_coefficient_failure(i),
            }
            corr_val = float(r[i].item())
            top_positive.append((i, float(base_coeff), corr_val, freq, stat))

      # For storage: also assemble top lists without masking for analysis
      top_positive_all = top_positive
      top_negative_all = top_negative
      top_pos_info = None
      if len(top_positive) > 0:
        top_pos_info = {"feature_index": top_positive[0][0], "coefficient": float(top_positive[0][1] * cfg.scale), "correlation": top_positive[0][2], "frequency": top_positive[0][3], "stats": top_positive[0][4]}
      top_neg_info = None  
      if len(top_negative) > 0:
        top_neg_info = {"feature_index": top_negative[0][0], "coefficient": float(top_negative[0][1] * cfg.scale), "correlation": top_negative[0][2], "frequency": top_negative[0][3], "stats": top_negative[0][4]}
      if cfg.pos and top_pos_info:
        selected = top_pos_info
      elif cfg.neg and top_neg_info:
        selected = {"feature_index": top_neg_info["feature_index"], "coefficient": -abs(top_neg_info["coefficient"]), "correlation": top_neg_info["correlation"], "frequency": top_neg_info["frequency"], "stats": top_neg_info["stats"]}
      elif top_pos_info and top_neg_info:
        if abs(top_pos_info["correlation"]) >= abs(top_neg_info["correlation"]):
          selected = top_pos_info
        else:
          selected = {"feature_index": top_neg_info["feature_index"], "coefficient": -abs(top_neg_info["coefficient"]), "correlation": top_neg_info["correlation"], "frequency": top_neg_info["frequency"], "stats": top_neg_info["stats"]}
      elif top_pos_info:
        selected = top_pos_info
      elif top_neg_info:
        selected = {"feature_index": top_neg_info["feature_index"], "coefficient": -abs(top_neg_info["coefficient"]), "correlation": top_neg_info["correlation"], "frequency": top_neg_info["frequency"], "stats": top_neg_info["stats"]}
      else:
        print(f"Warning: No valid features found for layer {layer} (expected for shuffle_labels)")
        continue
      layer_results[str(layer)] = {
        "top_positive": [{"feature_index": idx, "coefficient": float(coeff * cfg.scale), "correlation": corr, "frequency": freq, "stats": stats} 
                        for idx, coeff, corr, freq, stats in top_positive_all],
        "top_negative": [{"feature_index": idx, "coefficient": float(coeff * cfg.scale), "correlation": corr, "frequency": freq, "stats": stats} 
                        for idx, coeff, corr, freq, stats in top_negative_all],
        "selected": selected
      }
      pos_str = f"pos {top_positive[0][0]} r={top_positive[0][2]:.4f}" if len(top_positive) > 0 else "pos None"
      neg_str = f"neg {top_negative[0][0]} r={top_negative[0][2]:.4f}" if len(top_negative) > 0 else "neg None"
      print(f"Layer {layer}: {pos_str}, {neg_str}")
    if not layer_results:
      print("Warning: No valid features found in any layer. This is expected for shuffle_labels control experiment.")
      layer_results["_no_features"] = True
    if self.config.eval and self.snapshots:
      for ckpt, snap in self.snapshots.items():
        try:
          eval_controller = EvalController()
          snap_layer_results = {k: v for k, v in snap.items() if str(k).isdigit()}
          eval_stats = eval_controller.fixed_feature(
            layer_results=snap_layer_results,
            model=cfg.model,
            task=cfg.task,
            select_token=cfg.select_token,
            decode=cfg.decode,
            llm=self.llm,
            tokenizer=self.tokenizer,
            output_dir=cfg.output_dir,
            category=cfg.category,
            filter_value=cfg.filter_value,
            few=cfg.few,
            mask=cfg.mask,
            example=cfg.example,
            checkpoint=str(ckpt),
          )
          snap["accuracy"] = eval_stats.accuracy
        except Exception as e:
          snap["accuracy"] = None
          snap["error"] = str(e)
      try:
        last_step = max(int(k) for k in self.snapshots.keys() if str(k).isdigit())
        last_snap = self.snapshots[str(last_step)]
        last_selected = {}
        for k, v in last_snap.items():
          if str(k).isdigit() and isinstance(v, dict) and "selected" in v:
            last_selected[str(k)] = v["selected"].get("feature_index")
      except Exception:
        last_selected = {}
      for _, snap in self.snapshots.items():
        curr_selected = {}
        for k, v in snap.items():
          if str(k).isdigit() and isinstance(v, dict) and "selected" in v:
            curr_selected[str(k)] = v["selected"].get("feature_index")
        matched = 0
        if last_selected:
          for layer_key, feat_idx in curr_selected.items():
            try:
              if layer_key in last_selected and int(feat_idx) == int(last_selected[layer_key]):
                matched += 1
            except Exception:
              continue
        snap["matched_count"] = matched
        snap["total_layers"] = len(self.layers)
    if cfg.layer == "global":
      if cfg.pos or cfg.neg:
        best_correlation = -1
        best_layer = None
        best_feature = None
        for layer in layers:
          selected = layer_results[str(layer)]["selected"]
          correlation = abs(selected["correlation"])
          if correlation > best_correlation:
            best_correlation = correlation
            best_layer = layer
            best_feature = selected
        layer_results[str(best_layer)]["selected"] = best_feature
        steering_layers = layers
        feat_type = "positive" if best_feature["coefficient"] > 0 else "negative"
        if cfg.validate:
          baseline_acc = self._validate_baseline(train_loader, val_loader, max_new_tokens, layers)
          best_accuracy, feature_info = self._validate_layer_features(
            best_layer, layer_results[str(best_layer)], train_loader, val_loader, max_new_tokens
          )
          if isinstance(self.config.layer, int) and self.config.stable:
            best_feature = feature_info
            layer_results[str(best_layer)]["selected"] = best_feature
      elif cfg.validate:
        print(f"Validating layers on validation set ({val_loader.n_samples} samples)...")
        baseline_acc = self._validate_baseline(train_loader, val_loader, max_new_tokens, layers)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        best_accuracy = -1
        best_layer = None
        best_feature = None
        for layer in layers:
          accuracy, feature_info = self._validate_layer_features(layer, layer_results[str(layer)], train_loader, val_loader, max_new_tokens)
          if isinstance(self.config.layer, int) and self.config.stable:
            pos_feat = feature_info["positive"]
            neg_feat = feature_info["negative"]
            print(f"Layer {layer}: Using pos feature {pos_feat['feature_index']} (coeff={pos_feat['coefficient']:.4f}, corr={pos_feat['correlation']:.4f}) + neg feature {neg_feat['feature_index']} (coeff={neg_feat['coefficient']:.4f}, corr={neg_feat['correlation']:.4f}) [{accuracy-baseline_acc:+.2f}%]")
          else:
            feat_type = "positive" if feature_info["coefficient"] > 0 else "negative"
            print(f"Layer {layer}: Using {feat_type} feature {feature_info['feature_index']} with coefficient {feature_info['coefficient']:.4f} (corr={feature_info['correlation']:.4f}) [{accuracy-baseline_acc:+.2f}%]")
          if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_layer = layer
            best_feature = feature_info
      else:
        print("Selecting best feature by correlation (no validation)...")
        best_abs_corr = -1
        best_layer = None
        best_feature = None
        for layer in layers:
          feature_info = self._select_best_feature_by_correlation(layer_results[str(layer)])
          abs_corr = abs(feature_info["correlation"])
          feat_type = "positive" if feature_info["coefficient"] > 0 else "negative"
          print(f"Layer {layer}: {feat_type} feature {feature_info['feature_index']} with coefficient {feature_info['coefficient']:.4f} (corr={feature_info['correlation']:.4f})")
          if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best_layer = layer
            best_feature = feature_info
        layer_results[str(best_layer)]["selected"] = best_feature
        steering_layers = layers
      if cfg.validate and isinstance(self.config.layer, int) and self.config.stable:
        best_layer_result = {
          "top_positive": layer_results[str(best_layer)]["top_positive"],
          "top_negative": layer_results[str(best_layer)]["top_negative"],
          "selected": best_feature
        }
        layer_results = {str(best_layer): best_layer_result}
        steering_layers = [best_layer]
        pos_feat = best_feature["positive"]
        neg_feat = best_feature["negative"]
        print(f"Global best: Layer {best_layer} using pos+neg features {pos_feat['feature_index']}+{neg_feat['feature_index']} with validation accuracy {best_accuracy:.2f}% [{best_accuracy-baseline_acc:+.2f}%]")
      else:
        layer_results[str(best_layer)]["selected"] = best_feature
        steering_layers = layers
        if cfg.validate:
          print(f"Global: using all {len(layers)} layers (selected per-layer features)")
        else:
          print(f"Global: using all {len(layers)} layers (selected per-layer features)")
    elif cfg.layer == "foreach":
      if cfg.validate:
        print(f"Validating layers on validation set ({val_loader.n_samples} samples)...")
        baseline_acc = self._validate_baseline(train_loader, val_loader, max_new_tokens, layers)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        validated_layers = []
        validated_results = {}
        for layer in layers:
          accuracy, feature_info = self._validate_layer_features(layer, layer_results[str(layer)], train_loader, val_loader, max_new_tokens)
          if accuracy > baseline_acc:
            validated_layers.append(layer)
            validated_results[str(layer)] = {
              "top_positive": layer_results[str(layer)]["top_positive"],
              "top_negative": layer_results[str(layer)]["top_negative"],
              "selected": feature_info
            }
            layer_results[str(layer)]["selected"] = feature_info
            feat_type = "positive" if feature_info["coefficient"] > 0 else "negative"
            print(f"Layer {layer}: Using {feat_type} feature {feature_info['feature_index']} with coefficient {feature_info['coefficient']:.4f} (corr={feature_info['correlation']:.4f}) [{accuracy - baseline_acc:+.2f}%]")
          else:
            print(f"Layer {layer}: {accuracy - baseline_acc:+.2f}% (filtered)")
        steering_layers = validated_layers
        layer_results = validated_results
        print(f"Selected {len(validated_layers)} layers: {validated_layers}")
      elif cfg.pos or cfg.neg:
        steering_layers = layers
        for layer in layers:
          selected = layer_results[str(layer)]["selected"]
          feat_type = "positive" if selected["coefficient"] > 0 else "negative"
          print(f"Layer {layer}: Using {feat_type} feature {selected['feature_index']} with correlation {selected['correlation']:.4f}")
      else:
        print("Selecting best features by correlation for each layer (no validation)...")
        steering_layers = layers
        for layer in layers:
          feature_info = self._select_best_feature_by_correlation(layer_results[str(layer)])
          layer_results[str(layer)]["selected"] = feature_info
          feat_type = "positive" if feature_info["coefficient"] > 0 else "negative"
          print(f"Layer {layer}: {feat_type} feature {feature_info['feature_index']} with coefficient {feature_info['coefficient']:.4f} (corr={feature_info['correlation']:.4f})")
        print(f"Selected all {len(layers)} layers: {layers}")
    else:
      steering_layers = layers
      if len(layers) == 1:
        layer = layers[0]
        print(f"Single layer mode: Layer {layer}")
        print(f"Validating layer {layer} on validation set ({val_loader.n_samples} samples)...")
        baseline_acc = self._validate_baseline(train_loader, val_loader, max_new_tokens, layers)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        selected = layer_results[str(layer)]["selected"]
        feat_idx = selected["feature_index"]
        coeff = selected["coefficient"]
        accuracy = self._validate_feature(layer, feat_idx, coeff, train_loader, val_loader, max_new_tokens, collect_examples=True)
        feat_type = "positive" if selected["coefficient"] > 0 else "negative"
        print(f"Layer {layer}: Using {feat_type} feature {feat_idx} with coefficient {coeff:.4f} (corr={selected['correlation']:.4f}) accuracy={accuracy:.2f}% [{accuracy-baseline_acc:+.2f}%]")
    os.makedirs(cfg.output_dir, exist_ok=True)
    filename_parts = [f"{cfg.model}_{cfg.task}"]
    if isinstance(cfg.layer, int):
      filename_parts.append(str(cfg.layer))
    else:
      filename_parts.append(cfg.layer)
    # Add method identifier to prevent overwriting
    if cfg.mi:
      filename_parts.append("mi")
    elif cfg.fisher:
      filename_parts.append("fisher")
    elif cfg.caa:
      filename_parts.append("caa")
    else:
      filename_parts.append("corr")  # default CorrSteer
    if cfg.scale != 1.0:
      filename_parts.append(f"s{cfg.scale}")
    if cfg.pool != "max":
      filename_parts.append(cfg.pool)
    if cfg.steer_pool != "max":
      filename_parts.append(cfg.steer_pool)
    if cfg.mask != "generation":
      filename_parts.append(cfg.mask)
    if cfg.few is not None:
      filename_parts.append(f"f{cfg.few}")
    if cfg.category is not None:
      filename_parts.append(cfg.category)
    if cfg.filter_value is not None:
      filename_parts.append(cfg.filter_value)
    if cfg.decode:
      filename_parts.append("decode")
    if cfg.select_token:
      filename_parts.append("select")
    if cfg.raw:
      filename_parts.append("raw")
    if getattr(cfg, "reverse", False):
      filename_parts.append("reverse")
    save_path = os.path.join(cfg.output_dir, f"{'_'.join(filename_parts)}.json")
    payload = {
      "model": cfg.model,
      "task": cfg.task,
      "mode": cfg.layer,
      "steers": steering_layers,
      "results": layer_results,
      "samples": cfg.num_samples,
      "pool": cfg.pool,
      "steer_pool": cfg.steer_pool,
      "progress": self.snapshots if getattr(cfg, "eval", False) else None,
    }
    with open(save_path, "w") as f:
      json.dump(payload, f, indent=2)
    print(f"CorrSteer ({cfg.layer}) saved to {save_path}")
    if getattr(cfg, "eval", False) and self.snapshots:
      progress_path = save_path.replace(".json", "_progress.json")
      with open(progress_path, "w") as f:
        json.dump(self.snapshots, f, indent=2)
      print(f"Progress saved to {progress_path}")
    eval_controller = EvalController()
    selected_layers = layer_results 
    if cfg.layer == "global":
      selected_layers = {str(k): {"selected": layer_results[str(k)]["selected"]} for k in layers}
    elif cfg.layer == "foreach":
      selected_layers = {str(k): {"selected": layer_results[str(k)]["selected"]} for k in steering_layers}
    eval_stats = eval_controller.fixed_feature(
      layer_results=selected_layers,
      model=cfg.model,
      task=cfg.task,
      select_token=cfg.select_token,
      decode=cfg.decode,
      llm=self.llm,
      tokenizer=self.tokenizer,
      saes=self.saes,
      dtype=dtype,
      output_dir=os.path.dirname(save_path),
      category=cfg.category,
      filter_value=cfg.filter_value,
      few=cfg.few,
      mask=cfg.mask,
      example=cfg.example,
      cot=cfg.cot,
      raw=cfg.raw,
      reverse=cfg.reverse,
    )
    if cfg.layer == "global" and cfg.example:
      try:
        best_layer_eval = None
        best_abs_corr_eval = -1.0
        for lyr in layers:
          try:
            sel = layer_results[str(lyr)]["selected"]
            ac = abs(float(sel.get("correlation", 0.0)))
            if ac > best_abs_corr_eval:
              best_abs_corr_eval = ac
              best_layer_eval = int(lyr)
          except Exception:
            continue
        if best_layer_eval is not None and str(best_layer_eval) in layer_results:
          selected_layers_best = {str(best_layer_eval): {"selected": layer_results[str(best_layer_eval)]["selected"]}}
          _ = eval_controller.fixed_feature(
            layer_results=selected_layers_best,
            model=cfg.model,
            task=cfg.task,
            select_token=cfg.select_token,
            decode=cfg.decode,
            llm=self.llm,
            tokenizer=self.tokenizer,
            saes=self.saes,
            dtype=dtype,
            output_dir=os.path.dirname(save_path),
            category=cfg.category,
            filter_value=cfg.filter_value,
            few=cfg.few,
            mask=cfg.mask,
            example=cfg.example,
            cot=cfg.cot,
            reverse=cfg.reverse,
          )
      except Exception:
        pass
    if cfg.layer == "foreach" and cfg.example:
      try:
        global_layer_results = {}
        all_layers_selected: dict[str, dict] = {}
        for lyr in layers:
          try:
            top_pos, top_neg = self.accumulators[lyr].top_features(1)
            choice = None
            if len(top_pos) > 0 and len(top_neg) > 0:
              choice = top_pos[0] if abs(top_pos[0][2]) >= abs(top_neg[0][2]) else top_neg[0]
            elif len(top_pos) > 0:
              choice = top_pos[0]
            elif len(top_neg) > 0:
              choice = top_neg[0]
            if choice is None:
              continue
            idx, coeff, r, freq, stats = choice
            sel = {
              "feature_index": int(idx),
              "coefficient": float(coeff * cfg.scale) if r >= 0 else -abs(float(coeff * cfg.scale)),
              "correlation": float(r),
              "frequency": float(freq),
              "stats": stats,
            }
            all_layers_selected[str(lyr)] = {"selected": sel}
            top_positive_all, top_negative_all = self.accumulators[lyr].top_features_all(cfg.topk)
            global_layer_results[str(lyr)] = {
              "top_positive": [{"feature_index": idx, "coefficient": float(coeff * cfg.scale), "correlation": corr, "frequency": freq, "stats": stats} 
                              for idx, coeff, corr, freq, stats in top_positive_all],
              "top_negative": [{"feature_index": idx, "coefficient": float(coeff * cfg.scale), "correlation": corr, "frequency": freq, "stats": stats} 
                              for idx, coeff, corr, freq, stats in top_negative_all],
              "selected": sel
            }
          except Exception:
            continue
        if global_layer_results:
          global_filename_parts = [f"{cfg.model}_{cfg.task}", "global"]
          if cfg.scale != 1.0:
            global_filename_parts.append(f"s{cfg.scale}")
          if cfg.pool != "max":
            global_filename_parts.append(cfg.pool)
          if cfg.steer_pool != "max":
            global_filename_parts.append(cfg.steer_pool)
          if cfg.mask != "generation":
            global_filename_parts.append(cfg.mask)
          if cfg.few is not None:
            global_filename_parts.append(f"f{cfg.few}")
          if cfg.category is not None:
            global_filename_parts.append(cfg.category)
          if cfg.filter_value is not None:
            global_filename_parts.append(cfg.filter_value)
          if cfg.decode:
            global_filename_parts.append("decode")
          if cfg.select_token:
            global_filename_parts.append("select")
          global_save_path = os.path.join(cfg.output_dir, f"{'_'.join(global_filename_parts)}.json")
          global_payload = {
            "model": cfg.model,
            "task": cfg.task,
            "mode": "global",
            "steers": list(global_layer_results.keys()),
            "results": global_layer_results,
            "samples": cfg.num_samples,
            "pool": cfg.pool,
            "steer_pool": cfg.steer_pool,
          }
          with open(global_save_path, "w") as f:
            json.dump(global_payload, f, indent=2)
          print(f"Global features saved to {global_save_path}")
          global_features_path = global_save_path.replace(".json", "_features.json")
          model_cfg = model_config[cfg.model]
          print(f"Global features analysis saved to {global_features_path}")
        if all_layers_selected:
          _ = eval_controller.fixed_feature(
            layer_results=all_layers_selected,
            model=cfg.model,
            task=cfg.task,
            select_token=cfg.select_token,
            decode=cfg.decode,
            llm=self.llm,
            tokenizer=self.tokenizer,
            saes=self.saes,
            dtype=dtype,
            output_dir=os.path.dirname(save_path),
            category=cfg.category,
            filter_value=cfg.filter_value,
            few=cfg.few,
            mask=cfg.mask,
            example=cfg.example,
            cot=cfg.cot,
            name_suffix="_global",
            reverse=cfg.reverse,
          )
        best_layer_eval = None
        best_abs_corr_eval = -1.0
        best_sel = None
        for lyr in layers:
          try:
            top_pos, top_neg = self.accumulators[lyr].top_features(1)
            candidates = []
            if len(top_pos) > 0:
              candidates.append(top_pos[0])
            if len(top_neg) > 0:
              candidates.append(top_neg[0])
            for idx, coeff, r, freq, stats in candidates:
              if abs(r) > best_abs_corr_eval:
                best_abs_corr_eval = abs(r)
                best_layer_eval = int(lyr)
                best_sel = {
                  "feature_index": int(idx),
                  "coefficient": float(coeff * cfg.scale) if r >= 0 else -abs(float(coeff * cfg.scale)),
                  "correlation": float(r),
                  "frequency": float(freq),
                  "stats": stats,
                }
          except Exception:
            continue
        if best_layer_eval is not None and best_sel is not None:
          single_selected = {str(best_layer_eval): {"selected": best_sel}}
          _ = eval_controller.fixed_feature(
            layer_results=single_selected,
            model=cfg.model,
            task=cfg.task,
            select_token=cfg.select_token,
            decode=cfg.decode,
            llm=self.llm,
            tokenizer=self.tokenizer,
            saes=self.saes,
            dtype=dtype,
            output_dir=os.path.dirname(save_path),
            category=cfg.category,
            filter_value=cfg.filter_value,
            few=cfg.few,
            mask=cfg.mask,
            example=cfg.example,
            cot=cfg.cot,
            name_suffix="_single",
            raw=cfg.raw,
            reverse=cfg.reverse,
          )
      except Exception:
        pass
    accuracy_path = save_path.replace(".json", "_accuracy.json")
    accuracy_result = {
      "model": cfg.model,
      "task": cfg.task,
      "mode": cfg.layer,
      "steers": steering_layers,
      "accuracy": eval_stats.accuracy,
      "samples": cfg.num_samples,
      "output": eval_stats.output_json
    }
    with open(accuracy_path, 'w') as f:
      json.dump(accuracy_result, f, indent=2)
    print(f"Evaluation accuracy saved to {accuracy_path} (accuracy={(eval_stats.accuracy*100):.2f}%)")
    return {
      "path": save_path,
      "mode": cfg.layer,
      "steers": steering_layers,
      "results": layer_results,
      "accuracy": eval_stats.accuracy,
      "output": eval_stats.output_json,
    }


if __name__ == "__main__":
  fire.Fire(CorrSteerController, serialize=lambda x: None)
