import os
import json
import math
from typing import Optional, List, Tuple, Dict, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from tqdm import tqdm
import fire

from corrsteer.utils import (
  load_model_tokenizer,
  load_dataloaders,
  get_device,
  fix_seed,
  build_prompt,
  extract_answer,
  get_supervised_pair,
)
from corrsteer.utils import generate_options, get_logit_processor
from corrsteer.config import dataset_config, model_config, calculate_reward


class SFTConfig(BaseModel):
  num_samples: int = 4000
  batch_size: Optional[int] = None
  model: str = "gemma2b"
  task: str = "mmlu"
  seed: int = 42
  dtype: str = "bfloat16"
  output_dir: str = "checkpoints"
  few: Optional[int] = None
  category: Optional[str] = None
  filter_value: Optional[str] = None
  select_token: bool = False
  limit: Optional[int] = 200
  cot: bool = False
  lr: float = 1e-5
  epochs: int = 1


class SFTController:
  def __init__(self):
    self.device = get_device()
    self.llm = None
    self.tokenizer = None
    self.cfg: SFTConfig
    self.batch_size: int

  def _name_stem(self) -> str:
    parts = [self.cfg.model, self.cfg.task]
    if self.cfg.filter_value is not None:
      parts.append(self.cfg.filter_value)
    if self.cfg.few is not None:
      parts.append(f"f{self.cfg.few}")
    if getattr(self.cfg, "select_token", False):
      parts.append("select")
    if self.cfg.cot:
      parts.append("cot")
    parts.append("sft")
    return "_".join(parts)

  def _baseline_eval_name(self) -> Tuple[str, str]:
    name = f"{self.cfg.model}_{self.cfg.task}_eval"
    if self.cfg.filter_value is not None:
      name = f"{self.cfg.model}_{self.cfg.task}_{self.cfg.filter_value}_eval"
    if self.cfg.few is not None:
      name += f"_f{self.cfg.few}"
    if getattr(self.cfg, "select_token", False):
      name += "_select"
    if self.cfg.cot:
      name += "_cot"
    json_path = os.path.join("output", f"{name}.json")
    stats_path = os.path.join("output", f"{name}_stats.json")
    return json_path, stats_path

  def _load_baseline(self) -> Tuple[Optional[List[Tuple[str, str, str]]], Optional[float]]:
    base_eval = f"{self.cfg.model}_{self.cfg.task}_eval"
    if self.cfg.filter_value is not None:
      base_eval = f"{self.cfg.model}_{self.cfg.task}_{self.cfg.filter_value}_eval"
    f1_suffix = "_f1" if self.cfg.few is not None else ""
    select_suffix = "_select" if getattr(self.cfg, "select_token", False) else ""
    cot_suffix = "_cot" if self.cfg.cot else ""
    json_path = os.path.join("output", f"{base_eval}{f1_suffix}{select_suffix}{cot_suffix}.json")
    try:
      with open(json_path, "r") as f:
        data = json.load(f)
      results: List[Tuple[str, str, str]] = []
      for item in data:
        prompt = item.get("prompt", "")
        gt = item.get("ground_truth", "")
        pred = item.get("predicted", "")
        results.append((prompt, gt, pred))
      return results, None
    except Exception:
      stats_path = os.path.join("output", f"{base_eval}{f1_suffix}{select_suffix}{cot_suffix}_stats.json")
      try:
        with open(stats_path, "r") as f:
          stats_data = json.load(f)
        entry = stats_data[0] if isinstance(stats_data, list) and stats_data else stats_data
        if isinstance(entry, dict):
          pointer = entry.get("output_json")
          if isinstance(pointer, str):
            try:
              with open(pointer, "r") as pf:
                data = json.load(pf)
              results: List[Tuple[str, str, str]] = []
              for item in data:
                prompt = item.get("prompt", "")
                gt = item.get("ground_truth", "")
                pred = item.get("predicted", "")
                results.append((prompt, gt, pred))
              return results, None
            except Exception:
              pass
          acc = entry.get("accuracy")
          if isinstance(acc, (int, float)):
            print(f"Loaded baseline accuracy from stats: {acc:.4f} (no per-sample predictions available)")
            return None, float(acc)
      except Exception:
        pass
      return None, None

  @torch.no_grad()
  def _predict_batch(self, llm, tokenizer, prompts: List[str]) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(self.device)
    attention_mask = inputs.attention_mask.to(self.device)
    option_tokens = None
    processor = None
    if getattr(self.cfg, "select_token", False):
      option_tokens = generate_options(tokenizer, self.cfg.task)
      processor = get_logit_processor(option_tokens)
    max_new_tokens = dataset_config[self.cfg.task].max_new_tokens
    generated_ids = cast(torch.Tensor, llm.generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      logits_processor=processor,
      pad_token_id=tokenizer.pad_token_id,
      eos_token_id=tokenizer.eos_token_id,
    ))
    offsets = [len(input_ids[i]) for i in range(len(input_ids))]
    sliced = [generated_ids[i, off:] for i, off in enumerate(offsets)]
    texts = tokenizer.batch_decode(sliced, skip_special_tokens=True)
    preds: List[str] = []
    for t in texts:
      if self.cfg.cot:
        _, p = extract_answer(t, self.cfg.task)
      else:
        _, p = extract_answer(t, self.cfg.task)
      preds.append(p)
    return preds

  def _evaluate(self, llm, tokenizer, loader) -> Tuple[float, List[Tuple[str, str, str]]]:
    results: List[Tuple[str, str, str]] = []
    total_reward = 0.0
    total = 0
    batch_samples: List = []
    for sample in tqdm(loader, total=loader.n_samples, desc="Evaluating"):
      batch_samples.append(sample)
      if len(batch_samples) == self.batch_size:
        prompts, gts = zip(*[build_prompt(s, self.cfg.task, cot=self.cfg.cot, few_shots=None) for s in batch_samples])
        preds = self._predict_batch(llm, tokenizer, list(prompts))
        for pred, gt, pr in zip(preds, gts, prompts):
          reward = calculate_reward(pred, gt, self.cfg.task, tokenizer)
          total_reward += reward
          total += 1
          results.append((pr, gt, pred))
        batch_samples = []
    if batch_samples:
      prompts, gts = zip(*[build_prompt(s, self.cfg.task, cot=self.cfg.cot, few_shots=None) for s in batch_samples])
      preds = self._predict_batch(llm, tokenizer, list(prompts))
      for pred, gt, pr in zip(preds, gts, prompts):
        reward = calculate_reward(pred, gt, self.cfg.task, tokenizer)
        total_reward += reward
        total += 1
        results.append((pr, gt, pred))
    acc = float(total_reward / max(total, 1))
    return acc, results

  def _sft_epoch(self, model, tokenizer, train_loader, steps: int, optim, scheduler, max_grad_norm: float = 1.0) -> int:
    model.train()
    step_count = 0
    for sample in tqdm(train_loader, desc="SFT training"):
      prompt, target = get_supervised_pair(sample, self.cfg.task, cot=self.cfg.cot, few_shots=None)
      # Tokenize prompt and target separately (no special tokens), then concatenate
      delim = "\n" if self.cfg.task != "select" else ""
      prompt_with_delim = prompt + delim
      enc_prompt = tokenizer(prompt_with_delim, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
      enc_target = tokenizer(target, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
      prompt_ids = enc_prompt.input_ids.to(self.device)
      target_ids = enc_target.input_ids.to(self.device)
      input_ids = torch.cat([prompt_ids, target_ids], dim=1)
      attention_mask = torch.ones_like(input_ids, device=self.device)
      # Create labels: ignore prompt tokens, supervise exact target span
      labels = torch.full_like(input_ids, -100)
      labels[:, prompt_ids.shape[1]:] = target_ids
      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      optim.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
      optim.step()
      if scheduler is not None:
        scheduler.step()
      step_count += 1
      if step_count >= steps:
        break
    return step_count

  def train(self, **kwargs) -> Dict[str, object]:
    cfg = SFTConfig(**kwargs)
    self.cfg = cfg
    fix_seed(cfg.seed)
    dtype = getattr(torch, cfg.dtype) if isinstance(cfg.dtype, str) else cfg.dtype
    tokenizer, llm = load_model_tokenizer(cfg.model, dtype=dtype)
    self.tokenizer = tokenizer
    self.llm = llm
    train_loader, _, test_loader = load_dataloaders(
      cfg.task,
      category=cfg.category,
      filter_value=cfg.filter_value,
      seed=cfg.seed,
    )
    self.batch_size = cfg.batch_size if cfg.batch_size else model_config[cfg.model].batch_size
    os.makedirs(cfg.output_dir, exist_ok=True)

    baseline_results, baseline_acc_from_stats = self._load_baseline()
    if baseline_results is None and baseline_acc_from_stats is None:
      print("Evaluating baseline...")
      baseline_acc, baseline_results = self._evaluate(llm, tokenizer, test_loader)
      print(f"Baseline accuracy: {baseline_acc * 100:.2f}%")
    else:
      if baseline_results is not None:
        total_reward = 0.0
        for _, gt, pred in baseline_results:
          total_reward += calculate_reward(pred, gt, cfg.task, tokenizer)
        baseline_acc = total_reward / max(len(baseline_results), 1)
        print(f"Loaded baseline from predictions (accuracy: {baseline_acc * 100:.2f}%)")
      else:
        baseline_acc = float(baseline_acc_from_stats)
        print(f"Loaded baseline from stats only (accuracy: {baseline_acc * 100:.2f}%)")

    print("SFT training...")
    N_train = getattr(train_loader, 'n_samples', None) or 0
    steps_per_epoch = min(int(cfg.num_samples), int(N_train) if N_train else int(cfg.num_samples))
    epochs = max(1, int(math.ceil(cfg.num_samples / max(N_train, 1)))) if N_train else max(1, cfg.epochs)
    total_steps = epochs * steps_per_epoch
    eff_lr = cfg.lr
    if (N_train and N_train < 2000) or total_steps < 2000:
      eff_lr = min(eff_lr, 5e-6)
    trainable_params = [p for p in llm.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=eff_lr, weight_decay=0.01)
    warmup_steps = max(100, int(0.03 * total_steps)) if total_steps > 0 else 0
    def lr_lambda(current_step: int):
      if total_steps <= 0:
        return 1.0
      if current_step < warmup_steps and warmup_steps > 0:
        return float(current_step) / float(max(1, warmup_steps))
      progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
      return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    print(f"Training SFT for {epochs} epoch(s), {steps_per_epoch} steps/epoch, total_steps={total_steps}, warmup={warmup_steps}, lr={eff_lr}")
    progressed = 0
    for _ in range(epochs):
      progressed += self._sft_epoch(llm, tokenizer, train_loader, steps_per_epoch, optim, scheduler, max_grad_norm=1.0)

    print("Evaluating SFT model...")
    llm.eval()
    trained_acc, trained_results = self._evaluate(llm, tokenizer, test_loader)
    print(f"Trained accuracy: {trained_acc * 100:.2f}%")

    positives: List[Dict] = []
    negatives: List[Dict] = []
    if baseline_results is not None:
      print("Comparing with baseline (pos/neg changed samples)...")
      for (pr_b, gt_b, pred_b), (pr_s, gt_s, pred_s) in zip(baseline_results, trained_results):
        if pred_b == pred_s:
          continue
        assert gt_b == gt_s
        b_reward = calculate_reward(pred_b, gt_b, cfg.task, tokenizer)
        s_reward = calculate_reward(pred_s, gt_s, cfg.task, tokenizer)
        item = {
          "prompt": pr_b,
          "ground_truth": gt_b,
          "predicted": pred_s,
          "baseline": pred_b,
          "baseline_reward": b_reward,
          "trained_reward": s_reward,
        }
        if s_reward > b_reward:
          positives.append(item)
        elif s_reward < b_reward:
          negatives.append(item)
    else:
      print("Skipping per-sample comparison because only stats baseline is available.")

    name_stem = self._name_stem()
    pos_path = os.path.join(cfg.output_dir, f"{name_stem}_positive.json")
    neg_path = os.path.join(cfg.output_dir, f"{name_stem}_negative.json")
    with open(pos_path, "w") as f:
      json.dump(positives, f, indent=2)
    with open(neg_path, "w") as f:
      json.dump(negatives, f, indent=2)
    print(f"Positive examples saved to {pos_path}")
    print(f"Negative examples saved to {neg_path}")

    pos_n = len(positives)
    neg_n = len(negatives)
    denom = pos_n + neg_n
    ser_val = (neg_n / denom) if denom > 0 else None
    metrics = {
      "model": cfg.model,
      "task": cfg.task,
      "acc": trained_acc,
      "ser": ser_val,
      "pos_count": pos_n,
      "neg_count": neg_n,
      "total_changed": denom,
      "baseline_acc": baseline_acc,
      "steps": total_steps,
    }
    metrics_path = os.path.join(cfg.output_dir, f"{name_stem}_metrics.json")
    with open(metrics_path, "w") as f:
      json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return {
      "trained_accuracy": trained_acc,
      "baseline_accuracy": baseline_acc,
      "pos_examples": pos_path,
      "neg_examples": neg_path,
      "metrics": metrics_path,
    }


if __name__ == "__main__":
  fire.Fire(SFTController, serialize=lambda x: None)
