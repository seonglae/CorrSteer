
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from typing import Optional, cast, Dict, Union, List, Literal, Tuple

from tqdm import tqdm
import fire
from sae_lens import SAE
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from corrsteer.steer import SteeringHook
from corrsteer.config import dataset_config, model_config, calculate_reward
from corrsteer.utils import (
  load_model_tokenizer,
  build_prompt,
  get_device,
  fix_seed,
  load_dataloaders,
  generate_options,
  load_sae,
  get_dims,
  get_logit_processor,
  extract_answer,
)
from corrsteer.steer import get_steering_hook
from corrsteer.model import EvalResult
from corrsteer.dataset import SampleData


device = get_device()

class FixedFeaturePolicyNetwork(nn.Module):
  def __init__(self, feature_dim: int, feature_index: int, coefficient: float, raw: bool = False):
    super().__init__()
    self.feature_dim = feature_dim
    self.feature_index = feature_index
    self.coefficient = coefficient
    self.raw = raw
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = obs.shape
    action = torch.zeros(batch_size, seq_len, self.feature_dim, device=obs.device, dtype=obs.dtype)
    action[:, :, self.feature_index] = self.coefficient
    return action
  def select_action(self, observation: torch.Tensor, mask_features: Optional[torch.Tensor] = None):
    return self.forward(observation)


class EvalPrediction(BaseModel):
  prompt: str
  think: Optional[str] = None
  ground_truth: str
  predicted: str
  few_shots: Optional[List[SampleData]] = None

class EvalController:
  def __init__(self):
    self.policy_nets: Dict[int, nn.Module] = {}
    self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    self.layers: list[int] = []
    self.saes: Dict[int, SAE] = {}
    self.hooks: Dict[int, SteeringHook] = {}
    self.select_token: bool = False
    self.decode: bool = False
    self.multiple: int = 1
    self.cot: bool = False
    self.few_shots: Optional[List[SampleData]] = None
    self.mask: Literal['generation', 'all'] = 'generation'

  def get_prediction_result(
    self,
    prompt: str,
    think: Optional[str],
    predicted_label: str,
    correct_answer: str,
  ) -> tuple[EvalPrediction, float]:
    result_dict: EvalPrediction = EvalPrediction(
      prompt=prompt,
      ground_truth=correct_answer,
      think=think,
      predicted=predicted_label,
    )
    reward = calculate_reward(predicted_label, correct_answer, self.task, self.tokenizer)
    if getattr(self, "reverse", False):
      reward = 1.0 - float(reward)
    return result_dict, reward


  def generate_steered(self, batch_samples, max_new_tokens: int, select_token: bool, option_tokens) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Set up steering hooks and generate text"""
    return self.generate_with_layers(batch_samples, self.layers, max_new_tokens, select_token, option_tokens)

  def generate_with_layers(self, batch_samples, layers: list[int], max_new_tokens: int, select_token: bool, option_tokens) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Generate text with hooks applied for given layers"""
    prompts, correct_answers = zip(
      *[build_prompt(sample, self.task, cot=self.cot, few_shots=self.few_shots) for sample in batch_samples]
    )
    correct_answers = cast(list[str], correct_answers)
    prompts = cast(list[str], prompts)
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    self.hook_handles = []
    for layer in layers:
      lastk = 1
      hook = get_steering_hook(
        self.policy_nets[layer],
        self.saes[layer],
        decode=self.decode if layers and layer == layers[0] else False,
        lastk=lastk,
        multiple=self.multiple,
        mask=self.mask,
        raw=getattr(self.policy_nets[layer], 'raw', False),
      )
      self.hooks[layer] = hook
      hook_handle = self.llm.model.layers[layer].register_forward_pre_hook(hook)
      self.hook_handles.append(hook_handle)
    processor = get_logit_processor(option_tokens) if select_token else None
    generated_ids = cast(torch.Tensor, self.llm.generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      logits_processor=processor,
      pad_token_id=self.tokenizer.pad_token_id,
      eos_token_id=self.tokenizer.eos_token_id,
    ))
    for handle in self.hook_handles:
      handle.remove()
    return input_ids, generated_ids, correct_answers, prompts


  def get_stats(self, batch_samples, input_ids: torch.Tensor, generated_ids: torch.Tensor, correct_answers: list[str], prompts: list[str], unique_ids) -> tuple[list[EvalPrediction], float, int]:
    """CorrSteer: simplified stats without critic/feature tracking"""
    results = []
    total_reward = 0.0
    total = 0
    offsets = [len(input_ids[i]) for i in range(len(input_ids))]
    sliced = [generated_ids[i, offset:] for i, offset in enumerate(offsets)]
    generated_texts = self.tokenizer.batch_decode(sliced, skip_special_tokens=True)
    think = None
    for i in range(len(batch_samples)):
      if self.cot or dataset_config[self.task].type == "reason":
        think, predicted_label = extract_answer(generated_texts[i], self.task)
      else:
        generated_id = generated_ids[i, len(input_ids[i])]  # First generated token after input
        if unique_ids is not None:
          if int(generated_id) not in unique_ids:
            unique_ids[int(generated_id)] = 0
          unique_ids[int(generated_id)] += 1
        _, predicted_label = extract_answer(generated_texts[i], self.task)
        think = None
      result_dict, reward = self.get_prediction_result(
        prompts[i],
        think,
        predicted_label,
        correct_answers[i],
      )
      results.append(result_dict)
      total_reward += reward
      total += 1
    return results, total_reward, total


  def process_batch(
    self, batch_samples, max_new_tokens, select_token, unique_ids, option_tokens
  ) -> tuple[list[EvalPrediction], float, int]:
    input_ids, generated_ids, correct_answers, prompts = self.generate_steered(
      batch_samples, max_new_tokens, select_token, option_tokens
    )
    return self.get_stats(
      batch_samples, input_ids, generated_ids, correct_answers, prompts, unique_ids
    )


  def evaluate_loop(self, batch_size, max_new_tokens, select_token=False) -> tuple[list[dict[str, Union[str, float, int]]], float, int]:
    results = []
    total_reward = 0.0
    total = 0
    batch_samples = []
    unique_ids = {}
    option_tokens = generate_options(self.tokenizer, self.task)

    for sample in tqdm(
      self.test_loader, total=self.test_loader.n_samples, desc="Evaluating"
    ):
      batch_samples.append(sample)
      if len(batch_samples) == batch_size:
        batch_results, batch_reward, batch_total = self.process_batch(
          batch_samples, max_new_tokens, select_token, unique_ids, option_tokens
        )
        batch_results_dict = [result.model_dump() for result in batch_results]
        results.extend(batch_results_dict)
        total_reward += batch_reward
        total += batch_total
        batch_samples = []

    if batch_samples:
      batch_results, batch_reward, batch_total = self.process_batch(
        batch_samples, max_new_tokens, select_token, unique_ids, option_tokens
      )
      batch_results_dict = [result.model_dump() for result in batch_results]
      results.extend(batch_results_dict)
      total_reward += batch_reward
      total += batch_total
    return results, total_reward, total


  def baseline(
    self,
    batch_size=None,
    limit: Optional[int] = None,
    max_new_tokens=1,
    seed=42,
    model="gemma2b",
    task="mmlu",
    select_token=False,
    dtype=torch.bfloat16,
    output_dir="output",
    category=None,
    filter_value=None,
    cot=False,
    few: Optional[int] = None,
  ) -> EvalResult:
    task_type = dataset_config[task].type
    if task_type != "select":
      select_token = False
    seed = fix_seed(seed)
    output_dir = f"{output_dir}_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    self.task = task
    self.layers = []
    self.cot = cot
    self.select_token = select_token
    
    batch_size = batch_size if batch_size else model_config[model].batch_size
    tokenizer, llm = load_model_tokenizer(model, dtype=dtype)
    train_loader, _, test_loader = load_dataloaders(
      task,
      seed=seed,
      test_limit=limit,
      val_limit=limit,
      category=category,
      filter_value=filter_value,
    )
    self.few_shots = None
    if few is not None and few > 0:
      self.few_shots = train_loader.get_last_samples(few)
    print(f"Few shots: {self.few_shots}")
    max_new_tokens = dataset_config[task].max_new_tokens
    self.test_loader = test_loader
    self.llm = llm
    self.tokenizer = tokenizer
    results, total_reward, total = self.evaluate_loop(
      batch_size, max_new_tokens, select_token=select_token
    )
    accuracy = total_reward / total
    print(f"Final {task} Accuracy: {accuracy * 100:.2f}%")
    df = pd.DataFrame(results)
    name = (
      f"{model}_{task}_eval"
      if filter_value is None
      else f"{model}_{task}_{filter_value}_eval"
    )
    if limit is not None:
      name += f"_l{limit}"
    if few is not None:
      name += f"_f{few}"
    if select_token:
      name += "_select"
    if self.decode:
      name += "_decode"
    if cot:
      name += "_cot"
    output = os.path.join(output_dir, f"{name}.json")
    df.to_json(output, orient="records", indent=2)
    print(f"Results saved to {output}")
    stats = {
      "checkpoint": None,
      "model": model,
      "task": task,
      "layers": [],
      "accuracy": accuracy,
      "category": category,
      "limit": limit,
      "select_token": select_token,
      "decode": self.decode,
      "cot": self.cot,
      "output_json": output,
    }
    stats_output = os.path.join(output_dir, f"{name}_stats.json")
    try:
      with open(stats_output, "r") as f:
        all_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      all_stats = []
    all_stats.append(stats)
    with open(stats_output, "w") as f:
      json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_output}")
    return EvalResult(**stats)

  def load_baseline_from_output(self, model: str, task: str, output_dir: str = "output", 
                               filter_value: Optional[str] = None, few: Optional[int] = None,
                               select_token: bool = False, cot: bool = False) -> Optional[List[Tuple[str, str, str]]]:
    """Load baseline results from output directory"""
    baseline_name = f"{model}_{task}_eval"
    if filter_value is not None:
      baseline_name = f"{model}_{task}_{filter_value}_eval"
    if few is not None:
      baseline_name += f"_f{few}"
    if select_token:
      baseline_name += "_select"
    if cot:
      baseline_name += "_cot"
    baseline_json = os.path.join(output_dir, f"{baseline_name}.json")
    try:
      if os.path.exists(baseline_json):
        with open(baseline_json, "r") as f:
          baseline_data = json.load(f)
        baseline_results = []
        for item in baseline_data:
          prompt = item.get("prompt", "")
          gt = item.get("ground_truth", "")
          pred = item.get("predicted", "")
          baseline_results.append((prompt, gt, pred))
        print(f"Loaded baseline from {baseline_json}")
        return baseline_results
      else:
        print(f"Baseline file not found: {baseline_json}")
        return None
    except Exception as e:
      print(f"Failed to load baseline: {e}")
      return None

  def calculate_ser_with_baseline(self, baseline_results: List[Tuple[str, str, str]], 
                                 test_loader, steered_layers: List[int], batch_size: int,
                                 max_new_tokens: int, select_token: bool, option_tokens,
                                 output_dir: str, name: str) -> Tuple[List[dict], List[dict], float, float]:
    """Calculate SER using loaded baseline results"""
    positives: list[dict] = []
    negatives: list[dict] = []
    batch_samples = []
    baseline_idx = 0
    total_reward = 0.0
    total = 0
    for sample in tqdm(test_loader, total=test_loader.n_samples, desc="Collecting examples"):
      batch_samples.append(sample)
      if len(batch_samples) == batch_size:
        b_prompts = []
        b_preds = []
        b_gt_list = []
        for i in range(len(batch_samples)):
          if baseline_idx < len(baseline_results):
            b_prompt, b_gt, b_pred = baseline_results[baseline_idx]
            b_prompts.append(b_prompt)
            b_preds.append(b_pred)
            b_gt_list.append(b_gt)
            baseline_idx += 1
          else:
            break
        s_input_ids, s_generated_ids, s_correct_answers, s_prompts = self.generate_with_layers(
          batch_samples, steered_layers, max_new_tokens, select_token, option_tokens
        )
        s_offsets = [len(s_input_ids[i]) for i in range(len(s_input_ids))]
        s_sliced = [s_generated_ids[i, off:] for i, off in enumerate(s_offsets)]
        s_texts = self.tokenizer.batch_decode(s_sliced, skip_special_tokens=True)
        
        for i in range(len(batch_samples)):
          if self.cot or dataset_config[self.task].type == "reason":
            s_think, s_pred = extract_answer(s_texts[i], self.task)
          else:
            s_think, s_pred = None, extract_answer(s_texts[i], self.task)[1]
          b_pred = b_preds[i] if i < len(b_preds) else ""
          gt = b_gt_list[i] if i < len(b_gt_list) else ""
          s_reward = calculate_reward(s_pred, gt, self.task, self.tokenizer)
          if getattr(self, "reverse", False):
            s_reward = 1.0 - float(s_reward)
          total_reward += s_reward
          total += 1
          if b_pred != s_pred:
            b_reward = calculate_reward(b_pred, gt, self.task, self.tokenizer)
            if getattr(self, "reverse", False):
              b_reward = 1.0 - float(b_reward)
            base_item = {
              "prompt": b_prompts[i],
              "think": s_think if (self.cot or dataset_config[self.task].type == "reason") else None,
              "ground_truth": gt,
              "predicted": s_pred,
              "few_shots": self.few_shots,
            }
            item = dict(base_item)
            item["baseline"] = b_pred
            item["baseline_reward"] = b_reward
            item["steered_reward"] = s_reward
            if s_reward > b_reward:
              positives.append(item)
            elif s_reward < b_reward:
              negatives.append(item)
        batch_samples = []
    if batch_samples:
      b_prompts = []
      b_preds = []
      b_gt_list = []
      for i in range(len(batch_samples)):
        if baseline_idx < len(baseline_results):
          b_prompt, b_gt, b_pred = baseline_results[baseline_idx]
          b_prompts.append(b_prompt)
          b_preds.append(b_pred)
          b_gt_list.append(b_gt)
          baseline_idx += 1
        else:
          break
      
      s_input_ids, s_generated_ids, s_correct_answers, s_prompts = self.generate_with_layers(
        batch_samples, steered_layers, max_new_tokens, select_token, option_tokens
      )
      s_offsets = [len(s_input_ids[i]) for i in range(len(s_input_ids))]
      s_sliced = [s_generated_ids[i, off:] for i, off in enumerate(s_offsets)]
      s_texts = self.tokenizer.batch_decode(s_sliced, skip_special_tokens=True)
      
      for i in range(len(batch_samples)):
        if self.cot or dataset_config[self.task].type == "reason":
          s_think, s_pred = extract_answer(s_texts[i], self.task)
        else:
          s_think, s_pred = None, extract_answer(s_texts[i], self.task)[1]
        b_pred = b_preds[i] if i < len(b_preds) else ""
        gt = b_gt_list[i] if i < len(b_gt_list) else ""
        s_reward = calculate_reward(s_pred, gt, self.task, self.tokenizer)
        if getattr(self, "reverse", False):
          s_reward = 1.0 - float(s_reward)
        total_reward += s_reward
        total += 1
        if b_pred != s_pred:
          b_reward = calculate_reward(b_pred, gt, self.task, self.tokenizer)
          if getattr(self, "reverse", False):
            b_reward = 1.0 - float(b_reward)
          base_item = {
            "prompt": b_prompts[i],
            "think": s_think if (self.cot or dataset_config[self.task].type == "reason") else None,
            "ground_truth": gt,
            "predicted": s_pred,
            "few_shots": self.few_shots,
          }
          item = dict(base_item)
          item["baseline"] = b_pred
          item["baseline_reward"] = b_reward
          item["steered_reward"] = s_reward
          if s_reward > b_reward:
            positives.append(item)
          elif s_reward < b_reward:
            negatives.append(item)
    
    # Save results
    pos_output = os.path.join(output_dir, f"{name}_positive.json")
    neg_output = os.path.join(output_dir, f"{name}_negative.json")
    with open(pos_output, "w") as f:
      json.dump(positives, f, indent=2)
    with open(neg_output, "w") as f:
      json.dump(negatives, f, indent=2)
    print(f"Positive examples saved to {pos_output}")
    print(f"Negative examples saved to {neg_output}")
    pos_n = len(positives)
    neg_n = len(negatives)
    denom = pos_n + neg_n
    ser_val = (neg_n / denom) if denom > 0 else None
    if denom > 0:
      print(f"SER (side effect ratio) = {neg_n}/{denom} = {ser_val:.4f}")
    else:
      print("SER (side effect ratio) = N/A (no changed examples)")
    accuracy = total_reward / max(total, 1)
    return positives, negatives, ser_val, accuracy

  def fixed_feature(
    self,
    layer_results: Dict,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    max_new_tokens: Optional[int] = 1,
    model: str = "gemma2b",
    llm: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    task: str = "mmlu",
    select_token: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str = "output",
    category: Optional[str] = None,
    filter_value: Optional[str] = None,
    seed: int = 42,
    decode: bool = False,
    multiple: int = 1,
    cot: bool = False,
    few: Optional[int] = None,
    mask: str = 'generation',
    example: bool = False,
    checkpoint: Optional[str] = None,
    saes: Optional[Dict[int, SAE]] = None,
    name_suffix: Optional[str] = None,
    raw: bool = False,
    reverse: bool = False,
  ) -> EvalResult:
    task_type = dataset_config[task].type
    if task_type != "select":
      select_token = False
    seed = fix_seed(seed)
    self.task = task
    self.cot = cot
    self.multiple = multiple
    self.decode = decode
    self.mask = mask
    self.reverse = reverse
    batch_size = batch_size if batch_size else model_config[model].batch_size
    if llm is None or tokenizer is None:
      tokenizer, llm = load_model_tokenizer(model, dtype=dtype)
    train_loader, _, test_loader = load_dataloaders(
      task,
      seed=seed,
      test_limit=limit,
      val_limit=limit,
      category=category,
      filter_value=filter_value,
    )
    self.few_shots = None
    if few is not None and few > 0:
      self.few_shots = train_loader.get_last_samples(few)
    max_new_tokens = dataset_config[task].max_new_tokens
    self.layers = [int(k) for k in layer_results.keys()]
    self.saes = {}
    self.policy_nets = {}
    if saes is not None:
      for layer_id in self.layers:
        if layer_id in saes:
          self.saes[layer_id] = saes[layer_id]
    for layer_id in self.layers:
      if raw:
        # Raw mode: use hidden_dim directly
        self.saes[layer_id] = None
        feature_dim = llm.config.hidden_size
      else:
        # SAE mode: load SAE if not provided
        if layer_id not in self.saes:
          sae_loaded, _, _ = load_sae(model, layer_id, device)
          self.saes[layer_id] = sae_loaded
        sae_inst = self.saes[layer_id]
        _, dict_size = get_dims(llm, sae_inst)
        feature_dim = dict_size
      
      layer_result = layer_results[str(layer_id)]
      selected = layer_result["selected"]
      feat_idx = selected["feature_index"]
      coeff = selected["coefficient"]
      corr_val = selected["correlation"]
      feat_type = "positive" if coeff > 0 else "negative"
      mode_str = "raw" if raw else "SAE"
      print(f"Layer {layer_id}: Using {feat_type} feature {feat_idx} with coefficient {coeff:.4f} (corr={corr_val:.4f}) [{mode_str}]")
      self.policy_nets[layer_id] = FixedFeaturePolicyNetwork(feature_dim, feat_idx, coeff, raw)
    self.llm = llm
    self.tokenizer = tokenizer
    self.test_loader = test_loader
    self.hooks = {}
    results, total_reward, total = self.evaluate_loop(
      batch_size, max_new_tokens, select_token=select_token
    )
    accuracy = total_reward / total
    print(f"Fixed feature accuracy: {accuracy * 100:.2f}%")
    df = pd.DataFrame(results)
    if len(self.layers) == 1:
      layer_id = self.layers[0]
      layer_result = layer_results[str(layer_id)]
      selected = layer_result["selected"]
      feat_idx = selected["feature_index"]
      coeff = selected["coefficient"]
      name = (
        f"{model}_{task}_{layer_id}_{feat_idx}_{coeff:.0f}"
        if filter_value is None
        else f"{model}_{task}_{filter_value}_{layer_id}_{feat_idx}_{coeff:.0f}"
      )
    else:
      name = (
        f"{model}_{task}_multi_{len(self.layers)}"
        if filter_value is None
        else f"{model}_{task}_{filter_value}_multi_{len(self.layers)}"
      )
    if checkpoint is not None:
      name += f"_ckpt{checkpoint}"
    if limit is not None:
      name += f"_l{limit}"
    if few is not None:
      name += f"_f{few}"
    if select_token:
      name += "_select"
    if decode:
      name += "_decode"
    if cot:
      name += "_cot"
    if name_suffix:
      name = f"{name}{name_suffix}"
    if getattr(self, "reverse", False):
      name += "_reverse"
    output = os.path.join(output_dir, f"{name}.json")
    df.to_json(output, orient="records", indent=2)
    print(f"Results saved to {output}")
    if example:
      baseline_results = self.load_baseline_from_output(
        model, task, output_dir, filter_value, few, select_token, cot
      )
      if baseline_results is None:
        print("Baseline not found, computing baseline first...")
        original_layers = self.layers
        self.layers = []
        baseline_results, baseline_total_reward, baseline_total = self.evaluate_loop(
          batch_size, max_new_tokens, select_token=select_token
        )
        self.layers = original_layers
        baseline_accuracy = baseline_total_reward / max(baseline_total, 1)
        print(f"Baseline accuracy: {baseline_accuracy * 100:.2f}%")
        baseline_results_list = []
        for result in baseline_results:
          baseline_results_list.append((result["prompt"], result["ground_truth"], result["predicted"]))
        baseline_results = baseline_results_list
      else:
        baseline_total_reward = 0.0
        baseline_total = len(baseline_results)
        for prompt, gt, pred in baseline_results:
          reward = calculate_reward(pred, gt, task, self.tokenizer)
          baseline_total_reward += reward
        baseline_accuracy = baseline_total_reward / max(baseline_total, 1)
        print(f"Baseline accuracy (loaded): {baseline_accuracy * 100:.2f}%")
      
      _, _, test_loader_ex = load_dataloaders(
        task,
        seed=seed,
        test_limit=limit,
        val_limit=limit,
        category=category,
        filter_value=filter_value,
      )
      self.test_loader = test_loader_ex
      option_tokens = generate_options(self.tokenizer, self.task)
      steered_layers = self.layers
      positives, negatives, ser_val, accuracy = self.calculate_ser_with_baseline(
        baseline_results, self.test_loader, steered_layers, batch_size,
        max_new_tokens, select_token, option_tokens, output_dir, name
      )
      print(f"Fixed feature accuracy: {accuracy * 100:.2f}%")
      pos_n = len(positives)
      neg_n = len(negatives)
      denom = pos_n + neg_n
      if denom > 0:
        ser = neg_n / denom
        print(f"SER (side effect ratio) = {neg_n}/{denom} = {ser:.4f}")
      else:
        print("SER (side effect ratio) = N/A (no changed examples)")
      metrics = {
        "name": name,
        "model": model,
        "task": task,
        "layers": self.layers,
        "accuracy": accuracy,
        "pos_count": pos_n,
        "neg_count": neg_n,
        "total_changed": denom,
        "ser": ser_val,
        "pos_ratio": (pos_n / denom) if denom > 0 else None,
        "neg_ratio": (neg_n / denom) if denom > 0 else None,
      }
      metrics_output = os.path.join(output_dir, f"{name}_metrics.json")
      with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
      print(f"Metrics saved to {metrics_output}")
    results = {
      "checkpoint": checkpoint,
      "model": model,
      "task": task,
      "layers": self.layers,
      "accuracy": accuracy,
      "category": category,
      "limit": limit,
      "select_token": select_token,
      "decode": decode,
      "cot": cot,
      "output_json": output,
    }
    return EvalResult(**results)


  def manual(
    self,
    feature_file: str,
    method: Literal["global", "foreach", "single", "top"] = "global",
    topk: int = 5,
    neg: bool = False,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = 1,
    model: str = "gemma2b",
    task: str = "mmlu",
    select_token: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str = "output",
    category: Optional[str] = None,
    filter_value: Optional[str] = None,
    seed: int = 42,
    decode: bool = False,
    multiple: int = 1,
    cot: bool = False,
    few: Optional[int] = None,
    mask: str = 'generation',
    example: bool = False,
  ) -> EvalResult:
    """Manual steering with different feature selection methods"""
    task_type = dataset_config[task].type
    if task_type != "select":
      select_token = False
    seed = fix_seed(seed)
    self.task = task
    self.cot = cot
    self.multiple = multiple
    self.decode = decode
    self.mask = mask
    
    batch_size = batch_size if batch_size else model_config[model].batch_size
    tokenizer, llm = load_model_tokenizer(model, dtype=dtype)
    train_loader, _, test_loader = load_dataloaders(
      task,
      seed=seed,
      category=category,
      filter_value=filter_value,
    )
    self.few_shots = None
    if few is not None and few > 0:
      self.few_shots = train_loader.get_last_samples(few)
    max_new_tokens = dataset_config[task].max_new_tokens
    
    # Load baseline from file if exists
    baseline_results = self.load_baseline_from_output(
      model, task, output_dir, filter_value, few, select_token, cot
    )
    baseline_acc = None
    
    with open(feature_file, 'r') as f:
      feature_data = json.load(f)
    
    self.layers = []
    self.saes = {}
    self.policy_nets = {}
    
    if method == "foreach":
      # Use specific features provided in the file
      for layer_id, layer_data in feature_data["layers"].items():
        if "selected" in layer_data:
          layer_id = int(layer_id)
          self.layers.append(layer_id)
          sae_loaded, _, _ = load_sae(model, layer_id, device)
          self.saes[layer_id] = sae_loaded
          _, dict_size = get_dims(llm, sae_loaded)
          selected = layer_data["selected"]
          feat_idx = selected["feature_index"]
          coeff = selected["coefficient"]
          print(f"Layer {layer_id}: Using feature {feat_idx} with coefficient {coeff:.4f}")
          self.policy_nets[layer_id] = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    
    elif method == "global":
      # Use top feature from each layer
      for layer_id, layer_data in feature_data["layers"].items():
        if "analysis" in layer_data:
          layer_id = int(layer_id)
          feature_list = "top_negative_correlations" if neg else "top_positive_correlations"
          if feature_list in layer_data["analysis"]:
            self.layers.append(layer_id)
            sae_loaded, _, _ = load_sae(model, layer_id, device)
            self.saes[layer_id] = sae_loaded
            _, dict_size = get_dims(llm, sae_loaded)
            top_feature = layer_data["analysis"][feature_list][0]
            feat_idx = top_feature["feature_index"]
            coeff = top_feature["coefficient"]
            if neg:
              coeff = -abs(coeff)
            corr = top_feature["correlation"]
            feat_type = "negative" if neg else "positive"
            print(f"Layer {layer_id}: Using top {feat_type} feature {feat_idx} with coefficient {coeff:.4f} (corr={corr:.4f})")
            self.policy_nets[layer_id] = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    
    elif method == "single":
      best_layer = None
      best_feature = None
      best_corr = -1 if not neg else 1
      
      for layer_id, layer_data in feature_data["layers"].items():
        if "analysis" in layer_data:
          feature_list = "top_negative_correlations" if neg else "top_positive_correlations"
          if feature_list in layer_data["analysis"]:
            top_feature = layer_data["analysis"][feature_list][0]
            if (neg and top_feature["correlation"] < best_corr) or (not neg and top_feature["correlation"] > best_corr):
              best_corr = top_feature["correlation"]
              best_layer = int(layer_id)
              best_feature = top_feature
      
      if best_layer is not None:
        self.layers = [best_layer]
        sae_loaded, _, _ = load_sae(model, best_layer, device)
        self.saes[best_layer] = sae_loaded
        _, dict_size = get_dims(llm, sae_loaded)
        feat_idx = best_feature["feature_index"]
        coeff = best_feature["coefficient"]
        feat_type = "negative" if neg else "positive"
        print(f"Single best {feat_type}: Layer {best_layer}, feature {feat_idx} with coefficient {coeff:.4f} (corr={best_corr:.4f})")
        self.policy_nets[best_layer] = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    
    elif method == "top":
      # Use top-k features globally across all layers
      all_features = []
      for layer_id, layer_data in feature_data["layers"].items():
        if "analysis" in layer_data:
          layer_id = int(layer_id)
          feature_list = "top_negative_correlations" if neg else "top_positive_correlations"
          if feature_list in layer_data["analysis"]:
            for feature in layer_data["analysis"][feature_list]:
              all_features.append((layer_id, feature))
      
      # Sort by correlation and take top-k
      reverse_sort = not neg  # For negative, sort ascending (most negative first)
      all_features.sort(key=lambda x: x[1]["correlation"], reverse=reverse_sort)
      top_features = all_features[:topk]
      
      for layer_id, feature in top_features:
        if layer_id not in self.layers:
          self.layers.append(layer_id)
          sae_loaded, _, _ = load_sae(model, layer_id, device)
          self.saes[layer_id] = sae_loaded
          _, dict_size = get_dims(llm, sae_loaded)
        
        feat_idx = feature["feature_index"]
        coeff = feature["coefficient"]
        corr = feature["correlation"]
        feat_type = "negative" if neg else "positive"
        print(f"Top-{topk} {feat_type}: Layer {layer_id}, feature {feat_idx} with coefficient {coeff:.4f} (corr={corr:.4f})")
        self.policy_nets[layer_id] = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    
    if not self.layers:
      raise ValueError(f"No valid features found for method '{method}'")
    
    self.llm = llm
    self.tokenizer = tokenizer
    self.test_loader = test_loader
    self.hooks = {}
    
    name = f"{model}_{task}_manual_{method}"
    if neg:
      name += "_neg"
    if method == "top":
      name += f"_k{topk}"
    if filter_value is not None:
      name += f"_{filter_value}"
    if few is not None:
      name += f"_f{few}"
    if select_token:
      name += "_select"
    if decode:
      name += "_decode"
    if cot:
      name += "_cot"
    
    # Add source task info to prevent overwriting
    if "feature_file" in locals():
      source_task = feature_file.split("_")[1] if "_" in feature_file else "unknown"
      name += f"_from_{source_task}"

    if not example:
      results, total_reward, total = self.evaluate_loop(
        batch_size, max_new_tokens, select_token=select_token
      )
      accuracy = total_reward / total
      print(f"Manual {method} accuracy: {accuracy * 100:.2f}%")
      
      df = pd.DataFrame(results)
      output = os.path.join(output_dir, f"{name}.json")
      df.to_json(output, orient="records", indent=2)
      print(f"Results saved to {output}")
    
    if example:
      baseline_results = self.load_baseline_from_output(
        model, task, output_dir, filter_value, few, select_token, cot
      )
      if baseline_results is None:
        print("Baseline not found, computing baseline first...")
        original_layers = self.layers
        self.layers = []
        baseline_results, baseline_total_reward, baseline_total = self.evaluate_loop(
          batch_size, max_new_tokens, select_token=select_token
        )
        self.layers = original_layers
        baseline_accuracy = baseline_total_reward / baseline_total
        print(f"Baseline {method} accuracy: {baseline_accuracy * 100:.2f}%")
        baseline_results_list = []
        for result in baseline_results:
          baseline_results_list.append((result["prompt"], result["ground_truth"], result["predicted"]))
        baseline_results = baseline_results_list
      else:
        baseline_total_reward = 0.0
        baseline_total = len(baseline_results)
        for prompt, gt, pred in baseline_results:
          reward = calculate_reward(pred, gt, task, self.tokenizer)
          baseline_total_reward += reward
        baseline_accuracy = baseline_total_reward / baseline_total if baseline_total > 0 else 0.0
        print(f"Baseline accuracy (loaded): {baseline_accuracy * 100:.2f}%")
      _, _, test_loader_ex = load_dataloaders(
        task,
        seed=seed,
        category=category,
        filter_value=filter_value,
      )
      self.test_loader = test_loader_ex
      option_tokens = generate_options(self.tokenizer, self.task)
      positives, negatives, ser_val, accuracy = self.calculate_ser_with_baseline(
        baseline_results, self.test_loader, self.layers, batch_size,
        max_new_tokens, select_token, option_tokens, output_dir, name
      )
      print(f"Manual {method} accuracy: {accuracy * 100:.2f}%")
      pos_n = len(positives)
      neg_n = len(negatives)
      denom = pos_n + neg_n
      metrics = {
        "name": name,
        "model": model,
        "task": task,
        "layers": self.layers,
        "accuracy": accuracy,
        "pos_count": pos_n,
        "neg_count": neg_n,
        "total_changed": denom,
        "ser": ser_val,
        "pos_ratio": (pos_n / denom) if denom > 0 else None,
        "neg_ratio": (neg_n / denom) if denom > 0 else None,
      }
      metrics_output = os.path.join(output_dir, f"{name}_metrics.json")
      with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
      print(f"Metrics saved to {metrics_output}")
      output = ""
    else:
      output = os.path.join(output_dir, f"{name}.json")
    
    stats = {
      "checkpoint": None,
      "model": model,
      "task": task,
      "layers": self.layers,
      "accuracy": accuracy,
      "category": category,
      "select_token": select_token,
      "decode": decode,
      "cot": self.cot,
      "method": method,
      "neg": neg,
      "topk": topk if method == "top" else None,
      "feature_file": feature_file,
      "output_json": output,
    }
    stats_output = os.path.join(output_dir, f"{name}_stats.json")
    try:
      with open(stats_output, "r") as f:
        all_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      all_stats = []
    all_stats.append(stats)
    with open(stats_output, "w") as f:
      json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_output}")
    
    return EvalResult(**stats)


  def multi_feature_steering(
    self,
    features: List[Tuple[int, int, float]],  # List of (layer, feature_index, coefficient)
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = 1,
    model: str = "gemma2b",
    task: str = "mmlu",
    select_token: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str = "output",
    category: Optional[str] = None,
    filter_value: Optional[str] = None,
    seed: int = 42,
    decode: bool = False,
    multiple: int = 1,
    cot: bool = False,
    few: Optional[int] = None,
    mask: str = 'generation',
    example: bool = False,
  ) -> EvalResult:
    """Multi-feature steering with list of (layer, feature_index, coefficient)"""
    task_type = dataset_config[task].type
    if task_type != "select":
      select_token = False
    seed = fix_seed(seed)
    self.task = task
    self.cot = cot
    self.multiple = multiple
    self.decode = decode
    self.mask = mask
    
    batch_size = batch_size if batch_size else model_config[model].batch_size
    tokenizer, llm = load_model_tokenizer(model, dtype=dtype)
    train_loader, _, test_loader = load_dataloaders(
      task,
      seed=seed,
      category=category,
      filter_value=filter_value,
    )
    self.few_shots = None
    if few is not None and few > 0:
      self.few_shots = train_loader.get_last_samples(few)
    max_new_tokens = dataset_config[task].max_new_tokens
    
    self.layers = []
    self.saes = {}
    self.policy_nets = {}
    
    # Set up features
    for layer, feat_idx, coeff in features:
      if layer not in self.layers:
        self.layers.append(layer)
        sae_loaded, _, _ = load_sae(model, layer, device)
        self.saes[layer] = sae_loaded
        _, dict_size = get_dims(llm, sae_loaded)
        print(f"Layer {layer}: Using feature {feat_idx} with coefficient {coeff:.4f}")
        self.policy_nets[layer] = FixedFeaturePolicyNetwork(dict_size, feat_idx, coeff)
    
    self.llm = llm
    self.tokenizer = tokenizer
    self.test_loader = test_loader
    self.hooks = {}
    
    results, total_reward, total = self.evaluate_loop(
      batch_size, max_new_tokens, select_token=select_token
    )
    accuracy = total_reward / total
    print(f"Multi-feature steering accuracy: {accuracy * 100:.2f}%")
    
    df = pd.DataFrame(results)
    feature_str = "_".join([f"L{l}_F{f}_C{c:.0f}" for l, f, c in features])
    name = f"{model}_{task}_multi_{feature_str}"
    if filter_value is not None:
      name += f"_{filter_value}"
    if few is not None:
      name += f"_f{few}"
    if select_token:
      name += "_select"
    if decode:
      name += "_decode"
    if cot:
      name += "_cot"
    
    output = os.path.join(output_dir, f"{name}.json")
    df.to_json(output, orient="records", indent=2)
    print(f"Results saved to {output}")
    
    if example:
      baseline_results = self.load_baseline_from_output(
        model, task, output_dir, filter_value, few, select_token, cot
      )
      if baseline_results is None:
        print("Baseline not found, computing baseline first...")
        original_layers = self.layers.copy()
        self.layers = []
        baseline_results, baseline_total_reward, baseline_total = self.evaluate_loop(
          batch_size, max_new_tokens, select_token=select_token
        )
        baseline_accuracy = baseline_total_reward / baseline_total
        print(f"Baseline multi-feature accuracy: {baseline_accuracy * 100:.2f}%")
        baseline_results_list = []
        for result in baseline_results:
          baseline_results_list.append((result["prompt"], result["ground_truth"], result["predicted"]))
        baseline_results = baseline_results_list
        self.layers = original_layers
      else:
        baseline_total_reward = 0.0
        baseline_total = len(baseline_results)
        for prompt, gt, pred in baseline_results:
          reward = calculate_reward(pred, gt, task, self.tokenizer)
          baseline_total_reward += reward
        baseline_accuracy = baseline_total_reward / baseline_total if baseline_total > 0 else 0.0
        print(f"Baseline accuracy (loaded): {baseline_accuracy * 100:.2f}%")
      _, _, test_loader_ex = load_dataloaders(
        task,
        seed=seed,
        category=category,
        filter_value=filter_value,
      )
      self.test_loader = test_loader_ex
      option_tokens = generate_options(self.tokenizer, self.task)
      positives, negatives, ser_val, accuracy = self.calculate_ser_with_baseline(
        baseline_results, self.test_loader, self.layers, batch_size,
        max_new_tokens, select_token, option_tokens, output_dir, name
      )
      print(f"Multi-feature steering accuracy: {accuracy * 100:.2f}%")
      pos_n = len(positives)
      neg_n = len(negatives)
      denom = pos_n + neg_n
      metrics = {
        "name": name,
        "model": model,
        "task": task,
        "layers": self.layers,
        "accuracy": accuracy,
        "pos_count": pos_n,
        "neg_count": neg_n,
        "total_changed": denom,
        "ser": ser_val,
        "pos_ratio": (pos_n / denom) if denom > 0 else None,
        "neg_ratio": (neg_n / denom) if denom > 0 else None,
      }
      metrics_output = os.path.join(output_dir, f"{name}_metrics.json")
      with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
      print(f"Metrics saved to {metrics_output}")
    
    stats = {
      "checkpoint": None,
      "model": model,
      "task": task,
      "layers": self.layers,
      "accuracy": accuracy,
      "category": category,
      "select_token": select_token,
      "decode": decode,
      "cot": self.cot,
      "method": "multi_feature",
      "features": features,
      "output_json": output,
    }
    stats_output = os.path.join(output_dir, f"{name}_stats.json")
    try:
      with open(stats_output, "r") as f:
        all_stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      all_stats = []
    all_stats.append(stats)
    with open(stats_output, "w") as f:
      json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_output}")
    
    return EvalResult(**stats)


if __name__ == "__main__":
  fire.Fire(EvalController, serialize=lambda x: None)
