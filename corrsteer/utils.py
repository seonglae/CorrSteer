import os
import torch
import random
from typing import Optional, cast, List
import numpy as np
import re
import pandas as pd
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  LogitsProcessorList,
  PreTrainedTokenizer,
  PreTrainedModel,
)
from datasets import (
  load_dataset,
  Dataset,
  concatenate_datasets
)
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from functools import cache

from corrsteer.config import dataset_config, model_config
from corrsteer.dataset import DataLoader, SampleData


os.environ["TORCHDYNAMO_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_CUDAGRAPHS_DISABLE"] = "1"
os.environ["TORCH_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"



@cache
def get_sae_directory():
  df = pd.DataFrame.from_records(
    {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
  ).T
  df.drop(
    columns=[
      "expected_var_explained",
      "expected_l0",
      "config_overrides",
      "conversion_func",
    ],
    inplace=True,
  )
  return df


def load_model_tokenizer(
  model, dtype=torch.bfloat16, device_map="auto"
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
  tokenizer = AutoTokenizer.from_pretrained(model_config[model].id)
  tokenizer.padding_side = "left"
  llm = AutoModelForCausalLM.from_pretrained(
    model_config[model].id,
    device_map=device_map,
    torch_dtype=dtype,
    trust_remote_code=True,
  )
  llm.eval()
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    llm.config.pad_token_id = tokenizer.eos_token_id
  return tokenizer, llm


def fix_seed(seed: int = 42) -> int:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  return seed


def get_device() -> str:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "mps" if torch.backends.mps.is_available() else device
  return device


def get_dims(llm: PreTrainedModel, sae: Optional[SAE] = None) -> tuple[int, int]:
  latent_dim = getattr(llm.config, "hidden_size", None) or getattr(
    llm.config, "n_embd", None
  )
  if sae is None:
    dict_size = latent_dim
  else:
    dict_size = sae.cfg.d_sae
  return cast(int, latent_dim), cast(int, dict_size)


def load_sae(
  model: str, layer: int, device: str
) -> tuple[SAE, dict, Optional[torch.Tensor]]:
  sae: SAE
  cfg_dict: dict
  sparsity: Optional[torch.Tensor]
  sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=model_config[model].release,
    sae_id=model_config[model].id_template.format(layer),
  )
  return sae.to(device), cfg_dict, sparsity


def build_prompt(sample: SampleData, task: str, cot: bool = False, few_shots: Optional[List[SampleData]] = None) -> tuple[str, str]:
  task_type = dataset_config[task].type
  question: str = sample["question"]
  system_prompt = dataset_config[task].system_prompt
  prefix = ""
  if system_prompt:
    prefix = f"{system_prompt}\n\n"
  if few_shots:
    for example in few_shots:
      example_question = example["question"]
      if task_type == "select":
        example_choices = example.get("choices", [])
        if isinstance(example["answer"], int):
          example_answer = chr(65 + example["answer"])
        else:
          example_answer = example["answer"].strip().upper()
        example_prompt = (
          example_question + "\n" +
          "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example_choices)) +
          f"\nAnswer: {example_answer}\n"
        )
      else:
        if task == "gsm8k":
          match = re.search(r'####\s*(-?\d+)', example["answer"])
          example_answer = match.group(1) if match else example["answer"].strip()
          example_prompt = f"Q: {example_question}\nA: {example_answer}\n"
        elif task == "xstest":
          # For XSTest, show classification format
          example_answer = example["answer"].strip()
          example_prompt = f"Request: {example_question}\nClassification: {example_answer}\n"
        elif task == "harmbench":
          # For HarmBench, only show the request without answer
          example_prompt = f"Request: {example_question}\n"
        else:
          example_answer = example["answer"].strip()
          example_prompt = f"Q: {example_question}\nA: {example_answer}\n"
      prefix += example_prompt

  if task_type == "select":
    choices: Optional[list[str]] = sample.get("choices")
    if choices is None:
      raise ValueError(f"Task {task} requires choices but none provided")
    if task == "mmlu":
      candidates = "A, B, C, or D"
    elif task == "bbq":
      candidates = "A, B, or C"
    else:
      candidates = ", ".join(cast(list[str], dataset_config[task].options))
    if isinstance(sample["answer"], int):
      correct_answer = chr(65 + sample["answer"])
    else:
      correct_answer: str = sample["answer"].strip().upper()
    prompt: str = (
      question
      + "\n"
      + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
      + f"\nChoose one of the following options only: {candidates}"
      + "\nAnswer:"
    )
    if cot:
      prompt = (
        "Let's think step by step. Answer the question after the 'Answer:' tag.\n\n"
        + prompt
      )
    return prefix + prompt, correct_answer
  elif task_type == "reason":
    answer: str = sample["answer"]
    if task == "gsm8k":
      import re
      match = re.search(r'####\s*(-?\d+)', answer)
      correct_answer = match.group(1) if match else answer.strip()
    else:
      correct_answer = answer.strip()
    if cot:
      prompt = (
        f"Let's think step by step. Give your final numerical answer at the end after the ####n\n"
        f"Q: {question}\n"
      )
    else:
      if task == "gsm8k":
        prompt = f"Q: {question}\nA: Give your final numerical answer after ####\n"
      else:
        prompt = f"Q: {question}\nA:"
    return prefix + prompt, correct_answer
  elif task_type == "answer":
    answer: str = sample["answer"]
    correct_answer = answer.strip()
    if task == "xstest":
      # For XSTest, use classification format
      if cot:
        prompt = f"Let's think step by step.\nRequest: {question}\nClassification:"
      else:
        prompt = f"Request: {question}\nClassification:"
    elif task == "harmbench":
      # For HarmBench, just show the request
      if cot:
        prompt = f"Let's think step by step.\nRequest: {question}\nResponse:"
      else:
        prompt = f"Request: {question}\nResponse:"
    else:
      if cot:
        prompt = f"Let's think step by step.\nQ: {question}\nA:"
      else:
        prompt = f"Q: {question}\nA:"
    return prefix + prompt, correct_answer


def extract_answer_from_cot(response: str, task: str) -> tuple[str, str]:
  match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', response)
  if match:
    return response[:match.start()].strip(), match.group(1)
  match = re.search(r'[Tt]he answer is\s*(-?\d+(?:\.\d+)?)', response)
  if match:
    return response[:match.start()].strip(), match.group(1)
  match = re.search(r'(\d+(?:\.\d+)?)\s*$', response.strip())
  if match:
    return response[:match.start()].strip(), match.group(1)
  if dataset_config[task].type == "reason":
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    if numbers:
      answer = numbers[-1]
      last_match = None
      for match in re.finditer(re.escape(answer), response):
        last_match = match
      if last_match:
        return response[:last_match.start()].strip(), answer
  words = response.strip().split()
  if words:
    return " ".join(words[:-1]), words[-1]
  return response.strip(), ""


def extract_answer(response: str, task: str) -> tuple[str, str]:
  task_type = dataset_config[task].type
  if task_type == "select":
    return "", response.strip()
  elif task_type == "reason":
    return extract_answer_from_cot(response, task)  
  elif task_type == "answer":
    return "", response.strip()


def get_supervised_pair(sample: SampleData, task: str, cot: bool = False, few_shots: Optional[List[SampleData]] = None) -> tuple[str, str]:
  """Return (prompt, target_text) for SFT.
  - select: target is the short option token matching the answer (fills after 'Answer:').
  - reason (e.g., gsm8k): target is the rationale BEFORE the final marker plus the final numeric answer (e.g., "...\n#### 42").
  - answer: target is the full answer string (free-form).
  """
  prompt, correct_answer = build_prompt(sample, task, cot=cot, few_shots=few_shots)
  task_type = dataset_config[task].type
  if task_type == "select":
    target = f" {correct_answer}"
    return prompt, target
  elif task_type == "reason":
    if cot:
      full_ref = sample.get("answer", "")
      think, final_ans = extract_answer_from_cot(full_ref, task)
      think = think.strip()
      if final_ans and len(str(final_ans).strip()) > 0:
        target = (think + ("\n" if len(think) > 0 else "") + f"#### {final_ans}").strip()
      else:
        target = think if len(think) > 0 else full_ref.strip()
    else:
      full_ref = sample.get("answer", "")
      think, final_ans = extract_answer_from_cot(full_ref, task)
      think = think.strip()
      if final_ans and len(str(final_ans).strip()) > 0:
        target = (think + ("\n" if len(think) > 0 else "") + f"#### {final_ans}").strip()
      else:
        target = think if len(think) > 0 else full_ref.strip()
    return prompt, target
  else:  # "answer"
    target = str(sample.get("answer", "")).strip()
    return prompt, target


def load_dataloaders(
  task,
  seed=42,
  val_limit=None,
  test_limit=None,
  category=None,
  test_size=0.7,
  val_size=0.1,
  filter_value=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  if not dataset_config[task]:
    raise ValueError(f"Dataset config for task {task} not found")
  if category:
    ds = load_dataset(dataset_config[task].id, name=category)
    dataset = cast(Dataset, cast(Dataset, ds)[dataset_config[task].split]).shuffle(seed=seed)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset = train_dataset.train_test_split(
      test_size=val_size, seed=seed
    )
    val_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    train_loader = dataset_config[task].dataloader(train_dataset)
    val_loader = dataset_config[task].dataloader(
      val_dataset, limit=val_limit
    )
    test_loader = dataset_config[task].dataloader(
      test_dataset, limit=test_limit
    )
  elif category is None and not dataset_config[task].merged and dataset_config[task].subsets:
    subsets = cast(List[str], dataset_config[task].subsets)
    datasets: List[Dataset] = cast(List[Dataset], [
      cast(Dataset, load_dataset(dataset_config[task].id, name=subset))[dataset_config[task].split]
      for subset in subsets
    ])
    merged_dataset = concatenate_datasets(datasets)
    if filter_value:
      merged_dataset = merged_dataset.filter(
        lambda x: x[dataset_config[task].filter] == filter_value
      )
    merged_dataset = merged_dataset.shuffle(seed=seed)
    dataset = merged_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset = train_dataset.train_test_split(
      test_size=val_size, seed=seed
    )
    val_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    train_loader = dataset_config[task].dataloader(train_dataset)
    val_loader = dataset_config[task].dataloader(
      val_dataset, limit=val_limit
    )
    test_loader: DataLoader = dataset_config[task].dataloader(
      test_dataset, limit=test_limit
    )
  else:
    dataset = load_dataset(dataset_config[task].id, dataset_config[task].split).shuffle(
      seed=seed
    )
    if dataset_config[task].test is None and dataset_config[task].val is None:
      dataset = cast(Dataset, cast(Dataset, dataset)[dataset_config[task].train])
      dataset = dataset.train_test_split(test_size=test_size, seed=seed)
      train_dataset = dataset["train"]
      test_dataset = dataset["test"]
      train_dataset = train_dataset.train_test_split(
        test_size=val_size, seed=seed
      )
      val_dataset = train_dataset["test"]
      train_dataset = train_dataset["train"]
      train_loader = dataset_config[task].dataloader(train_dataset)
      val_loader = dataset_config[task].dataloader(
        val_dataset, limit=val_limit
      )
      test_loader = dataset_config[task].dataloader(
        test_dataset, limit=test_limit
      )
    elif dataset_config[task].val is None and dataset_config[task].test is not None:
      test_loader = dataset_config[task].dataloader(
        dataset, split=dataset_config[task].test, limit=test_limit
      )
      dataset = cast(Dataset, cast(Dataset, dataset)[dataset_config[task].train])
      dataset = dataset.train_test_split(test_size=val_size, seed=seed)
      train_loader = dataset_config[task].dataloader(dataset["train"])
      val_loader = dataset_config[task].dataloader(dataset["test"], limit=val_limit)
    elif dataset_config[task].val is not None and dataset_config[task].test is not None:
      train_loader = dataset_config[task].dataloader(
        dataset, split=dataset_config[task].train
      )
      val_loader = dataset_config[task].dataloader(
        dataset, split=dataset_config[task].val, limit=val_limit
      )
      test_loader = dataset_config[task].dataloader(
        dataset, split=dataset_config[task].test, limit=test_limit
      )
  return train_loader, val_loader, test_loader


def generate_options(tokenizer, task) -> list[int]:
  option_tokens = []
  options: list[str] = dataset_config[task].options or []
  for option in options:
    token_id = tokenizer.encode(option, add_special_tokens=False)[0]
    option_tokens.append(token_id)
  return option_tokens


class RestrictTokens:
  def __init__(self, allowed_token_ids):
    self.allowed_token_ids = allowed_token_ids

  def __call__(self, input_ids, scores):
    mask = torch.full_like(scores, float("-inf"))
    mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
    return mask


def get_logit_processor(option_tokens):
  return LogitsProcessorList([RestrictTokens(option_tokens)])


def get_eos_positions(generated_ids: torch.Tensor, input_ids: torch.Tensor, tokenizer, task: str) -> torch.Tensor:
  offset = input_ids.shape[1]
  sliced = generated_ids[:, offset:]
  task_type = dataset_config[task].type
  if task_type != "select":
    pad_mask = (sliced == tokenizer.pad_token_id).float()
    eos_mask = (sliced == tokenizer.eos_token_id).float() if tokenizer.eos_token_id is not None else torch.zeros_like(pad_mask)
    combined_mask = pad_mask + eos_mask
    combined_mask = (combined_mask > 0).float()
    eos_position = combined_mask.argmax(dim=1)
    eos_positions = torch.where(combined_mask.any(dim=1), eos_position, sliced.shape[1])
  else:
    eos_positions = torch.ones(sliced.shape[0], dtype=torch.int32, device=sliced.device)
  return eos_positions
