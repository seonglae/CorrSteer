from abc import ABC, abstractmethod
from typing import Optional, List
from typing_extensions import TypedDict, NotRequired


class SampleData(TypedDict):
  question: str
  answer: str
  choices: NotRequired[Optional[List[str]]]
  evaluation_data: NotRequired[Optional[dict]]
  metadata: NotRequired[Optional[dict]]


class DataLoader(ABC):
  def __init__(self, dataset, split=None, limit=None):
    if split is not None:
      self.data = dataset[split]
    else:
      self.data = dataset
    if limit is not None:
      actual_limit = min(limit, len(self.data))
      self.data = self.data.select(range(actual_limit))
    self.n_samples = len(self.data)
    self.index = 0

  @abstractmethod
  def apply_row(self, sample) -> SampleData:
    pass

  def get_batch(self, batch_size: int) -> List[SampleData]:
    batch = []
    for _ in range(batch_size):
      if self.index >= self.n_samples:
        break
      sample = self.data[self.index]
      self.index += 1
      batch.append(self.apply_row(sample))
    if self.index >= self.n_samples:
      self.index = 0
    return batch

  def get_last_samples(self, num_samples: int) -> List[SampleData]:
    """Get the last N samples from the dataset for few-shot examples"""
    start_idx = max(0, self.n_samples - num_samples)
    last_samples = []
    for i in range(start_idx, self.n_samples):
      sample = self.data[i]
      last_samples.append(self.apply_row(sample))
    return last_samples

  def __iter__(self):
    for sample in self.data:
      yield self.apply_row(sample)


class MMLUDataLoader(DataLoader):
  def __init__(self, dataset, split, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": sample["question"],
      "choices": sample["choices"],
      "answer": sample["answer"],
    }


class MMLUProDataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": sample["question"],
      "choices": sample["options"],
      "answer": sample["answer"],
    }


class GSM8KDataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": sample["question"],
      "answer": sample["answer"],
    }


class BBQDataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    answer = sample["label"]
    if answer == 0:
      answer = "A"
    elif answer == 1:
      answer = "B"
    elif answer == 2:
      answer = "C"
    return {
      "question": f"{sample['context']}\n{sample['question']}",
      "choices": [sample["ans0"], sample["ans1"], sample["ans2"]],
      "answer": answer,
    }


class SimpleQADataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": sample["problem"],
      "answer": sample["answer"],
    }



class HarmBenchDataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": f"{sample['context']}\n{sample['prompt']}" if sample['context'] else sample['prompt'],
      "answer": "",
    }


class XSTestDataLoader(DataLoader):
  def __init__(self, dataset, split=None, limit=None):
    super().__init__(dataset, split=split, limit=limit)

  def apply_row(self, sample) -> SampleData:
    return {
      "question": sample["prompt"],
      "answer": sample["label"]
    }
