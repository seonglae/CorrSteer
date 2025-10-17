from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from corrsteer.config import TaskType


class FeatureExample(BaseModel):
  prompt: str
  baseline: str
  steered: str
  oracle: str
  think: Optional[str] = None


class FeatureAnalysis(BaseModel):
  index: int
  description: str
  url: str
  accuracy: Dict[str, str]
  examples: List[FeatureExample]


class CriticResult(BaseModel):
  """Structured critic analysis results with explicit fields."""
  avg: float
  std: float
  correct_avg: float
  correct_std: float
  corrected_avg: float
  corrected_std: float
  misguided_avg: float
  misguided_std: float
  incorrect_avg: float
  incorrect_std: float


CriticStats = Dict[str, List[CriticResult]]
FeatureStats = Dict[str, List[List[FeatureAnalysis]]]


class EvalResult(BaseModel):
  checkpoint: Optional[str] = None
  model: str
  task: TaskType
  layers: List[int]
  accuracy: float
  category: Optional[str] = None
  limit: Optional[int] = None
  select_token: bool
  decode: bool
  cot: bool
  output_json: str
  critic: Optional[CriticStats] = None
  feature: Optional[FeatureStats] = None


class TrainStepResult(BaseModel):
  layer_metrics: Dict[int, Dict[str, float]]
  train_accuracy: float
  think_length: Optional[float] = None


class TrainValidationResult(BaseModel):
  val_accuracy: float
  unique_indices: Dict[int, int]
  think_length: Optional[float] = None
  act_values: Dict[int, List[float]]


class TrainResult(BaseModel):
  step: int
  layers: List[int]
  decode: bool
  select_token: bool
  category: Optional[str]
  critic_deep: bool
  policy_deep: bool
  filter_value: Optional[str]
  multiple: int
  act: str
  sparse: bool
  grpo: bool
  cot: bool
  sigma: float
  raw: bool
  policy_state_dict: Dict[int, Dict[str, Any]]
  critic_state_dict: Dict[int, Dict[str, Any]]


class TrainStats(BaseModel):
  steered: Optional[EvalResult] = None
  fixed: Optional[EvalResult] = None
  universal: Optional[EvalResult] = None
