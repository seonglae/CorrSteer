from torch import Tensor
from torch import nn
from sae_lens import SAE
import torch
from typing import Literal

class SteeringHook:
  def __init__(
    self,
    policy_net: nn.Module,
    sae: SAE = None,
    substract: bool = False,
    decode: bool = False,
    lastk: int = 1,
    multiple: int = 1,
    mask: Literal['generation', 'all'] = 'generation',
    raw: bool = False,
  ):
    self.substract: bool = substract
    self.decode: bool = decode
    self.policy_net: nn.Module = policy_net
    self.sae: SAE = sae
    self.raw: bool = raw
    self.lastk: int = lastk
    self.multiple: int = multiple
    self.action: Tensor = None
    self.observation: Tensor = None
    self.mask: str = mask
    self.mask_features: Tensor = None
    self.stacked_tokens: int = 0
    
  def __call__(self, module: nn.Module, inputs: tuple[Tensor]) -> Tensor:
    residual: Tensor = inputs[0]  # shape: (B, seq_len, hidden_dim)
    
    if self.raw:
      # Raw mode: use residual stream directly
      observation: Tensor = residual[:, -self.lastk :, :]  # (B, lastk, hidden_dim)
    else:
      # SAE mode: convert to SAE dtype
      observation: Tensor = residual[:, -self.lastk :, :].to(
        self.sae.dtype
      )  # (B, lastk, hidden_dim)
    
    if self.sae is not None and not self.raw:
      if self.mask == 'generation':
        mask_tokens = residual[:, -1:, :].to(self.sae.dtype)  # (B, 1, hidden_dim)
      elif self.mask == 'all':
        mask_tokens = residual.to(self.sae.dtype)  # (B, seq_len, hidden_dim)
      batch_size, seq_len, hidden_dim = mask_tokens.shape
      mask_tokens_2d = mask_tokens.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
      mask_encoded_2d = self.sae.encode(mask_tokens_2d)  # (B*seq_len, dict_size)
      dict_size = mask_encoded_2d.shape[-1]
      mask_encoded = mask_encoded_2d.view(batch_size, seq_len, dict_size)  # (B, seq_len, dict_size)
      current_mask = mask_encoded.sum(dim=1)  # (B, dict_size)
      if self.mask_features is None:
        self.mask_features = current_mask
        self.stacked_tokens = seq_len
      else:
        total_tokens = self.stacked_tokens + seq_len
        self.mask_features = (self.mask_features * self.stacked_tokens + current_mask * seq_len) / total_tokens
        self.stacked_tokens = total_tokens
    
    # For policy and action computation, we detach to prevent gradients
    observation_detached = observation.detach()
    mask_for_policy = self.mask_features
    try:
      out = self.policy_net.select_action(observation_detached, mask_for_policy)
      action = out[0] if isinstance(out, tuple) else out
    except TypeError:
      out = self.policy_net.select_action(observation_detached)
      action = out[0] if isinstance(out, tuple) else out
    action = action.detach()

    if self.action is None:
      self.action = action
      self.observation = observation_detached
    else:
      if action.size(0) != self.action.size(0):
        min_batch = min(action.size(0), self.action.size(0))
        action = action[:min_batch]
        observation_detached = observation_detached[:min_batch]
        self.action = self.action[:min_batch]
        self.observation = self.observation[:min_batch]
      
      self.action = torch.cat([self.action, action], dim=1) # (batch_size, inference, dict_size)
      self.observation = torch.cat([self.observation, observation_detached], dim=1) # (batch_size, inference, latent_dim or dict_size)
      
    if self.raw:
      # Raw mode: action is already in hidden_dim space
      steering: Tensor = action * self.multiple  # shape: (B, lastk, hidden_dim)
    elif self.sae is not None:
      if self.decode:
        batch_size, seq_len, dict_size = action.shape
        action_2d = action.view(-1, dict_size)  # (B*lastk, dict_size)
        decoded_2d = self.sae.decode(action_2d)  # (B*lastk, hidden_dim)
        hidden_dim = decoded_2d.shape[-1]
        steering = decoded_2d.view(batch_size, seq_len, hidden_dim) * self.multiple # (B, lastk, hidden_dim)
      else:
        steering = action @ self.sae.W_dec * self.multiple # shape: (B, lastk, hidden_dim)
    else:
      steering: Tensor = action  # shape: (B, latent_dim)
    steering_vector: Tensor = -steering if self.substract else steering
    residual_copy = residual.clone()
    residual_copy[:, -self.lastk :, :] = residual_copy[:, -self.lastk :, :] + steering_vector
    return residual_copy


def get_steering_hook(
  policy_net: nn.Module,
  sae: SAE = None,
  substract: bool = False,
  decode: bool = False,
  lastk: int = 1,
  multiple: int = 1,
  mask: Literal['generation', 'all'] = 'generation',
  raw: bool = False,
):
  return SteeringHook(
    policy_net=policy_net,
    sae=sae, 
    substract=substract,
    decode=decode, 
    lastk=lastk,
    multiple=multiple,
    mask=mask,
    raw=raw,
  )
