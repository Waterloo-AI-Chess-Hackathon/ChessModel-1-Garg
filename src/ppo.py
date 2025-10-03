import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOConfig:
    device: str = "mps"
    gamma: float = 0.997
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 2.5e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    rollout_steps: int = 512
    n_envs: int = 32
    update_epochs: int = 6
    minibatch_size: int = 8192
    value_clip: bool = True
    use_huber_vf: bool = True

class PPOAgent:
    def __init__(self, model) -> None:
        pass