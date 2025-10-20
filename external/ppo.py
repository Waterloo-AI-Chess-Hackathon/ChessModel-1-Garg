from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math


@dataclass
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


class RolloutBuffer:
    """Buffer to store rollout data for PPO training."""
    
    def __init__(self, rollout_steps: int, n_envs: int, obs_shape: Tuple[int, ...], device: str):
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.device = device
        
        self.observations = torch.zeros((rollout_steps, n_envs, *obs_shape), device=device)
        self.actions = torch.zeros((rollout_steps, n_envs), device=device, dtype=torch.long)
        self.rewards = torch.zeros((rollout_steps, n_envs), device=device)
        self.values = torch.zeros((rollout_steps, n_envs), device=device)
        self.log_probs = torch.zeros((rollout_steps, n_envs), device=device)
        self.advantages = torch.zeros((rollout_steps, n_envs), device=device)
        self.returns = torch.zeros((rollout_steps, n_envs), device=device)
        self.dones = torch.zeros((rollout_steps, n_envs), device=device, dtype=torch.bool)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, 
            value: torch.Tensor, log_prob: torch.Tensor, done: torch.Tensor):
        """Add a single step to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        if self.ptr == self.rollout_steps:
            self.full = True
            self.ptr = 0
    
    def compute_gae(self, next_value: torch.Tensor, gamma: float, lam: float):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        
        self.returns = advantages + self.values
        self.advantages = advantages
    
    def get(self):
        """Get all data from buffer."""
        if not self.full:
            raise ValueError("Buffer not full yet")
        
        obs = self.observations.view(-1, *self.observations.shape[2:])
        actions = self.actions.view(-1)
        old_log_probs = self.log_probs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        
        return obs, actions, old_log_probs, advantages, returns


class PPOAgent:
    def __init__(self, model: nn.Module, config: PPOConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=1000
        )
        
        obs_shape = (119, 8, 8)
        self.buffer = RolloutBuffer(
            config.rollout_steps, config.n_envs, obs_shape, config.device
        )
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'explained_variance': []
        }
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from model.
        
        Args:
            obs: Observation tensor of shape (batch_size, 119, 8, 8)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected actions
            log_prob: Log probabilities of actions
            value: State values
        """
        with torch.no_grad():
            policy_logits, value = self.model(obs)
            
            if deterministic:
                # TODO: Implement MCTS to determine best move rather than greedy
                action = torch.argmax(policy_logits, dim=-1)
                log_prob = F.log_softmax(policy_logits, dim=-1)
                log_prob = torch.gather(log_prob, 1, action.unsqueeze(1)).squeeze(1)
            else:
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, value.squeeze(-1)
    
    def collect_rollouts(self, envs: List) -> Dict[str, float]:
        """Collect rollouts from multiple environments."""
        obs = torch.zeros((self.config.n_envs, 119, 8, 8), device=self.device)
        
        for i, env in enumerate(envs):
            obs[i], _ = env.reset()
        
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        
        for step in range(self.config.rollout_steps):
            with torch.no_grad():
                action, log_prob, value = self.get_action(obs)
            
            rewards = torch.zeros(self.config.n_envs, device=self.device)
            dones = torch.zeros(self.config.n_envs, device=self.device, dtype=torch.bool)
            
            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, _ = env.step(action[i].item())
                rewards[i] = reward
                dones[i] = terminated or truncated
                obs[i] = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
            
            self.buffer.add(obs, action, rewards, value, log_prob, dones)
        
        with torch.no_grad():
            _, _, next_value = self.get_action(obs)
        
        self.buffer.compute_gae(next_value, self.config.gamma, self.config.lam)
        
        advantages = self.buffer.advantages
        returns = self.buffer.returns
        
        stats = {
            'mean_reward': self.buffer.rewards.mean().item(),
            'mean_value': self.buffer.values.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'explained_variance': self._explained_variance(returns, self.buffer.values)
        }
        
        return stats
    
    def update(self) -> Dict[str, float]:
        """Update the model using PPO loss."""
        obs, actions, old_log_probs, advantages, returns = self.buffer.get()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.config.update_epochs):
            # Create mini-batches
            batch_size = len(obs)
            indices = torch.randperm(batch_size, device=self.device)
            
            for start_idx in range(0, batch_size, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                policy_logits, values = self.model(batch_obs)
                values = values.squeeze(-1)
                
                # Compute policy loss
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.value_clip:
                    value_pred_clipped = batch_returns + torch.clamp(
                        values - batch_returns, -self.config.clip_ratio, self.config.clip_ratio
                    )
                    value_loss1 = (values - batch_returns) ** 2
                    value_loss2 = (value_pred_clipped - batch_returns) ** 2
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                
                if self.config.use_huber_vf:
                    value_loss = F.huber_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.config.vf_coef * value_loss + 
                            self.config.ent_coef * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
                # KL divergence
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    total_kl_div += kl_div.item()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Compute final statistics
        n_updates = self.config.update_epochs * (batch_size // self.config.minibatch_size)
        
        stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'kl_divergence': total_kl_div / n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute explained variance."""
        var_y = torch.var(y_true)
        return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
    
    def save(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

