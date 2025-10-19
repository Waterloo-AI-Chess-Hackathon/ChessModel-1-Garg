#!/usr/bin/env python3
"""
Training script for Chess MCTS-based AlphaZero-style agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import chess
import random
from collections import deque

from model import ChessModel
from mcts import MCTS
from chess_env import ChessEnv
from enc_dec import encode_state, decode


class SelfPlayGame:
    """Represents a single self-play game."""
    
    def __init__(self):
        self.positions = []  # List of (board, policy, value) tuples
        self.result = 0.0  # Game result from white's perspective
        self.moves = []  # List of moves made
    
    def add_position(self, board: chess.Board, policy: Dict[chess.Move, float], value: float):
        """Add a position to the game."""
        self.positions.append((board.copy(), policy.copy(), value))
    
    def add_move(self, move: chess.Move):
        """Add a move to the game."""
        self.moves.append(move)
    
    def set_result(self, result: float):
        """Set the game result."""
        self.result = result


class AlphaZeroTrainer:
    """AlphaZero-style trainer using MCTS for self-play."""
    
    def __init__(self, device: str = "cuda", lr: float = 1e-3, 
                 num_simulations: int = 200, num_games: int = 100,
                 buffer_size: int = 10000, batch_size: int = 32,
                 num_workers: int = 1):
        self.device = torch.device(device)
        self.lr = lr
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_workers = max(1, int(num_workers))
        
        self.model = ChessModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.training_buffer = deque(maxlen=buffer_size)
        
        # MCTS configuration
        self.mcts_config = {
            'c_puct': 1.0,
            'num_simulations': num_simulations,
            'temperature': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25
        }
        
        print(f"AlphaZero trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def play_self_game(self) -> SelfPlayGame:
        """Play a single self-play game using MCTS."""
        game = SelfPlayGame()
        env = ChessEnv()
        obs, _ = env.reset()
        
        # Set model to eval mode for self-play
        self.model.eval()
        
        # Initialize MCTS
        mcts = MCTS(
            model=self.model,
            device=self.device,
            **self.mcts_config
        )
        
        done = False
        step_count = 0
        max_steps = 200  # Prevent infinite games
        
        while not done and step_count < max_steps:
            # Get current board state for MCTS
            board = env.board
            
            # Temperature scheduling: use temp=1.0 for first 30 moves, then 0.1
            current_temp = 1.0 if step_count < 30 else 0.1
            mcts.temperature = current_temp
            
            # Get MCTS move and probabilities
            best_move, move_probs = mcts.search(board)
            
            # Convert move to action index
            action = self._move_to_action_index(best_move, board)
            if action is None:
                # Fallback to random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    best_move = random.choice(legal_moves)
                    action = self._move_to_action_index(best_move, board)
                else:
                    break
            
            # Add position to game
            value = self._evaluate_position(board)
            game.add_position(board, move_probs, value)
            game.add_move(best_move)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        # Set game result from environment
        if env.board.is_checkmate():
            result = -1.0 if env.board.turn == chess.WHITE else 1.0
        else:
            result = 0.0  # Draw
        
        game.set_result(result)
        return game
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a position using the neural network."""
        with torch.no_grad():
            history = [board]
            state_tensor = encode_state(history, device=self.device).unsqueeze(0)
            _, value = self.model(state_tensor)
            return value.item()
    
    def collect_training_data(self, num_games: int) -> List[SelfPlayGame]:
        """Collect training data through self-play.
        Runs multiple games concurrently when num_workers > 1.
        """
        print(f"Collecting {num_games} self-play games with {self.num_workers} worker(s)...")
        games: List[SelfPlayGame] = []

        if self.num_workers <= 1:
            # Fallback to serial execution
            for i in range(num_games):
                if i % 10 == 0:
                    print(f"Playing game {i+1}/{num_games}")
                game = self.play_self_game()
                games.append(game)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.play_self_game) for _ in range(num_games)]
                completed = 0
                for future in as_completed(futures):
                    game = future.result()
                    games.append(game)
                    completed += 1
                    print(f"Completes games: {completed}/{num_games}")
                    # if completed % 10 == 0 or completed == num_games:
                    #     print(f"Completed games: {completed}/{num_games}")

        # Append to training buffer on the main thread
        for game in games:
            for board, policy, value in game.positions:
                # Convert policy dict to tensor
                policy_tensor = torch.zeros(4672, device=self.device)
                for move, prob in policy.items():
                    move_idx = self._move_to_index(move, board)
                    if move_idx is not None:
                        policy_tensor[move_idx] = prob

                # Value from perspective of side to move in this position
                result_from_perspective = game.result if board.turn == chess.WHITE else -game.result

                self.training_buffer.append({
                    'state': encode_state([board], device=self.device),
                    'policy': policy_tensor,
                    'value': torch.tensor(result_from_perspective, dtype=torch.float32, device=self.device)
                })

        return games
    
    def _move_to_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        """Convert move to policy index."""
        try:
            from enc_dec import _move_to_policy_index
            return _move_to_policy_index(move, board)
        except:
            return None
    
    def _move_to_action_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        """Convert move to action index for ChessEnv."""
        try:
            from enc_dec import _move_to_policy_index
            return _move_to_policy_index(move, board)
        except:
            return None
    
    def train_model(self, epochs: int = 1) -> Dict[str, float]:
        """Train the model on collected data."""
        if len(self.training_buffer) < self.batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
        
        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        # Create batches
        buffer_list = list(self.training_buffer)
        random.shuffle(buffer_list)
        
        for i in range(0, len(buffer_list), self.batch_size):
            batch = buffer_list[i:i + self.batch_size]
            if len(batch) < self.batch_size:
                break
            
            # Prepare batch
            states = torch.stack([item['state'] for item in batch])
            target_policies = torch.stack([item['policy'] for item in batch])
            target_values = torch.stack([item['value'] for item in batch])
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_policies, pred_values = self.model(states)
            
            # Calculate losses
            # Policy loss: Use KL divergence (target_policies are probabilities from MCTS)
            policy_loss = F.kl_div(
                F.log_softmax(pred_policies, dim=1),
                target_policies,
                reduction='batchmean'
            )
            
            # Value loss: MSE between predicted and target values
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            
            total_loss = policy_loss + value_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        self.model.eval()
        
        return {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'total_loss': (total_policy_loss + total_value_loss) / max(num_batches, 1)
        }
    
    def evaluate_model(self, num_games: int = 10) -> Dict[str, float]:
        """Evaluate the model against a random opponent."""
        self.model.eval()
        
        wins = 0
        draws = 0
        losses = 0
        
        for _ in range(num_games):
            result = self._play_evaluation_game()
            if result > 0:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
        
        return {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games
        }
    
    def _play_evaluation_game(self) -> float:
        """Play a game against random opponent using ChessEnv."""
        env = ChessEnv()
        obs, _ = env.reset()
        
        # Use lower temperature and fewer simulations for evaluation
        eval_config = self.mcts_config.copy()
        eval_config['temperature'] = 0.1  # More deterministic during evaluation
        eval_config['num_simulations'] = 400 # Fewer simulations for faster eval
        
        mcts = MCTS(
            model=self.model,
            device=self.device,
            **eval_config
        )
        
        done = False
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            board = env.board
            
            if board.turn == chess.WHITE:
                # Our model plays white
                best_move, _ = mcts.search(board)
                action = self._move_to_action_index(best_move, board)
                if action is None:
                    # Fallback to random move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        best_move = random.choice(legal_moves)
                        action = self._move_to_action_index(best_move, board)
                    else:
                        break
            else:
                # Random opponent plays black
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    best_move = random.choice(legal_moves)
                    action = self._move_to_action_index(best_move, board)
                else:
                    break
            
            if action is None:
                break
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        # Return result from white's perspective
        if env.board.is_checkmate():
            return -1.0 if env.board.turn == chess.WHITE else 1.0
        else:
            return 0.0
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def train_alphazero(
    total_iterations: int = 100,
    games_per_iteration: int = 100,
    training_epochs: int = 1,
    save_freq: int = 10,
    eval_freq: int = 5,
    device: str = "cuda",
    num_workers: int = 1,
):
    """Train an AlphaZero-style chess model."""
    
    trainer = AlphaZeroTrainer(
        device=device,
        lr=1e-3,
        num_simulations=100,
        num_games=games_per_iteration,
        num_workers=num_workers,
    )
    
    print(f"Starting AlphaZero training for {total_iterations} iterations")
    print(f"Device: {device}")
    print(f"Games per iteration: {games_per_iteration}")
    
    start_time = time.time()
    
    for iteration in range(total_iterations):
        print(f"\n=== Iteration {iteration + 1}/{total_iterations} ===")
        
        games = trainer.collect_training_data(games_per_iteration)
        print(f"Collected {len(games)} games")
        print(f"Training buffer size: {len(trainer.training_buffer)}")
        
        if len(trainer.training_buffer) >= trainer.batch_size:
            train_stats = trainer.train_model(training_epochs)
            print(f"Training - Policy Loss: {train_stats['policy_loss']:.4f} | "
                  f"Value Loss: {train_stats['value_loss']:.4f} | "
                  f"Total Loss: {train_stats['total_loss']:.4f}")
        
        if iteration % eval_freq == 0:
            eval_stats = trainer.evaluate_model(num_games=10)
            print(f"Evaluation - Win Rate: {eval_stats['win_rate']:.2f} | "
                  f"Draw Rate: {eval_stats['draw_rate']:.2f} | "
                  f"Loss Rate: {eval_stats['loss_rate']:.2f}")
        
        # Save model
        if iteration % save_freq == 0:
            save_path = f"alphazero_model_iter_{iteration}.pt"
            trainer.save_model(save_path)
        
        # Log timing
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.1f}s")
    
    # Final save
    final_save_path = f"alphazero_final_model.pt"
    trainer.save_model(final_save_path)
    print(f"\nTraining completed! Final model saved to {final_save_path}")


def evaluate_agent(trainer: AlphaZeroTrainer, n_eval_episodes: int = 10) -> float:
    """Evaluate the agent."""
    eval_stats = trainer.evaluate_model(num_games=n_eval_episodes)
    return eval_stats['win_rate']


if __name__ == "__main__":
    seed = int((time.time() % (2**32 - 1)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Current CUDA devide name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    train_alphazero(
        total_iterations=500,
        games_per_iteration=100,
        training_epochs=10,
        save_freq=50,
        eval_freq=25,
        device=device
    )
