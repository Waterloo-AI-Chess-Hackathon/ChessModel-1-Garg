from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
import chess
import torch
from enc_dec import encode_state, decode

class ChessEnv(gym.Env):
    def __init__(self, size: int = 8) -> None:
        super().__init__()
        self.size = 8  # Chess grid is 8 x 8
        self.board = chess.Board()
        self.move_count = 0
        self.history = []
        
        self.action_space = gym.spaces.Discrete(4672)
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(119, 8, 8), dtype=np.float32
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action index (0-4671) representing a chess move
            
        Returns:
            observation: Current board state encoded as (119, 8, 8) tensor
            reward: Reward for this step
            terminated: Whether the game is over
            truncated: Whether the episode was truncated (max moves reached)
            info: Additional information about the game state
        """
        legal_moves = list(self.board.legal_moves)
        
        if not legal_moves:
            reward = self._calculate_final_reward()
            terminated = True
            truncated = False
            info = {"game_result": self.board.result(claim_draw=True)}
            return self._get_observation(), reward, terminated, truncated, info
        
        policy_logits = torch.zeros(4672)
        policy_logits[action] = 1.0
        
        move, move_probs = decode(policy_logits, self.board)
        
        if move not in legal_moves:
            move = legal_moves[0]
        
        # Execute the move
        self.board.push(move)
        self.move_count += 1
        self.history.append(self.board.copy())
        
        if len(self.history) > 8:
            self.history = self.history[-8:]
        
        terminated = self.board.is_game_over(claim_draw=True)
        truncated = self.move_count >= 600
        
        # Calculate reward
        if terminated:
            reward = self._calculate_final_reward()
        else:
            reward = self._calculate_step_reward()
        
        info = {
            "move_count": self.move_count,
            "game_result": self.board.result(claim_draw=True) if terminated else None,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_draw": self.board.is_game_over(claim_draw=True) and not self.board.is_checkmate()
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary containing 'fen' key for custom starting position
            
        Returns:
            observation: Initial board state
            info: Additional information
        """
        super().reset(seed=seed)
        
        fen = ""
        if options and "fen" in options:
            fen = options["fen"]
        
        # Reset board
        if fen:
            self.board.set_fen(fen)
        else:
            self.board = chess.Board()
        
        self.move_count = 0
        self.history = [self.board.copy()]
        
        info = {
            "move_count": 0,
            "game_result": None,
            "is_check": self.board.is_check(),
            "is_checkmate": False,
            "is_stalemate": False,
            "is_draw": False
        }
        
        return self._get_observation(), info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current board state.
        
        Args:
            mode: Rendering mode ('human' for console output, 'rgb_array' for image)
            
        Returns:
            Rendered board as string or None
        """
        if mode == "human":
            print(f"\nMove {self.move_count}")
            print(self.board)
            print(f"FEN: {self.board.fen()}")
            return None
        elif mode == "rgb_array":
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        """Clean up resources."""
        pass

    def _get_observation(self) -> np.ndarray:
        """Get current observation as encoded board state."""
        if not self.history:
            self.history = [self.board.copy()]
        
        encoded_state = encode_state(self.history)
        return encoded_state.numpy()

    def _calculate_step_reward(self) -> float:
        """Calculate reward for a single step."""
        # Basic reward structure - can be enhanced with more sophisticated evaluation
        if self.board.is_check():
            return -0.1
        return 0.0 

    def _calculate_final_reward(self) -> float:
        """Calculate final reward based on game outcome."""
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -1.0  # Black won
            else:
                return 1.0   # White won
        elif self.board.is_stalemate():
            return 0.0 
        elif self.board.is_insufficient_material():
            return 0.0 
        elif self.board.is_seventyfive_moves():
            return 0.0 
        elif self.board.is_fivefold_repetition():
            return 0.0 
        else:
            return 0.0

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        legal_moves = list(self.board.legal_moves)
        legal_actions = []
        
        for move in legal_moves:
            policy_logits = torch.zeros(4672)
            _, _ = decode(policy_logits, self.board)
            pass
        
        return legal_actions

