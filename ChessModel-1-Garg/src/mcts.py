import math
import random
import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
import chess
from enc_dec import encode_state, decode

class MCTSNode:
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 action: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over(claim_draw=True)
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves from this position."""
        return list(self.board.legal_moves)
    
    def add_child(self, action: chess.Move, prior: float) -> 'MCTSNode':
        """Add a child node for the given action."""
        # Create new board state
        new_board = self.board.copy()
        new_board.push(action)
        
        child = MCTSNode(new_board, parent=self, action=action, prior=prior)
        self.children[action] = child
        return child
    
    def update(self, value: float) -> None:
        """Update this node's statistics after a simulation."""
        self.visit_count += 1
        self.value_sum += value
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """Calculate UCB1 score for this node."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula with prior probability
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero-style chess."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 c_puct: float = 1.0, num_simulations: int = 800,
                 temperature: float = 1.0, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        self.model.eval()
    
    def search(self, board: chess.Board) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """
        Perform MCTS search and return the best move and move probabilities.
        
        Args:
            board: Current chess position
        Returns:
            Tuple of (best_move, move_probabilities)
        """
        root = MCTSNode(board)
        
        self._expand_node(root)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Calculate move probabilities
        move_probs = self._get_move_probabilities(root)
        
        # Select best move
        if self.temperature == 0:
            # Greedy selection
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        else:
            # Temperature-based selection
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            
            # Apply temperature
            if self.temperature > 0:
                logits = torch.tensor([math.log(p + 1e-8) for p in probs]) / self.temperature
                probs = torch.softmax(logits, dim=0).numpy()
            
            # Sample move
            best_move = np.random.choice(moves, p=probs)
        
        return best_move, move_probs
    
    def _simulate(self, root: MCTSNode) -> None:
        """Perform one simulation from the root."""
        node = root

        # Selection phase - traverse down the tree
        while not node.is_leaf() and not node.is_terminal():
            node = self._select_child(node)
        
        # Expansion phase
        if not node.is_terminal():
            if not node.is_expanded:
                self._expand_node(node)
            
            # Select a child for rollout
            if node.children:
                node = self._select_child(node)
        
        # Evaluation phase
        value = self._evaluate_node(node)
        
        # Backpropagation phase
        self._backpropagate(node, value)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child node using UCB1."""
        if not node.children:
            return node
        
        best_child = None
        best_score = float('-inf')
        
        for child in node.children.values():
            score = child.get_ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand a node using neural network predictions."""
        if node.is_expanded or node.is_terminal():
            return
        
        policy, value = self._get_neural_network_prediction(node.board)
        
        # Add Dirichlet noise for exploration
        if node.parent is None:
            legal_moves = node.get_legal_moves()
            if legal_moves:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                
                for i, move in enumerate(legal_moves):
                    move_idx = self._move_to_index(move, node.board)
                    if move_idx is not None:
                        # Blend policy with Dirichlet noise
                        original_prob = policy[move_idx].item()
                        noisy_prob = (1 - self.dirichlet_epsilon) * original_prob + \
                                   self.dirichlet_epsilon * noise[i]
                        policy[move_idx] = torch.tensor(noisy_prob)
        
        # Create children for legal moves
        legal_moves = node.get_legal_moves()
        for move in legal_moves:
            move_idx = self._move_to_index(move, node.board)
            if move_idx is not None:
                prior = policy[move_idx].item()
                node.add_child(move, prior)
        
        node.is_expanded = True
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate a node using neural network or game outcome."""
        if node.is_terminal():
            return self._get_terminal_value(node.board)
        
        # Use neural network evaluation
        _, value = self._get_neural_network_prediction(node.board)
        return value.item()
    
    def _get_neural_network_prediction(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value predictions from the neural network."""
        # Encode the board state
        history = [board]
        state_tensor = encode_state(history, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy, value = self.model(state_tensor)
        
        return policy[0], value[0]
    
    def _get_terminal_value(self, board: chess.Board) -> float:
        """Get the value of a terminal position."""
        if board.is_checkmate():
            # Return value from perspective of the player who just moved
            return -1.0 if board.turn == chess.WHITE else 1.0
        else:
            # Draw
            return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate the value up the tree."""
        while node is not None:
            node.update(value)
            value = -value  # Flip value for opponent
            node = node.parent
    
    def _get_move_probabilities(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """Get move probabilities from visit counts."""
        if not root.children:
            return {}
        
        total_visits = sum(child.visit_count for child in root.children.values())
        
        move_probs = {}
        for move, child in root.children.items():
            if total_visits > 0:
                move_probs[move] = child.visit_count / total_visits
            else:
                move_probs[move] = 1.0 / len(root.children)
        
        return move_probs
    
    def _move_to_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        """Convert a chess move to policy index using the enc_dec module."""
        try:
            # Import the move_to_policy_index function from enc_dec
            from enc_dec import _move_to_policy_index
            return _move_to_policy_index(move, board)
        except ImportError:
            # Fallback: try to find the index by testing each position
            try:
                for i in range(4672):
                    test_policy = torch.zeros(4672)
                    test_policy[i] = 1.0
                    _, test_probs = decode(test_policy, board)
                    if move in test_probs and test_probs[move] > 0:
                        return i
                return None
            except:
                return None


# Example usage:
def example_usage():
    """Example of how to use the MCTS with a chess model."""
    import torch
    from model import ChessModel
    
    # Initialize model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = ChessModel().to(device)
    
    # Initialize MCTS
    mcts = MCTS(
        model=model,
        device=device,
        c_puct=1.0,
        num_simulations=800,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )
    
    # Create a chess board
    board = chess.Board()
    
    # Search for the best move
    best_move, move_probabilities = mcts.search(board)
    
    print(f"Best move: {best_move}")
    print(f"Move probabilities: {move_probabilities}")
    
    return best_move, move_probabilities


if __name__ == "__main__":
    example_usage()