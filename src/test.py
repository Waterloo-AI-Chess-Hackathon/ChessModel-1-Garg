import random

import chess
import torch
import os

from enc_dec import encode_state, decode
from model import ChessModel
from mcts import MCTS


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_move(mcts: MCTS, board: chess.Board) -> chess.Move:
    best_move, _ = mcts.search(board)
    if best_move is not None:
        return best_move
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves available to choose from.")
    return random.choice(legal_moves)


def load_model_weights(model: ChessModel, device: torch.device, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)


def play_self_game(max_moves: int = 200) -> tuple[str, list[str], chess.Board]:
    device = select_device()
    model = ChessModel().to(device)
    model.eval()
    # Attempt to load pretrained weights
    try:
        load_model_weights(model, device, "/Users/ash/Documents/ChessModel-1-Garg/alphazero_model_iter_0.pt")
        print("Loaded model weights from alphazero_model_iter_0.pt")
    except Exception as e:
        print(f"Warning: failed to load weights: {e}. Using randomly initialized model.")

    board = chess.Board()
    history = [board.copy()]
    moves: list[str] = []

    # Initialize MCTS
    mcts = MCTS(
        model=model,
        device=device,
        c_puct=1.0,
        num_simulations=10,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Temperature schedule: explore early, become more deterministic later
        ply = len(moves)
        mcts.temperature = 1.0 if ply < 30 else 0.1

        move = choose_move(mcts, board)
        moves.append(move.uci())
        board.push(move)

        os.system("clear")
        print(board)

        history.append(board.copy())
        if len(history) > 8:
            history = history[-8:]

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    return result, moves, board


if __name__ == "__main__":
    from chess.pgn import Game
    result, moves, board = play_self_game()
    print(f"Result: {result}")
    print(f"Total moves: {len(moves)}")
    print("Move list:")
    print(" ".join(moves))
    print("\nFinal board:")
    print(board)
    print("PGN:")
    print(Game.from_board(board))