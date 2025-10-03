import random

import chess
import torch

from enc_dec import encode_state, decode
from model import ChessModel


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_move(model: ChessModel, board: chess.Board, history: list[chess.Board], device: torch.device) -> chess.Move:
    input_tensor = encode_state(history, device=device).unsqueeze(0)
    with torch.no_grad():
        policy_logits, _ = model(input_tensor)

    policy_logits = policy_logits.detach().cpu()
    best_move, move_probs = decode(policy_logits, board)

    if best_move is not None:
        return best_move

    # Safety catch
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves available to choose from.")
    return random.choice(legal_moves)


def play_self_game(max_moves: int = 200) -> tuple[str, list[str], chess.Board]:
    device = select_device()
    model = ChessModel().to(device)
    model.eval()

    board = chess.Board()
    history = [board.copy()]
    moves: list[str] = []

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        move = choose_move(model, board, history, device)
        moves.append(move.uci())
        board.push(move)

        history.append(board.copy())
        if len(history) > 8:
            history = history[-8:]

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    return result, moves, board


if __name__ == "__main__":
    result, moves, board = play_self_game()
    print(f"Result: {result}")
    print(f"Total moves: {len(moves)}")
    print("Move list:")
    print(" ".join(moves))
    print("\nFinal board:")
    print(board)
