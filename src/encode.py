import chess
import torch

def encode_board(board: chess.Board) -> torch.Tensor:
    # 12 pieces + 2 pawns + 1 empty square
    board_tensor = torch.zeros(8, 8, 12)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = square // 8
            file = square % 8
            # Encode piece type (1-6) and color (white=0-5, black=6-11)
            piece_index = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
            board_tensor[rank][file][piece_index] = 1
    return board_tensor

