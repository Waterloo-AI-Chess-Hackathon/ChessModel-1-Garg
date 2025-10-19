import chess
import torch

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_14_planes(board: chess.Board) -> torch.Tensor:
    """
    14 planes for a single board position:
      - 12 planes: [white P..K (6), black P..K (6)]
      - 2 planes:
          * plane 12: side to move (all-ones if white to move, else zeros)
          * plane 13: repetition flag (all-ones if position is a repetition, else zeros)
    Shape: (14, 8, 8)
    """
    planes = torch.zeros(14, 8, 8, dtype=torch.float32)

    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for t_idx, ptype in enumerate(PIECE_TYPES):
            plane_idx = color_idx * 6 + t_idx
            for sq in board.pieces(ptype, color):
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                planes[plane_idx, 7 - r, f] = 1.0

    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # repetition, another repeat = draw
    planes[13, :, :] = 1.0 if board.is_repetition(2) else 0.0

    return planes

def aux_planes(board: chess.Board) -> torch.Tensor:
    """
    Extra planes to reach 119 total:
      112 from history (8 * 14)
      + 7 aux planes here:
        0: white kingside castling right
        1: white queenside castling right
        2: black kingside castling right
        3: black queenside castling right
        4: fifty-move clock normalized (filled with value)
        5: move number normalized (filled with value)
        6: (spare) e.g., no-legal-move or check flag
    Shape: (7, 8, 8)
    """
    planes = torch.zeros(7, 8, 8, dtype=torch.float32)
    planes[0].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[1].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[2].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[3].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    fifty = min(board.halfmove_clock, 100) / 100.0
    planes[4].fill_(fifty)

    # move number normalization
    move_no = min(board.fullmove_number, 200) / 200.0
    planes[5].fill_(move_no)

    # check flag
    planes[6].fill_(1.0 if board.is_check() else 0.0)
    return planes

def encode_state(
    boards: list[chess.Board],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Encode a single sample (most recent board is boards[-1]) to (119, 8, 8).
    Collect up to 8 past positions (pad with earliest if fewer).
    """
    T = 8
    history = boards[-T:] if len(boards) >= T else [boards[0]] * (T - len(boards)) + boards
    # Most recent first, per AlphaZero convention
    history = list(reversed(history))

    planes = [board_to_14_planes(b) for b in history]  # 8 * (14,8,8)
    feat = torch.cat(planes, dim=0)                    # (112, 8, 8)
    feat = torch.cat([feat, aux_planes(boards[-1])], dim=0)  # (119, 8, 8)
    if device is not None:
        feat = feat.to(device)
    return feat

def encode_batch(
    batch_of_histories: list[list[chess.Board]],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    batch_of_histories: list of histories, one per sample
    Returns: (B, 119, 8, 8)
    """
    feats = [encode_state(hist, device=device) for hist in batch_of_histories]
    return torch.stack(feats, dim=0)


NUM_POLICY_PLANES = 73
NUM_SQUARES = 64

SLIDING_DIRECTIONS = [
    (0, 1),   # north
    (0, -1),  # south
    (1, 0),   # east
    (-1, 0),  # west
    (1, 1),   # north-east
    (1, -1),  # south-east
    (-1, 1),  # north-west
    (-1, -1), # south-west
]
SLIDING_DIRECTION_INDEX = {direction: i for i, direction in enumerate(SLIDING_DIRECTIONS)}

KNIGHT_OFFSETS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
KNIGHT_OFFSET_INDEX = {offset: i for i, offset in enumerate(KNIGHT_OFFSETS)}

PROMOTION_DX = {
    chess.WHITE: [(-1, 1), (0, 1), (1, 1)],
    chess.BLACK: [(-1, -1), (0, -1), (1, -1)],
}
PROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMOTION_PIECE_INDEX = {piece: idx for idx, piece in enumerate(PROMOTION_PIECES)}


def _move_to_policy_index(move: chess.Move, board: chess.Board | None = None) -> int | None:
    """Return flattened policy index (0-4671) for a move using AlphaZero mapping."""
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    dx = to_file - from_file
    dy = to_rank - from_rank

    plane: int | None = None

    if move.promotion and move.promotion != chess.QUEEN:
        color = chess.WHITE if dy > 0 else chess.BLACK
        if board is not None:
            piece_color = board.color_at(from_sq)
            if piece_color is not None:
                color = piece_color

        directions = PROMOTION_DX[color]
        try:
            dir_idx = directions.index((dx, dy))
        except ValueError:
            return None

        piece_idx = PROMOTION_PIECE_INDEX.get(move.promotion)
        if piece_idx is None:
            return None

        plane = 64 + piece_idx * 3 + dir_idx
    elif (dx, dy) in KNIGHT_OFFSET_INDEX:
        plane = 56 + KNIGHT_OFFSET_INDEX[(dx, dy)]
    else:
        distance = max(abs(dx), abs(dy))
        if distance == 0 or distance > 7:
            return None

        step_dx = 0 if dx == 0 else dx // abs(dx)
        step_dy = 0 if dy == 0 else dy // abs(dy)
        direction = (step_dx, step_dy)

        if direction not in SLIDING_DIRECTION_INDEX:
            return None

        # Verify the move aligns exactly with the direction (no L-shaped skips).
        if step_dx * distance != dx or step_dy * distance != dy:
            return None

        plane = SLIDING_DIRECTION_INDEX[direction] * 7 + (distance - 1)

    # if plane is None:
    #     return None

    return from_sq * NUM_POLICY_PLANES + plane


def decode(
    policy_logits: torch.Tensor,
    board: chess.Board,
    temperature: float = 1.0,
) -> tuple[chess.Move | None, dict[chess.Move, float]]:
    """
    Mask policy logits by legal moves and return a sampled move distribution

    Args:
        policy_logits: Tensor of shape (4672,) or (1, 4672) containing raw policy logits.
        board: Current chess position to evaluate moves for.
        temperature: Optional softmax temperature for exploration (>0).

    Returns:
        A tuple of (best_move, move_probabilities) where move_probabilities maps each
        legal move to its normalized probability mass. best_move is the move with the
        highest probability or None if no legal move is found/mapped.
    """

    if policy_logits.dim() == 2:
        logits = policy_logits[0]
    else:
        logits = policy_logits

    logits = logits.flatten()

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=0)

    move_probs: dict[chess.Move, float] = {}
    for move in board.legal_moves:
        idx = _move_to_policy_index(move, board)
        if idx is None:
            continue
        move_probs[move] = probs[idx].item()

    total = sum(move_probs.values())
    if total > 0:
        for move in move_probs:
            move_probs[move] /= total
    elif move_probs:
        uniform = 1.0 / len(move_probs)
        for move in move_probs:
            move_probs[move] = uniform

    best_move = None
    if move_probs:
        best_move = max(move_probs.items(), key=lambda item: item[1])[0]

    return best_move, move_probs
