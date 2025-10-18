"""
Collecting some data on inference 
"""

import time
from model import ChessModel
import torch
from test import choose_move 
import chess

def time_per_move():
    device = torch.device("mps")
    model = ChessModel().to(device)
    model.eval()
    board = chess.Board()
    history = [board.copy()]

    # Inference
    start_time = time.time() 

    choose_move(model, board, history, device)

    end_time = time.time()

    print(f"{end_time - start_time} milliseconds") 

    return end_time - start_time

if __name__ == '__main__':
    total = 0
    for _ in range(100):
        total += time_per_move()
    
    print(f"Average for 100 moves {total / 100.0}")
    

