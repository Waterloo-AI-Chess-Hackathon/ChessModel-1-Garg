import chess
from typing import List
import random

class GameStorage:
    def __init__(self):
        with open("data/games.txt", "r") as f:
            self.games = f.readlines()
    
    def write_game(self, game: chess.Board) -> None:
        with open("data/games.txt", "a") as f:
            f.write(game.fen() + "\n")
            f.flush()

        self.games.append(game.fen())
        if len(self.games) > 10000:
            self.games.pop(0)
    
    def get_relevant_games(self) -> List[chess.Board]:
        return random.sample(self.games, len(self.games) if len(self.games) < 9 else 9)

if __name__ == "__main__":
    game_storage = GameStorage()
    game_storage.write_game(chess.Board())

    print(game_storage.get_relevant_games())
