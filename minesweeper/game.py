import itertools
import random

import torch

BOMB_CELL = 9
HIDDEN_CELL = 10
NUM_CLASSES = 11

class Game:
    def __init__(self, width: int = 10, height: int = 10, mines: int = 10):
        assert mines <= width * height
        self.mines = set()
        self.width = width
        self.height = height
        self.free_tiles = {(x, y) for x in range(self.width) for y in range(self.height)}
        for _ in range(mines):
            new_mine = random.choice(list(self.free_tiles))
            self.mines.add(new_mine)
            self.free_tiles.remove(new_mine)
        self.revealed = [[False for _y in range(height)] for _x in range(width)]
        self.precalculated = [[self._get_cell_number(x, y) for y in range(height)] for x in range(width)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_cell_number(self, x: int, y: int) -> int:
        if (x, y) in self.mines:
            return BOMB_CELL
        res = 0
        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            if (x + dx, y + dy) in self.mines:
                res += 1
        return res

    def _crumble(self, x: int, y: int):
        if not self._in_bounds(x, y) or self.revealed[x][y]:
            return
        self.revealed[x][y] = True
        self.free_tiles.remove((x, y))
        if self.precalculated[x][y] != 0:
            return
        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            self._crumble(x + dx, y + dy)

    def make_move(self, x: int, y: int) -> int:
        assert self._in_bounds(x, y)
        if self.revealed[x][y]:
            return 0
        if (x, y) in self.mines:
            self.revealed[x][y] = True
            return -1
        self._crumble(x, y)
        return 1

    @property
    def finished(self) -> bool:
        blew_up = any(self.revealed[x][y] for x, y in self.mines)
        free_tiles_left = len(self.free_tiles) != 0
        return blew_up or not free_tiles_left

    @property
    def has_blew_up(self) -> bool:
        return any(self.revealed[x][y] for x, y in self.mines)

    def get_board(self) -> list[list[int]]:
        board = [[self.precalculated[x][y] if self.revealed[x][y] else HIDDEN_CELL for y in range(self.height)] for x in range(self.width)]
        return board

    def get_bitboard(self) -> torch.Tensor:
        board = self.get_board()
        bitboard = F.one_hot(torch.tensor(board), num_classes=NUM_CLASSES)
        bitboard = torch.transpose(bitboard, 2, 0)
        return bitboard.float()

    def random_move(self):
        x, y = random.choice(list(self.free_tiles))
        self.make_move(x, y)

    def get_revealed_bitmask(self) -> torch.Tensor:
        result = torch.zeros((self.height, self.width), dtype=torch.float)
        for x, y in self.free_tiles:
            result[y, x] = 1
        for x, y in self.mines:
            result[y, x] = 1
        return result

    def get_target(self) -> torch.Tensor:
        result = torch.zeros((3, self.height, self.width), dtype=torch.float)
        result[0, :, :] = 1
        for x, y in self.free_tiles:
            result[0, y, x] = 0
            result[1, y, x] = 1
        for x, y in self.mines:
            result[0, y, x] = 0
            result[2, y, x] = 1
        return result