import numpy as np
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Optional, Tuple, List, Union
from algorithms import Algorithm, Sarsa, NN


def state_to_symbol(num):
    match num:
        case -1:
            return ' '
        case 1:
            return 'X'
        case 2:
            return 'O'
        case _:
            raise ValueError(f"board state should never be {num}")


class UltimateTicTacToe:
    def __init__(self, player1:Algorithm, player2:Algorithm) -> None:
        self.player1 = player1
        self.player2 = player2
        self.results = []
        self.reset()

    def step(self, action, position: Optional[Tuple[int, int]] = None):
        assert position is None or position == self.current_position, f"Position: {position} is not equal to {self.current_position}"
        

    def reset(self):
        self.current_player = self.player1
        self.board = np.full((9,9), -1, dtype=np.int64)
        self.master_board = np.full(9, -1, dtype=np.int64)
        self.current_position = -1

    @property
    def _possible_actions(self) -> List[int]:
        valid_i1, valid_i2 = 0, 81
        if self.current_position != -1:
            valid_i1, valid_i2 = self.current_position * 9, self.current_position * 9 + 9
        return [i for i, value in enumerate(self.board.flatten()) if value == -1 and i >= valid_i1 and i < valid_i2]

    def _simulate_action(self, action: int, player: Optional[int] = None):
        """
        returns:
            the board that would result for the action
        """
        new_board = self.board.copy()
        new_board[action//9, action%9] = self.current_player.number if player is None else player.number
        return new_board

    def _perform_action(self, action:Union[int, Tuple[int, int]]) -> bool:
        if isinstance(action, tuple):
            a1, a2 = action
        else:
            a1, a2 = action // 9, action % 9
        if a1 != self.current_position and self.current_position != -1 or self.board[a1, a2] != -1:
            return False
        self.board[a1, a2] = self.current_player.number
        self.current_position = a2
        if self._asses_board(self.board[a2]) != -1:
            self.current_position = -1
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
        return True

    def possible_next_states(self, player: Optional[int] = None) -> List[Tuple[np.ndarray[int], int]]:
        actions = self._possible_actions
        next_states = []
        for action in actions:
            board = self._simulate_action(action, player=player)
            position = action % 9
            # check if the position is available
            if self._asses_board(board[position]) != -1:
                position = -1
            next_states.append((board, position))
        return next_states


    def render(self, board:Optional[np.ndarray] = None):
        board = self.board.reshape((3,3,3,3)) if board is None else board.reshape((3,3,3,3))
        for row in range(3):
            for s_row in range(3):
                if s_row == 0:
                    print('-------------    '*3)
                string = '| '
                for col in range(3):
                    if col != 0:
                        string += '   | '
                    for s_col in range(3):
                        string += str(state_to_symbol(board[row, col, s_row, s_col])) + ' | '
                print(string)
                print('-------------    '*3)
            print()

    @property
    def terminated(self):
        """
        returns:
           -1    if game is still going
            0    if game is a draw
            1    if player 1 won
            2    if player 2 won
        """
        for sub_board in range(9):
            self.master_board[sub_board] = self._asses_board(self.board[sub_board])
        return self._asses_board(self.master_board)


    def _asses_board(self, board: np.ndarray) -> Optional[int]:
        """
        returns:
           -1    if game is still going
            0    if game is a draw
            1    if player 1 won
            2    if player 2 won
        """
        assert board.shape == (9,)
        board = board.reshape(3, 3)
        p1s = board == 1
        p2s = board == 2
        if ((p1s.sum(-1) == 3).any()
            or (p1s.sum(0) == 3).any()
            or p1s[[0, 1, 2], [0, 1, 2]].sum() == 3
            or p1s[[0, 1, 2], [2, 1, 0]].sum() == 3):
            return 1
        elif ((p2s.sum(-1) == 3).any()
            or (p2s.sum(0) == 3).any()
            or p2s[[0, 1, 2], [0, 1, 2]].sum() == 3
            or p2s[[0, 1, 2], [2, 1, 0]].sum() == 3):
            return 2
        elif not (board == -1).any():
            return 0
        else: 
            return -1

    def state_from_action(self, action, player=None):
        board = self.board.copy()
        board[action//9, action%9] = self.current_player.number if player is None else player.number
        return (board, action%9)
        


    def train(self, num_episodes=1):
        self.results = []
        for ep in tqdm(range(num_episodes)):

            player1, player2 = self.player1, self.player2
            # for each episode randomly let the players start
            # if np.random.uniform() > 0.5:
            #     player1, player2 = self.player1, self.player2
            # else:
            #     player1, player2 = self.player2, self.player1

            first_move = True
            while True:
                pos_actions = self._possible_actions
                pos_states = [self.state_from_action(action) for action in pos_actions]
                action_index = player1.sample_next_action(pos_states, (self.board, self.current_position), first_move=first_move)
                _ = self._perform_action(pos_actions[action_index])
                if self.terminated != -1:
                    break


                pos_actions = self._possible_actions
                pos_states = [self.state_from_action(action) for action in pos_actions]
                action_index = player2.sample_next_action(pos_states, (self.board, self.current_position), first_move=first_move)
                _ = self._perform_action(pos_actions[action_index])
                if self.terminated != -1:
                    break
                first_move = False
            term = self.terminated
            self.results.append(term)
            if term == player1.number:
                player1.reward(1)
                player2.reward(0)
            elif term == player2.number:
                player1.reward(1)
                player2.reward(0)
            else:
                player1.reward(0.1)
                player2.reward(0.1)
            self.reset()

    # used only for testing
    def _random_board(self):
        self.board = np.random.randint(-1, 2, size=(9,9))


def heat_map(values):
    res = np.zeros((9, 9))
    vs = np.array(values)
    for i in range(9):
        for j in range(9):
            v_i = (3*(i%3)) + (i // 3) * 27
            v_j = (j // 3) * 6 + j
            res[i, j] = vs[v_i+v_j]
    plt.imshow(res, cmap='hot', interpolation='nearest', vmin=np.min(res), vmax=np.max(res))
    plt.colorbar()
    plt.savefig('./graphs/heat_map.png', dpi=300)

            




if __name__ == '__main__':
    episodes = 1000
    nn = NN(in_size=(9*9*3+9+1), alpha=0.0001)
    p1 = Sarsa(1, nn, gamma=1., epsilon=0.10, first_move_rndm=0.5)
    p2 = Sarsa(2, nn, gamma=1., epsilon=0.10, first_move_rndm=0.5)
    game = UltimateTicTacToe(p1, p2)
    game.train(num_episodes=episodes)
    p1.value_network.eval()
    torch.save(p1.value_network.state_dict(), f"./models/both_players_{episodes}.pth")
    # torch.save(p2.value_network.state_dict(), f"./models/p2_{episodes}.pth")
    pos_actions = game._possible_actions
    pos_states = [game.state_from_action(action) for action in pos_actions]
    conv_states = [p1._convert_state(pos_state) for pos_state in pos_states]
    states = torch.stack(conv_states, 0)
    values = p1.value_network(states.float()).detach().numpy()

    best_action = np.argmax(values)
    best_state = pos_states[best_action][0]
    best_state[best_state == 0] = -1
    game.render(best_state)
    heat_map(values.flatten())
    results = np.array(game.results, dtype=np.int64)
    results = results[results != -1]
    print(results)
    print(np.bincount(results))
    print(values)
    for d in torch.rand(10, 253):
        print(p1.value_network(d))
    # print(pos_states[best_action][1])


    
