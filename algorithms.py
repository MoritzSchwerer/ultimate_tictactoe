import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Tuple


class NN(nn.Module):
    def __init__(self, in_size=253, out_size=1, alpha=0.003):
        super(NN, self).__init__()
        self.alpha = alpha
        self.in_size = in_size
        self.out_size = out_size
        self.main = nn.ModuleList([
            nn.Linear(in_features=in_size, out_features=253),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=253, out_features=253),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=253, out_features=81),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=81, out_features=81),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=81, out_features=9),
            nn.LeakyReLU(),
            nn.Linear(in_features=9, out_features=1),
            nn.Sigmoid(),
        ])
        self.optim = torch.optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x



class Algorithm(ABC):
    def __init__(self, number):
        self.number = number

    @abstractmethod
    def _convert_state(self, state):
        """
        takes the game state and converts it into
        the appropriat format to feed to the algorithm
        """
        pass

    @abstractmethod
    def sample_next_action(self, cur_state=None, pos_next_states=None, greedy=False):
        """
        takes the current state and all the possible_next_states
        and returns index of state that is chosen
        """
        pass

    @abstractmethod
    def reward(self, reward):
        """
        takes the reward for the finished episode and 
        learns the policy
        """
        pass

class Human(Algorithm):
    def __init__(self, number):
        super(Human, self).__init__(number)


    def _convert_state(self, state):
        pass

    def sample_next_action(self, pos_next_states=None, greedy=False, **kwargs):
        grid = int(input("please enter a number for the grid: "))
        cell = int(input("please enter a number for the cell: "))
        return (grid, cell)

    def reward(self, reward):
        pass

class Sarsa(Algorithm):
    def __init__(self, number, network, alpha=0.1, gamma=1, epsilon=0.3, first_move_rndm:float = 0.3):
        super(Sarsa, self).__init__(number)
        self.value_network = network 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_move_rndm = first_move_rndm
        self.reset_episode()

    def reset_episode(self):
        self.past_states = []


    def _convert_state(self, state:Tuple[np.ndarray, int]):
        """
        recieves Tuple[np.array(9,9), int] with board and next cell
        returns the binary encoding for both values as a tensor
        """
        board, cell = state
        board = torch.from_numpy(board)
        board[board==-1] = 0
        if self.number != 1:
            my_cells = board == 2
            others_cells = board == 1
            board[my_cells] = 1
            board[others_cells] = 2
        board_one_hot = F.one_hot(board, num_classes=3).flatten()
        cell_one_hot = F.one_hot(torch.tensor(cell+1), num_classes=10)
        total_state = torch.concatenate([board_one_hot, cell_one_hot])
        return total_state

    def sample_next_action(self, pos_next_states=None, greedy=False, first_move=False, **kwargs):
        encoded_next_states = torch.stack([self._convert_state(pos_next_state) for pos_next_state in pos_next_states], 0)
        with torch.no_grad():
            values = self.value_network(encoded_next_states.float()).flatten()

        if first_move and np.random.uniform() > self.first_move_rndm:
            index = np.random.choice(np.arange(0, len(pos_next_states)))
            self.past_states.append(encoded_next_states[index])
            return index

        # if always greedy or not random sample
        # return greedy action
        if greedy or np.random.uniform() >= self.epsilon:
            index = values.argmax()
            self.past_states.append(encoded_next_states[index])
            return index
        # else take random choice
        else:
            index = np.random.choice(np.arange(0, len(pos_next_states)))
            self.past_states.append(encoded_next_states[index])
            return index

    def reward(self, reward):
        for i, state in enumerate(reversed(self.past_states)):
            pred_value = self.value_network(state.float())
            target_value = self.gamma**i * torch.Tensor([reward])

            loss = F.mse_loss(pred_value, target_value)
            # print(f"Player {self.number}: ", target_value.item(), pred_value.item(), loss.item())

            self.value_network.optim.zero_grad()
            loss.backward()
            self.value_network.optim.step()
        self.reset_episode()
    
