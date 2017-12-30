import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, num_steps, num_processes, screen_width, screen_height):
        print(f"the number of steps is: {num_steps}")
        self.buffer_size = buffer_size = num_steps + 1
        self.screens = torch.zeros(num_processes, buffer_size, 1, screen_width, screen_height).long()
        self.minimaps = torch.zeros(num_processes, buffer_size, 1, screen_width, screen_height).long()
        self.games = torch.zeros(num_processes, buffer_size, 11)
        self.dones = torch.zeros(num_processes, buffer_size, 1)
        self.returns = torch.zeros(num_processes, buffer_size, 1)
        self.rewards = torch.zeros(num_processes, buffer_size, 1)
        self.advantages = torch.zeros(num_processes, buffer_size, 1)
        self.values = torch.zeros(num_processes, buffer_size, 1)
        self.actions = torch.zeros(num_processes, buffer_size, 1).long()
        self.x1s = torch.zeros(num_processes, buffer_size, 1).long()
        self.y1s = torch.zeros(num_processes, buffer_size, 1).long()
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.idx = 0


    def __getitem__(self, index):
        process, idx = self.indexes[index]
        return (self.screens[process][idx],
                self.minimaps[process][idx],
                self.games[process][idx],
                self.actions[process][idx],
                self.x1s[process][idx],
                self.y1s[process][idx],
                self.rewards[process][idx],
                self.returns[process][idx])

    def __len__(self):
        return len(self.indexes)

    def insert(self, process_num, screen, minimap, gs, action, x1, y1, value_pred, reward, dones):
        self.rewards[process_num][self.idx - 1] = reward
        self.screens[process_num][self.idx] = screen
        self.minimaps[process_num][self.idx] = minimap
        self.games[process_num][self.idx] = gs
        self.values[process_num][self.idx] = value_pred
        self.actions[process_num][self.idx] = action
        self.x1s[process_num][self.idx] = x1
        self.y1s[process_num][self.idx] = y1
        self.dones[process_num][self.idx] = dones

    def step(self):
        self.last_step = self.idx
        self.idx += 1
        # if we have hit our buffer limit, reset
        if self.idx == self.buffer_size:
            self.idx = 0

    def build_view(self):
        self.indexes = [(p, i) for p in range(self.num_processes)
                               for i in self.get_trajectory_idx()]

    def get_trajectory_idx(self):
        return [(self.last_step - i) % self.buffer_size for i in range(self.num_steps)]

    def compute_advantages(self, gamma):
        # build our current view
        self.build_view()

        for p in range(self.num_processes):
            r = 0
            for t in self.get_trajectory_idx():
                self.returns[p][t] = r = self.rewards[p][t] + gamma*r*(1.0 - self.dones[p][t])
