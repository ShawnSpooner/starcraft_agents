import torch
import numpy as np

class SavedActions(object):
    def __init__(self, num_steps, num_processes, action_space):
        self.screens = torch.zeros(num_steps + 1, num_processes, 16, 32, 32)
        self.minimaps = torch.zeros(num_steps + 1, num_processes, 16, 32, 32)
        self.games = torch.zeros(num_steps + 1, num_processes, 11)
        self.masks = torch.zeros(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.values = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.actions = torch.zeros(num_steps + 1, num_processes, 1).long()
        self.x1s = torch.zeros(num_steps + 1, num_processes, 1).long()
        self.y1s = torch.zeros(num_steps + 1, num_processes, 1).long()
        self.num_steps = num_steps

    def cuda(self):
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.x1s = self.x1s.cuda()
        self.y1s = self.y1s.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, screen, minimap, gs, action, x1, y1, value_pred, reward, mask):
        self.screens[step].copy_(screen)
        self.minimaps[step].copy_(minimap)
        self.games[step].copy_(gs)
        self.values[step].copy_(value_pred)
        self.actions[step].copy_(action)
        self.x1s[step].copy_(x1)
        self.y1s[step].copy_(y1)
        self.rewards[step - 1].copy_(reward)
        self.masks[step].copy_(mask)

    def reset(self, index=1):
        self.screens[index:] *= 0
        self.minimaps[index:] *= 0
        self.games[index:] *= 0
        self.values[index:] *= 0
        self.actions[index:] *= 0
        self.x1s[index:] *= 0
        self.y1s[index:] *= 0
        self.rewards[index:] *= 0
        self.returns[index:] *= 0
        self.masks[index:] *= 0

    def compute_returns(self, end_step, gamma, tau):
        returns = torch.Tensor(end_step, 1)
        deltas = torch.Tensor(end_step, 1)
        advantages = torch.Tensor(end_step, 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(end_step)):
            self.returns[i] = self.rewards[i] + gamma * prev_return
            deltas[i] = self.rewards[i] + gamma * prev_value - self.values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage

            prev_return = self.returns[i, 0]
            prev_value = self.values[i, 0]
            prev_advantage = advantages[i, 0]

        return advantages
