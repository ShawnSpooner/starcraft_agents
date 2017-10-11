import torch
import numpy as np

class SavedActions(object):
    def __init__(self, num_steps, num_processes, action_space):
        self.screens = torch.zeros(num_steps + 1, num_processes, 13, 84, 84)
        self.minimaps = torch.zeros(num_steps + 1, num_processes, 7, 84, 84)
        self.games = torch.zeros(num_steps + 1, num_processes, 11)
        self.masks = torch.zeros(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.actions = torch.zeros(num_steps + 1, num_processes, 1).long()
        self.num_steps = num_steps

    def cuda(self):
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, screen, minimap, gs, action, value_pred, reward, mask):
        self.screens[step].copy_(screen)
        self.minimaps[step].copy_(minimap)
        self.games[step].copy_(gs)
        self.value_preds[step].copy_(value_pred)
        self.actions[step].copy_(action)
        self.rewards[step - 1].copy_(reward)
        self.masks[step].copy_(mask)

    def reset(self, index=1):
        self.screens[index:] *= 0
        self.minimaps[index:] *= 0
        self.games[index:] *= 0
        self.value_preds[index:] *= 0
        self.actions[index:] *= 0
        self.rewards[index:] *= 0
        self.masks[index:] *= 0

    def compute_returns(self, next_value, gamma, tau):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step] + self.rewards[step]
