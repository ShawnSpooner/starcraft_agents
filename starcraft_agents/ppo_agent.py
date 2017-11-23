import numpy as np
import copy
from collections import namedtuple
from pysc2.agents import base_agent
from pysc2.lib import actions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from starcraft_agents.ppo_model import PPOModel
from starcraft_agents.learning_agent import LearningAgent
from starcraft_agents.saved_actions import SavedActions

expirement_name = "ppo_lings_1"


class PPOAgent(LearningAgent):
    """The start of a basic PPO agent for Starcraft."""
    def __init__(self, screen_width=64, screen_height=64, horizon=128, num_processes=1, expirement_name=expirement_name):
        super(PPOAgent, self).__init__(expirement_name)
        num_functions = len(actions.FUNCTIONS)
        self.model = PPOModel(num_functions=num_functions, expirement_name=expirement_name).cuda()
        self.old_model = copy.deepcopy(self.model)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.horizon = horizon
        #self.model.load_state_dict(torch.load(f"./{expirement_name}.pth"))
        #self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.tau = 0.97
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.clip_param = 0.2
        self.saved_actions = SavedActions(self.horizon,
                                          1,
                                          num_functions)
        self.saved_actions.cuda()
        self.episode_rewards = torch.zeros(1, 1)

    def rollout(self):
        if self.rollout_step < 1:
            self.rollout_step = 0
            return

        st = self.saved_actions
        advantages = st.compute_returns(self.rollout_step, self.gamma, self.tau)
        self.old_model.load_state_dict(self.model.state_dict())

        screens = Variable(st.screens[:self.rollout_step].squeeze(1)).cuda()
        minimaps = Variable(st.minimaps[:self.rollout_step].squeeze(1)).cuda()
        games = Variable(st.games[:self.rollout_step].squeeze(1)).cuda()
        actions = Variable(st.actions[:self.rollout_step].squeeze(1)).cuda()
        x1s = Variable(st.x1s[:self.rollout_step].squeeze(1)).cuda()
        y1s = Variable(st.y1s[:self.rollout_step].squeeze(1)).cuda()

        values, entropy, lp, x_lp, y_lp = self.model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)
        old_values, _, old_lp, old_xlp, old_ylp = self.old_model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)

        adv = Variable(advantages.view(-1, 1)).cuda()
        R = Variable(st.rewards[::self.rollout_step]).cuda()
        vpredclipped = old_values + torch.clamp(values - old_values, - self.clip_param, self.clip_param)
        vf_losses1 = (values - R).pow(2)
        vf_losses2 = (vpredclipped - R).pow(2)
        vf_loss = .5 * torch.max(vf_losses1, vf_losses2).mean()

        def ppo_loss(lp, oldlp):
            ratio = torch.exp(oldlp - lp)
            pg_losses = -adv * ratio
            pg_losses2 = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            pg_loss = torch.max(pg_losses, pg_losses2).mean()
            #approxkl = .5 * (lp - old_lp).pow(2).mean()
            #clipfrac = torch.max(torch.abs(ratio - 1.0), self.clip_param).mean()
            return pg_loss

        dist_entp = self.entropy_coef * entropy.mean()
        function_loss = ppo_loss(lp, old_lp)
        x1_loss = ppo_loss(lp, old_lp)
        y1_loss = ppo_loss(lp, old_lp)

        self.optimizer.zero_grad()
        total_loss = function_loss + x1_loss + y1_loss - dist_entp + vf_loss * self.value_coef
        total_loss.backward()
        self.optimizer.step()

        self.model.summary.add_scalar_value("reward", int(self.reward))
        self.model.summary.add_scalar_value("loss", total_loss.data[0])
        self.model.summary.add_scalar_value("dist_entropy", dist_entp.data[0])
        self.model.summary.add_scalar_value("value_loss", vf_loss.data[0])

        self.rollout_step = 0
