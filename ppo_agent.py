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

expirement_name = "pr_ppo_lings_5"


class PPOAgent(LearningAgent):
    """The start of a basic PPO agent for Starcraft."""
    def __init__(self, screen_width=64, screen_height=64, horizon=40, num_processes=1, expirement_name=expirement_name):
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

        values, new_neglogp, dist_entropy = self.model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)
        _, old_neglogp, _ = self.old_model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)

        ratio = torch.exp(new_neglogp - old_neglogp)
        adv_targ = Variable(advantages.view(-1, 1)).cuda()
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
        policy_entp = (-self.entropy_coef) * dist_entropy.mean()

        value_loss = (values - Variable(st.returns[:self.rollout_step]).cuda()).pow(2).mean()

        self.optimizer.zero_grad()
        total_loss = (value_loss + action_loss + policy_entp)
        total_loss.backward()
        self.optimizer.step()

        self.model.summary.add_scalar_value("reward", int(self.reward))
        self.model.summary.add_scalar_value("loss", total_loss.data[0])
        self.model.summary.add_scalar_value("dist_entropy", policy_entp.data[0])
        self.model.summary.add_scalar_value("value_loss", value_loss.data[0])

        self.rollout_step = 0
