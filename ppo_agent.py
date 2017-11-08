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

expirement_name = "ppo_zerglings_pre"

class PPOAgent(LearningAgent):
    """The start of a basic PPO agent for Starcraft."""
    def __init__(self, screen_width=84, screen_height=84, horizon=64, num_processes=1):
        super(PPOAgent, self)
        num_functions = len(actions.FUNCTIONS)
        self.model = PPOModel(num_functions=num_functions, expirement_name=expirement_name).cuda()
        self.old_model = copy.deepcopy(self.model)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.horizon = horizon
        self.rollout_step = 0
        #self.model.load_state_dict(torch.load(f"./{expirement_name}.pth"))
        #self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.tau = 0.97
        self.entropy_coef = 0.01
        self.clip_param = 0.2
        self.saved_actions = SavedActions(self.horizon,
                                          1,
                                          num_functions)
        self.saved_actions.cuda()
        self.episode_rewards = torch.zeros(1, 1)

    def step(self, obs):
        super(PPOAgent, self).step(obs)
        self.rollout_step += 1

        reward = torch.from_numpy(np.expand_dims(obs.reward, 1)).float()
        self.episode_rewards += reward

        screen, minimap, game, allowed_actions = self.preprocess(obs)

        value_pred, action, x1, y1 = self.model.act(Variable(screen).cuda(),
                                                    Variable(minimap).cuda(),
                                                    Variable(game).cuda(),
                                                    Variable(allowed_actions).cuda())
        function_id = action.data[0]
        act_args = []
        for args in actions.FUNCTIONS[function_id].args:
            if args.name in ['screen', 'minimap', 'screen2']:
                act_args.append([x1.data[0,0], y1.data[0,0]])
            else:
                act_args.append([0])

        mask = torch.FloatTensor([0.0] if obs.step_type == 2 else [1.0])
        self.saved_actions.insert(self.rollout_step, screen, minimap, game, action.data, x1.data, y1.data, value_pred.data, reward, mask)

        if self.rollout_step == self.horizon:
            self.rollout()

        return actions.FunctionCall(function_id, act_args)

    def reset(self):
        super(PPOAgent, self).reset()
        print(f"Episode Reward was: {self.episode_rewards[0,0]}")
        self.model.summary.add_scalar_value("episode_reward", int(self.episode_rewards[0,0]))
        self.episode_rewards = torch.zeros(1, 1)

        if self.steps > 1:
            self.rollout()

        torch.save(self.model.state_dict(), f"./{expirement_name}.pth")
        self.saved_actions.reset()

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

        values, action_log_probs, dist_entropy = self.model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)
        _, old_action_log_probs, _ = self.old_model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)

        ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
        adv_targ = Variable(advantages.view(-1, 1)).cuda()
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

        value_loss = (Variable(st.returns[:self.rollout_step]).cuda() - values).pow(2).mean()

        self.optimizer.zero_grad()
        total_loss = (value_loss + action_loss - dist_entropy * self.entropy_coef)
        total_loss.backward()
        self.optimizer.step()

        self.model.summary.add_scalar_value("reward", int(self.reward))
        self.model.summary.add_scalar_value("loss", total_loss.data[0])
        self.model.summary.add_scalar_value("dist_entropy", dist_entropy.data[0])

        self.rollout_step = 0
