import numpy as np
from collections import namedtuple
from pysc2.agents import base_agent
from pysc2.lib import actions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from starcraft_agents.a2c_model import A2CModel
from starcraft_agents.fully_conv_model import FullyConvModel
from starcraft_agents.learning_agent import LearningAgent
from starcraft_agents.saved_actions import SavedActions

expirement_name = "collect_minerals"

class A2CAgent(LearningAgent):
    """The start of a basic A2C agent for learning agents."""
    def __init__(self, screen_width=84, screen_height=84, num_steps=64, num_processes=1, fully_conv=False):
        super(A2CAgent, self)
        num_functions = len(actions.FUNCTIONS)
        if fully_conv:
            self.model = FullyConvModel(num_functions=num_functions).cuda()
        else:
            self.model = A2CModel(num_functions=num_functions, expirement_name=expirement_name).cuda()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.max_grad = 0.5
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.episode_rewards = torch.zeros([num_processes, 1])
        self.final_rewards = torch.zeros([num_processes, 1])
        self.gamma = 0.99
        self.tau = 0.97
        self.saved_actions = SavedActions(self.num_steps,
                                          self.num_processes,
                                          num_functions)
        self.saved_actions.cuda()
        self.rollout_step = 0
        #self.model.load_state_dict(torch.load(f"./{expirement_name}.pth"))
        #self.model.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                        lr=7e-4,
                                        eps=1e-10,
                                        alpha=0.99)
        self.episode_rewards = torch.zeros(1, 1)
        self.final_rewards = torch.zeros(1, 1)

    def step(self, obs):
        super(A2CAgent, self).step(obs)
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

        masks = torch.FloatTensor([0.0] if obs.step_type == 2 else [1.0])

        self.final_rewards *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards

        self.saved_actions.insert(self.rollout_step, screen, minimap, game, action.data, x1.data, y1.data, value_pred.data, reward, masks)

        if self.rollout_step == self.num_steps:
            self.rollout()

        return actions.FunctionCall(function_id, act_args)

    def reset(self):
        super(A2CAgent, self).reset()
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

        advantages = self.saved_actions.compute_returns(self.rollout_step, self.gamma, self.tau)

        screens = Variable(st.screens[:self.rollout_step].squeeze(1)).cuda()
        minimaps = Variable(st.minimaps[:self.rollout_step].squeeze(1)).cuda()
        games = Variable(st.games[:self.rollout_step].squeeze(1)).cuda()
        actions = Variable(st.actions[:self.rollout_step].squeeze(1)).cuda()
        x1s = Variable(st.x1s[:self.rollout_step].squeeze(1)).cuda()
        y1s = Variable(st.y1s[:self.rollout_step].squeeze(1)).cuda()

        values, logpac, entropy = self.model.evaluate_actions(screens, minimaps, games, actions, x1s, y1s)

        pg_loss = -(Variable(advantages).cuda() * logpac).mean()
        pg_loss = pg_loss - self.entropy_coef * entropy
        vf_loss = advantages.pow(2).mean()

        train_loss = pg_loss + self.value_coef * vf_loss

        self.optimizer.zero_grad()
        train_loss.backward()

        nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad)

        self.optimizer.step()

        self.model.summary.add_scalar_value("reward", int(self.reward))
        self.model.summary.add_scalar_value("value_loss", vf_loss)
        self.model.summary.add_scalar_value("loss", train_loss.data[0])

        # if we are still running an episode, copy over the state to position zero
        last_full_action = self.rollout_step
        self.saved_actions.screens[0].copy_(self.saved_actions.screens[last_full_action])
        self.saved_actions.minimaps[0].copy_(self.saved_actions.minimaps[last_full_action])
        self.saved_actions.games[0].copy_(self.saved_actions.games[last_full_action])
        self.rollout_step = 0
