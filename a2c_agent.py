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

exp_num = 16

Coordinate = namedtuple("Coordinate", ["x", "y"])

class A2CAgent(LearningAgent):
    """The start of a basic A2C agent for learning agents."""
    def __init__(self, screen_width=84, screen_height=84, num_steps=40, num_processes=1, fully_conv=False):
        super(A2CAgent, self)
        num_functions = len(actions.FUNCTIONS)
        if fully_conv:
            self.model = FullyConvModel(num_functions=num_functions).cuda()
        else:
            self.model =  A2CModel(num_functions=num_functions).cuda()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.max_grad = 0.5
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.episode_rewards = torch.zeros([num_processes, 1])
        self.final_rewards = torch.zeros([num_processes, 1])
        self.gamma = 0.99
        self.saved_actions = SavedActions(self.num_steps,
                                          self.num_processes,
                                          num_functions)
        self.saved_actions.cuda()
        self.rollout_step = 0
        #self.model.load_state_dict(torch.load(f"./learning_agent_{exp_num}.pth"))
        self.model.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                        lr=5e-4,
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

        value_pred, action, spatial = self.model.act(Variable(screen).cuda(),
                                                    Variable(minimap).cuda(),
                                                    Variable(game).cuda(),
                                                    Variable(allowed_actions).cuda())
        function_id = action.data[0]

        xy_index = spatial.data[0,0]
        spatial_pts = Coordinate(x=xy_index % self.screen_width, y=xy_index // self.screen_height)

        act_args = []
        for args in actions.FUNCTIONS[function_id].args:
            if args.name in ['screen', 'minimap', 'screen2']:
                act_args.append([spatial_pts.x, spatial_pts.y])
            else:
                act_args.append([0])

        masks = torch.FloatTensor([0.0] if obs.step_type == 2 else [1.0])

        self.final_rewards *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards

        self.saved_actions.insert(self.rollout_step, screen, minimap, game, action.data, value_pred.data, reward, masks)

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

        torch.save(self.model.state_dict(), f"./learning_agent_{exp_num}.pth")
        self.saved_actions.reset()

    def rollout(self):
        st = self.saved_actions
        next_value = self.model(Variable(st.screens[-1]).cuda(),
                                Variable(st.minimaps[-1]).cuda(),
                                Variable(st.games[-1]).cuda())[2].data

        self.saved_actions.compute_returns(next_value, self.gamma, self.tau)

        screens = Variable(st.screens[:-1].squeeze(1)).cuda()
        minimaps = Variable(st.minimaps[:-1].squeeze(1)).cuda()
        games = Variable(st.games[:-1].squeeze(1)).cuda()
        actions = Variable(st.actions[:-1].view(-1, 1))

        values, a_lp, a_ent, sp_lp, sp_ent = self.model.evaluate_actions(screens, minimaps, games, actions)

        values = values.view(self.num_steps, self.num_processes, 1)
        action_log_probs = a_lp.view(self.num_steps, self.num_processes, 1)
        spatial_log_probs = sp_lp.view(self.num_steps, self.num_processes, 1)

        advantages = Variable(self.saved_actions.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        spatial_loss = -(Variable(advantages.data) * spatial_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef
            + (action_loss - a_ent * self.entropy_coef)
            + (spatial_loss - sp_ent * self.entropy_coef)
        ).backward()

        nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad)

        self.optimizer.step()

        self.model.summary.add_scalar_value("reward", int(self.reward))
        self.model.summary.add_scalar_value("value_loss", value_loss.data[0])
        self.model.summary.add_scalar_value("action_loss", action_loss.data[0])
        self.model.summary.add_scalar_value("spatial_loss", spatial_loss.data[0])

        # if we are still running an episode, copy over the state to position zero
        last_full_action = self.rollout_step
        self.saved_actions.screens[0].copy_(self.saved_actions.screens[last_full_action])
        self.saved_actions.minimaps[0].copy_(self.saved_actions.minimaps[last_full_action])
        self.saved_actions.games[0].copy_(self.saved_actions.games[last_full_action])
        self.rollout_step = 0
