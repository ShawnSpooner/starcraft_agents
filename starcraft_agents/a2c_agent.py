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
from torch.utils.data import DataLoader, Dataset

from starcraft_agents.a2c_model import A2CModel
from starcraft_agents.learning_agent import LearningAgent
from starcraft_agents.saved_actions import TrajectoryDataset
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import torchnet as tnt


class A2CAgent(LearningAgent):
    """The start of a basic A2C agent for learning agents."""
    def __init__(self, screen_width, screen_height, horizon,
                 num_processes=2,
                 fully_conv=False,
                 expirement_name="default_expirement",
                 learning_rate=7e-4,
                 value_coef=1.0,
                 entropy_coef=1e-4,
                 in_channels=8,
                 continue_training=False,
                 summary=None):
        super(A2CAgent, self).__init__(expirement_name)
        num_functions = len(actions.FUNCTIONS)
        self.model = A2CModel(num_functions=num_functions,
                              expirement_name=expirement_name,
                              screen_width=screen_width,
                              screen_height=screen_height).cuda()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.summary = summary
        self.in_channels = in_channels
        self.horizon = horizon
        self.num_processes = num_processes
        self.max_grad = 0.5
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = 0.95
        self.tau = 0.97
        self.saved_actions = TrajectoryDataset(self.horizon,
                                               self.num_processes,
                                               screen_width,
                                               screen_height)
        if continue_training:
            self.model.load_state_dict(torch.load(f"./models/{expirement_name}.pth"))
            self.model.eval()

        print(f"learning rate set to: {learning_rate}")
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate)

        self.final_rewards = torch.zeros(1, 1)
        self.setup_loggers()

    def setup_loggers(self):
        # visdom setup
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.loss_logger = VisdomPlotLogger('line',
                                            env=self.expirement_name,
                                            opts={'title': 'Train Loss'})

        self.pi_loss_meter = tnt.meter.AverageValueMeter()
        self.pi_loss_logger = VisdomPlotLogger('line',
                                               env=self.expirement_name,
                                               opts={'title': 'Policy Loss'})

        self.xy_loss_meter = tnt.meter.AverageValueMeter()
        self.xy_loss_logger = VisdomPlotLogger('line',
                                               env=self.expirement_name,
                                               opts={'title': 'XY Loss'})

        self.value_loss_meter = tnt.meter.AverageValueMeter()
        self.value_loss_logger = VisdomPlotLogger('line',
                                                  env=self.expirement_name,
                                                  opts={'title': 'Value Loss'})

        self.reward_meter = tnt.meter.AverageValueMeter()
        self.reward_logger = VisdomPlotLogger('line',
                                              env=self.expirement_name,
                                              opts={'title': 'Batch Reward'})

        self.entropy_meter = tnt.meter.AverageValueMeter()
        self.entropy_logger = VisdomPlotLogger('line',
                                               env=self.expirement_name,
                                               opts={'title': 'Entropy'})

        self.adv_meter = tnt.meter.AverageValueMeter()
        self.adv_logger = VisdomPlotLogger('line',
                                           env=self.expirement_name,
                                           opts={'title': 'Advantage'})

        self.episode_logger = VisdomPlotLogger('line',
                                               env=self.expirement_name,
                                               opts={'title': "Episode Score"})
        self.episode_meter = tnt.meter.MovingAverageValueMeter(windowsize=3)

    def finish_step(self):
        self.saved_actions.step()

    def reset_meters(self):
        self.adv_meter.reset()
        self.loss_meter.reset()
        self.pi_loss_meter.reset()
        self.value_loss_meter.reset()
        self.entropy_meter.reset()
        self.xy_loss_meter.reset()

    def rollout(self):
        self.reset_meters()
        self.saved_actions.compute_advantages(self.gamma)

        loader = DataLoader(self.saved_actions, batch_size=self.horizon, shuffle=True)

        for screens, minimaps, games, actions, x1s, y1s, rewards, returns in loader:
            values, lp, x_lp, y_lp, dist_entropy, spatial_entropy = self.model.evaluate_actions(
                                                    Variable(screens).cuda(),
                                                    Variable(minimaps).cuda(),
                                                    Variable(games).cuda(),
                                                    Variable(actions).cuda(),
                                                    Variable(x1s).cuda(),
                                                    Variable(y1s).cuda())

            rewards_var = Variable(rewards).cuda()
            returns_var = Variable(returns).cuda()
            advs = (returns_var - values).data
            advs_var = Variable(advs).cuda()

            dist_entropy *= self.entropy_coef
            spatial_entropy *= self.entropy_coef
            pg_loss = ((lp + x_lp + y_lp) * advs_var).mean()
            pg_loss -= dist_entropy
            pg_loss -= spatial_entropy
            vf_loss = (values - rewards_var).pow(2).mean() * self.value_coef

            train_loss = pg_loss + vf_loss

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad)
            train_loss.backward()

            self.optimizer.step()
            self.loss_meter.add(train_loss.data[0])
            self.pi_loss_meter.add(pg_loss.data[0])
            self.entropy_meter.add(dist_entropy.data[0] + spatial_entropy.data[0])
            self.value_loss_meter.add(vf_loss.data[0])
            self.reward_meter.add(rewards.sum())
            self.adv_meter.add(advs.mean())

        self.loss_logger.log(self.steps, self.loss_meter.value()[0])
        self.pi_loss_logger.log(self.steps, self.pi_loss_meter.value()[0])
        self.reward_logger.log(self.steps, self.reward_meter.value()[0])
        self.entropy_logger.log(self.steps, self.entropy_meter.value()[0])
        self.value_loss_logger.log(self.steps, self.value_loss_meter.value()[0])
        self.adv_logger.log(self.steps, self.adv_meter.value()[0])
        self.episode_logger.log(self.steps, self.episode_meter.value()[0])
