import numpy as np
from pysc2.lib import actions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from pycrayon import CrayonClient
from collections import namedtuple


class A2CModel(nn.Module):
    def __init__(self, num_functions, screen_width=84, screen_height=84):
        super(A2CModel, self).__init__()
        self.num_functions = num_functions
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.cc = CrayonClient(hostname="localhost")
        expirement_name = type(self).__name__

        # if we have an existing expirement with this name, clean it up first
        if expirement_name in self.cc.get_experiment_names():
            self.summary = self.cc.remove_experiment(expirement_name)

        self.summary = self.cc.create_experiment(expirement_name)

        # our model specification
        self.conv1 = nn.Conv2d(13, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.mm_conv1 = nn.Conv2d(7, 16, kernel_size=8, stride=4)
        self.mm_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.feature_input = nn.Linear(11, 128)

        self.fc = nn.Linear(5312, 256)
        self.action_head = nn.Linear(256, self.num_functions)
        self.value_head = nn.Linear(256, 1)

        self.spatial_x = nn.Linear(256, 84)
        self.spatial_y = nn.Linear(256, 84)

        self.saved_actions = []
        self.rewards = []

    def forward(self, screen, mm, game):
        # screen network
        screen = F.elu(self.conv1(screen))
        screen = F.elu(self.conv2(screen))
        #x = F.relu(self.conv3(x))

        # minimap network
        mm = F.elu(self.mm_conv1(mm))
        mm = F.elu(self.mm_conv2(mm))
        #mm = F.relu(self.mm_conv3(mm))

        #non spatial structured data
        xs = F.tanh(self.feature_input(game))

        # non-spatial policy
        st = torch.cat([screen.view(-1, 2592), mm.view(-1, 2592), xs], dim=1)
        ns = F.relu(self.fc(st))

        # critic prediction
        state_values = self.value_head(ns)

        # action predictions
        action_scores = self.action_head(ns)

        # need to predict the spatial action policy here
        spatial_x = F.softmax(self.spatial_y(ns))
        expanded_x = spatial_x.repeat(self.screen_width, 1)
        spatial_y = F.softmax(self.spatial_x(ns))
        expanded_y = spatial_y.repeat(self.screen_height, 1).t()

        spatial = expanded_x * expanded_y
        return (action_scores,
               spatial.view(-1, self.screen_height*self.screen_width),
               state_values)

    def act(self, state, minimap, game_state, available_actions):
        action_logits, spatial_probs, state_value = self(state,
                                                         minimap,
                                                         game_state)

        action = F.softmax(action_logits.squeeze(0).gather(0, available_actions)).multinomial()
        spatial = F.softmax(spatial_probs).multinomial()

        return state_value, available_actions[action], spatial

    def evaluate_actions(self, screens, minimaps, games, actions):
        logits, spatial_logits, values = self(screens, minimaps, games)

        log_probs = F.log_softmax(logits)
        probs = F.softmax(logits)
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        spatial_log_probs = F.log_softmax(spatial_logits)
        spatial_probs = F.softmax(spatial_logits)
        spatial_act_log_probs = spatial_log_probs.gather(1, actions)
        spatial_dist_entropy = -(spatial_log_probs * spatial_probs).sum(-1).mean()

        return (values, action_log_probs, dist_entropy,
                       spatial_act_log_probs, spatial_dist_entropy)
