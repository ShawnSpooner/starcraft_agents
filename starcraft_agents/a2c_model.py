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
    def __init__(self, num_functions, expirement_name, screen_width, screen_height):
        super(A2CModel, self).__init__()
        self.neglogp = nn.CrossEntropyLoss()
        self.num_functions = num_functions
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.cc = CrayonClient(hostname="localhost")

        # if we have an existing expirement with this name, clean it up first
        if expirement_name in self.cc.get_experiment_names():
            self.summary = self.cc.remove_experiment(expirement_name)

        self.summary = self.cc.create_experiment(expirement_name)

        # our model specification
        self.conv1 = nn.Conv2d(16, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.mm_conv1 = nn.Conv2d(16, 16, kernel_size=8, stride=4)
        self.mm_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.feature_input = nn.Linear(11, 128)

        self.fc = nn.Linear(384, 512)
        self.action_head = nn.Linear(512, self.num_functions)
        self.value_head = nn.Linear(512, 1)

        self.spatial_x = nn.Linear(512, screen_width)
        self.spatial_y = nn.Linear(512, screen_height)

        self.saved_actions = []
        self.rewards = []

    def forward(self, screen, mm, game):
        # screen network
        screen = F.elu(self.conv1(screen))
        screen = F.elu(self.conv2(screen.squeeze(2)))
        #x = F.relu(self.conv3(x))

        # minimap network
        mm = F.elu(self.mm_conv1(mm))
        mm = F.elu(self.mm_conv2(mm.squeeze(2)))
        #mm = F.relu(self.mm_conv3(mm))

        #non spatial structured data
        xs = F.tanh(self.feature_input(game))

        # non-spatial policy
        st = torch.cat([screen.view(-1, 128), mm.view(-1, 128), xs], dim=1)
        ns = F.relu(self.fc(st))

        # critic prediction
        state_values = self.value_head(ns)

        # action predictions
        action_scores = self.action_head(ns)

        # need to predict the spatial action policy here
        x1 = self.spatial_x(ns)
        y1 = self.spatial_y(ns)

        return (action_scores, x1, y1, state_values)

    def act(self, state, minimap, game_state, available_actions):
        action_logits, x1_logits, y1_logits, state_value = self(state,
                                                         minimap,
                                                         game_state)

        action = F.softmax(action_logits.squeeze(0).gather(0, available_actions)).multinomial()
        x1 = F.softmax(x1_logits).multinomial()
        y1 = F.softmax(y1_logits).multinomial()

        return state_value, available_actions[action], x1, y1

    def entropy(self, logits):
        a0 = logits - torch.max(logits)
        ea0 = torch.exp(a0)
        z0 = ea0.sum(-1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (torch.log(z0) - a0)).sum(-1)

    def evaluate_actions(self, screens, minimaps, games, actions, x1s, y1s):
        logits, x1_logits, y1_logits, values = self(screens, minimaps, games)

        #log_probs = F.log_softmax(logits)
        action_nlp = self.neglogp(logits, actions.squeeze(1))
        dist_entropy = self.entropy(logits)

        #x1_log_probs = F.log_softmax(x1_logits)
        x1_nlp = self.neglogp(x1_logits, x1s.squeeze(1))
        x1_entropy = self.entropy(x1_logits)

        #y1_log_probs = F.log_softmax(y1_logits)
        y1_nlp = self.neglogp(y1_logits, y1s.squeeze(1))
        y1_entropy = self.entropy(y1_logits)

        #neg_logpac = action_nlp + x1_nlp + y1_nlp
        entropy = dist_entropy + x1_entropy + y1_entropy

        return values, entropy, action_nlp, x1_nlp, y1_nlp
