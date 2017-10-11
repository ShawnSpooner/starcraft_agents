import numpy as np
from collections import namedtuple
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from pycrayon import CrayonClient

class LearningAgent(base_agent.BaseAgent):
    def __init__(self):
        super(LearningAgent, self)

    def preprocess(self, obs):
        screen = obs.observation["screen"]
        minimap = obs.observation["minimap"]
        game = obs.observation["player"]
        allowed_actions = obs.observation["available_actions"]

        screen = self.process(screen).unsqueeze(0)
        minimap = self.process(minimap, features.MINIMAP_FEATURES).unsqueeze(0)
        #kscreen = torch.from_numpy(screen).float().unsqueeze(0)
        #minimap = torch.from_numpy(minimap).float().unsqueeze(0)
        game = torch.log(torch.from_numpy(game).float().unsqueeze(0))

        return screen, minimap, game, torch.from_numpy(allowed_actions).long()

    def process(self, feature_layers, feature_space=features.SCREEN_FEATURES):
        layer_count, screen_width, screen_height = feature_layers.shape
        layers = torch.zeros(layer_count, screen_width, screen_height)

        for i in range(layer_count):
            # if the feature layer type is scalar, log scale it
            if feature_space[i].type == features.FeatureType.SCALAR:
                l = torch.log(torch.from_numpy(feature_layers[i] + np.finfo(float).eps))
                l = l.unsqueeze(0).unsqueeze(0)
                layers[i].copy_(l)
            # other wise follow the paper and make it continous
            else:
                #one hot encode the channel dimension
                fl = torch.from_numpy(feature_layers[i])
                channels = feature_space[i].scale
                fl_ = torch.unsqueeze(fl, 0).long()

                one_hot = torch.FloatTensor(channels, screen_width, screen_height).zero_()
                one_hot.scatter_(0, fl_, 1)

                m = nn.Conv2d(channels, 1, 1, stride=1).cuda()
                l = m(Variable(one_hot.unsqueeze(0)).cuda())
                layers[i].copy_(l.data)

        return layers
