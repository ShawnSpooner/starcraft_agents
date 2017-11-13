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

    def build_embeddings(self, feature_space, embedding_dim=8):
        """
        Build up a list of the embeddings for each layer in the supplied
        feature collections

        @param feature_space a feature.Feature constant from pysc2
               either SCREEN_FEATURES or MINIMAP_FEATURES

        @param embedding_dim how many dimensions to encode the categorical
               features into
        """
        return [nn.Embedding(layer.scale, embedding_dim) for layer in feature_space]

    def __init__(self, expirement_name):
        super(LearningAgent, self).__init__()
        self.expirement_name = expirement_name
        self.rollout_step = 0
        self.episode_rewards = torch.zeros(1, 1)

        self.embeddings = {"screen": self.build_embeddings(features.SCREEN_FEATURES),
                           "minimap": self.build_embeddings(features.MINIMAP_FEATURES)}

    def preprocess(self, obs):
        screen = obs.observation["screen"]
        minimap = obs.observation["minimap"]
        game = obs.observation["player"]
        allowed_actions = obs.observation["available_actions"]

        screen = self.embed(screen, space="screen").unsqueeze(0)
        minimap = self.embed(minimap, space="minimap").unsqueeze(0)
        game = torch.log(torch.from_numpy(game).float().unsqueeze(0))

        return screen, minimap, game, torch.from_numpy(allowed_actions).long()

    def process(self, feature_layers, feature_space=features.SCREEN_FEATURES):
        layer_count, screen_width, screen_height = feature_layers.shape
        layers = torch.zeros(layer_count, screen_width, screen_height)

        for i in range(layer_count):
            # if the feature layer type is scalar, log scale it
            if feature_space[i].type == features.FeatureType.SCALAR:
                l = torch.log(torch.from_numpy(feature_layers[i] + np.finfo(float).eps))
                layers[i].copy_(l)
            # otherwise follow the paper and make it continous
            else:
                l = torch.from_numpy(feature_layers[i] / feature_space[i].scale)
                layers[i].copy_(l)

        return layers

    def embed(self, feature_layers, embedding_dim=8, space="screen"):
        layer_count, screen_width, screen_height = feature_layers.shape
        layers = torch.zeros(1, embedding_dim, screen_width, screen_height)
        feature_space = features.SCREEN_FEATURES if space == "screen" else features.MINIMAP_FEATURES
        # layers we are using: player_id, player_relative, selected, unit_type
        for i in range(layer_count):
            # quick filter to only include the core layers needed to bootstrap an agent
            if feature_space[i].name == "player_relative":
                embedding = self.embeddings[space][i]
                embedded = embedding(Variable(torch.from_numpy(feature_layers[i])).long())
                layers[0] = embedded.view(1, 8, self.screen_height, self.screen_width).data
        return layers

    def step(self, obs):
        super(LearningAgent, self).step(obs)
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
                act_args.append([x1.data[0, 0], y1.data[0, 0]])
            else:
                act_args.append([0])

        mask = torch.FloatTensor([0.0] if obs.step_type == 2 else [1.0])
        self.saved_actions.insert(self.rollout_step,
                                  screen,
                                  minimap,
                                  game,
                                  action.data,
                                  x1.data,
                                  y1.data,
                                  value_pred.data,
                                  reward,
                                  mask)

        if self.rollout_step == self.horizon:
            self.rollout()

        return actions.FunctionCall(function_id, act_args)

    def reset(self):
        super(LearningAgent, self).reset()
        self.model.summary.add_scalar_value("episode_reward",
                                            int(self.episode_rewards[0, 0]))
        self.episode_rewards = torch.zeros(1, 1)

        if self.steps > 1:
            self.rollout()

        torch.save(self.model.state_dict(), f"./{self.expirement_name}.pth")
        self.saved_actions.reset()
