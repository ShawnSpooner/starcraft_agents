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
        return {layer.name: nn.Embedding(layer.scale, embedding_dim) for layer in feature_space}

    def __init__(self, expirement_name):
        super(LearningAgent, self).__init__()
        self.expirement_name = expirement_name
        self.rollout_step = 0
        self.episode_rewards = torch.zeros(1, 1)
        self.episodes = 0
        self.steps = 0
        self.reward = 0

        self.embeddings = {"screen": self.build_embeddings(features.SCREEN_FEATURES),
                           "minimap": self.build_embeddings(features.MINIMAP_FEATURES)}

    def preprocess(self, obs):
        screen = obs.observation["screen"]
        minimap = obs.observation["minimap"]
        game = obs.observation["player"]
        #allowed_actions = [o for o in obs.observation["available_actions"] if o in [0,7,333, 334, 12, 13]]
        allowed_actions = obs.observation["available_actions"]

        allowed_vec = np.zeros(524)
        allowed_vec[allowed_actions] = 1

        screen = self.embed(screen, space="screen").unsqueeze(0)
        minimap = self.embed(minimap, space="minimap").unsqueeze(0)
        game = torch.log(torch.from_numpy(game).float().unsqueeze(0))

        return screen, minimap, game, torch.from_numpy(allowed_vec).float()

    def embed(self, feature_layers, embedding_dim=8, space="screen"):
        layer_count, screen_height, screen_width = feature_layers.shape

        feature_space = features.SCREEN_FEATURES if space == "screen" else features.MINIMAP_FEATURES
        # layers we are using: player_id, player_relative, selected, unit_type
        allowed_layers = ["player_relative"]
        layers = torch.zeros(len(allowed_layers), screen_width, screen_height)
        current_layer = 0

        for l in feature_space:
            if l.name in allowed_layers:
                #transpose yx into xy
                v = torch.t(torch.from_numpy(feature_layers[l.index]).long())
                layers[current_layer] = v
        return layers.long()
        # the concat route
        #return torch.cat(layers, dim=0)
        # the sum over channels route
        return torch.sum(layers, dim=0)

    def step(self, step, pid, obs):
        super(LearningAgent, self).step(obs)

        # if this was a last step, log the score
        if obs.last():
            print("ending score was: ", obs.observation["score_cumulative"][0])
            self.episode_meter.add(int(obs.observation["score_cumulative"][0]))

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
                act_args.append([x1.data[0], y1.data[0]])
            else:
                act_args.append([0])

        dones = torch.FloatTensor([1.0] if obs.first() else [0.0])
        self.saved_actions.insert(
                                  pid,
                                  screen,
                                  minimap,
                                  game,
                                  action.data,
                                  x1.data,
                                  y1.data,
                                  value_pred.data,
                                  reward,
                                  dones)
        #print(function_id, act_args)
        return actions.FunctionCall(function_id, act_args)

    def reset(self):
        super(LearningAgent, self).reset()

        torch.save(self.model.state_dict(), f"./models/{self.expirement_name}.pth")
