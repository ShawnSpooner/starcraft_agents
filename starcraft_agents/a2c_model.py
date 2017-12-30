import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Conv2d, ELU


def has_nan(tensor):
    """
    :param tensor to check for NaN values
    :returns bool true if the tensor has any NaN values in it"""
    return tensor[tensor != tensor].size()[0] > 1


class A2CModel(nn.Module):
    def __init__(self, num_functions, expirement_name, screen_width, screen_height):
        super(A2CModel, self).__init__()
        self.embed_dim = 8
        self.embed = nn.Embedding(5, self.embed_dim)
        self.embed_mm = nn.Embedding(5, self.embed_dim)
        self.num_functions = num_functions
        self.screen_width = screen_width
        self.screen_height = screen_height

        # our model specification
        self.conv1 = Conv2d(self.embed_dim, 16, kernel_size=8, stride=4, padding=1)
        #self.conv1 = weight_norm(self.conv1, name="weight")
        self.elu1 = ELU(inplace=True)
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2, padding=2)
        #self.conv2 = weight_norm(self.conv2, name="weight")
        self.elu2 = ELU(inplace=True)

        self.conv_mm1 = Conv2d(self.embed_dim, 16, kernel_size=8, stride=4, padding=1)
        #self.conv_mm1 = weight_norm(self.conv_mm1, name="weight")
        self.elu_mm1 = ELU(inplace=True)
        self.conv_mm2 = Conv2d(16, 32, kernel_size=4, stride=2, padding=2)
        #self.conv_mm2 = weight_norm(self.conv_mm2, name="weight")
        self.elu_mm2 = ELU(inplace=True)

        self.feature_input = nn.Linear(11, 128)

        self.fc = nn.Linear(64 * 64, 128)
        self.fc_relu = ELU(inplace=True)
        self.action_head = nn.Linear(128, self.num_functions)
        self.value_head = nn.Linear(128, 1)

        self.x = nn.Linear(128, self.screen_width)
        self.y = nn.Linear(128, self.screen_height)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)


    def forward(self, screen, minimap, game):
        b, c, h, w = screen.size()
        emb = self.embed(screen.view(-1, screen.size(2)))
        emb = emb.view(b, self.embed_dim, h, w)

        emb_mm = self.embed_mm(minimap.view(-1, minimap.size(2)))
        emb_mm = emb_mm.view(b, self.embed_dim, h, w)

        # screen network
        screen = self.conv1(emb)
        screen = self.elu1(screen)

        screen = self.conv2(screen)
        screen = self.elu2(screen)

        # minimap network
        minimap = self.conv_mm1(emb_mm)
        minimap = self.elu_mm1(minimap)

        minimap = self.conv_mm2(minimap)
        minimap = self.elu_mm2(minimap)

        #non spatial structured data
        #xs = F.tanh(self.feature_input(game))
        st = torch.cat([screen.view(screen.size()[0], -1),
                        minimap.view(minimap.size()[0], -1)], dim=1)

        ns = self.fc(st)
        ns = self.fc_relu(ns)

        # critic prediction
        values = self.value_head(ns)

        # action predictions
        probs = F.softmax(self.action_head(ns), dim=1)

        # need to predict the spatial action policy here
        x = F.softmax(self.x(ns), dim=1)
        y = F.softmax(self.y(ns), dim=1)

        return (probs, x, y, values)

    def act(self, state, minimap, game_state, available_actions):
        action_probs, x_probs, y_probs, state_value = self(state,
                                                         minimap,
                                                         game_state)

        probs = action_probs * available_actions
        if has_nan(probs) or has_nan(x_probs) or has_nan(y_probs):
            raise Exception("got NaN in our softmax outputs")

        probs /= probs.sum()
        act = torch.distributions.Categorical(probs)
        action = act.sample()

        x_dist = torch.distributions.Categorical(x_probs)
        x = x_dist.sample()

        y_dist = torch.distributions.Categorical(y_probs)
        y = y_dist.sample()

        return state_value, action, x, y

    def entropy(self, probs):
        log_p = torch.log(probs) * probs
        return -log_p.sum(-1)

    def evaluate_actions(self, screens, minimaps, games, actions, xs, ys):
        probs, x_probs, y_probs, values = self(screens, minimaps, games)

        a = torch.distributions.Categorical(probs)
        neg_lp = -a.log_prob(actions.squeeze())
        dist_entropy = self.entropy(probs).mean()

        x_dist = torch.distributions.Categorical(x_probs)
        x_neglp = -x_dist.log_prob(xs.squeeze())
        x_entropy = self.entropy(x_probs).mean()

        y_dist = torch.distributions.Categorical(y_probs)
        y_neglp = -y_dist.log_prob(ys.squeeze())
        y_entropy = self.entropy(y_probs).mean()

        spatial_entropy = x_entropy + y_entropy
        return values, neg_lp, x_neglp, y_neglp, dist_entropy, spatial_entropy
