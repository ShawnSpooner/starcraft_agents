import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from pycrayon import CrayonClient
from collections import namedtuple


class FullyConvModel(nn.Module):
    def __init__(self, num_functions, expirement_name, screen_width=64, screen_height=64):
        super(FullyConvModel, self).__init__()
        self.num_functions = num_functions
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.cc = CrayonClient(hostname="localhost")

        # if we have an existing expirement with this name, clean it up first
        if expirement_name in self.cc.get_experiment_names():
            self.summary = self.cc.remove_experiment(expirement_name)

        self.summary = self.cc.create_experiment(expirement_name)

        self.conv1 = nn.Conv2d(13, 16, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)

        self.mm_conv1 = nn.Conv2d(7, 16, kernel_size=5, stride=1, padding=1)
        self.mm_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)

        self.feature_input = nn.Linear(11, 128)

        self.fc = nn.Linear(64 * self.screen_width * self.screen_height, 256)

        self.action_head = nn.Linear(256, self.num_functions)
        self.value_head = nn.Linear(256, 1)

        self.spatial_policy = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data)

    def forward(self, screen, mm, game):
        # screen network
        screen = F.elu(self.conv1(screen))
        screen = F.elu(self.conv2(screen))

        # minimap network
        mm = F.elu(self.mm_conv1(mm))
        mm = F.elu(self.mm_conv2(mm))

        st = torch.cat([screen, mm], dim=1)

        # non-spatial policy
        ns = F.relu(self.fc(st.view(-1, 64 * self.screen_width * self.screen_height)))

        # critic prediction
        state_values = self.value_head(ns)

        # action predictions
        action_scores = self.action_head(ns)

        # need to predict the spatial action policy here
        spatial = self.spatial_policy(st)
        flat_yx = spatial.view(-1, self.screen_height * self.screen_width)

        return (action_scores,
                flat_yx // screen_width, flat_yx % screen_height,
                state_values)

    def act(self, state, minimap, game_state, available_actions):
        action_logits, spatial_probs, state_value = self(state,
                                                         minimap,
                                                         game_state)

        action = F.softmax(action_logits.squeeze(0).gather(0, available_actions)).multinomial()
        spatial = F.softmax(spatial_probs).multinomial()
        return state_value, available_actions[action], spatial

    def evaluate_actions(self, screens, minimaps, games, actions, x1s, y1s):
        def recover_yx_index(xs, ys):
            return [y * self.screen_height + x for x,y in zip(xs, ys)]

        logits, x1_logits, y1_logits, values = self(screens, minimaps, games)

        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = (log_probs * probs).sum(1).mean()

        spatial_log_probs = F.log_softmax(x1_logits, dim=1)
        spatil_probs = F.softmax(x1_logits, dim=1)
        spatial_act_lp = (x1_log_probs * x1_probs).gather(1, x1s)
        spatial_entropy = (x1_log_probs * x1_probs).sum(1).mean()

        logpac = action_log_probs + spatial_act_lp
        entropy = dist_entropy + spatial_entropy

        return values, logpac, entropy
