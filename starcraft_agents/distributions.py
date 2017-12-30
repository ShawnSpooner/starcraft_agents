import torch
import torch.nn as nn


class Multinoulli(object):
    def __init__(self):
        super(Multinoulli, self).__init__()
        self.neglogp = nn.CrossEntropyLoss()

    def entropy(self, logits):
        a0 = logits - torch.max(logits)
        ea0 = torch.exp(a0)
        z0 = ea0.sum(-1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (torch.log(z0) - a0)).sum(-1)

    def negative_log_probability(self, logits, actions):
        return self.neglogp(logits, actions)
