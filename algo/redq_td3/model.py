import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))

    def forward(self, x):
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias
        return x


class Actor(nn.Module):
    def __init__(self, state_dim,action_dim, hidden_width, action_space=None):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_width)
        self.linear2 = nn.Linear(hidden_width, hidden_width)
        self.output = nn.Linear(hidden_width, action_dim)
        self.max_action = float(action_space.high[0])

    def forward(self, state):
        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))
        return self.max_action * torch.tanh(self.output(a))



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_width, num_nets=10):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            EnsembleLinear(state_dim + action_dim, hidden_width, ensemble_size=num_nets),
            nn.ReLU(),
            EnsembleLinear(hidden_width, hidden_width, ensemble_size=num_nets),
            nn.ReLU(),
            EnsembleLinear(hidden_width, 1, ensemble_size=num_nets)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.critic(x)  # return dim: (num_nets, batch, 1)
