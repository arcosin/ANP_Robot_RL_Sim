import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Note that weights and biases in pytorch are initialized to k,b = 1/input_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(SoftQNetwork, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Note that weights and biases in pytorch are initialized to k,b = 1/input_dim

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Note that weights and biases in pytorch are initialized to k,b = 1/input_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Distributions are parameterized by their mean (a location param, loc), and standard deviation (a scale param, scale)
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi