import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from math import floor

# Calculations are based off of https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
# under the Shape section
def calc_convout_shape(image_size, kernel_size, stride = 1, pad = 0, dilation = 1):
    height = floor(((image_size[0] + (2 * pad) - (dilation * (kernel_size[0] - 1) ) - 1 ) / stride) + 1)
    width = floor(((image_size[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1 ) / stride) + 1)
    return height, width

# Feature extractor layers (conv) also go here
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, load_weights=None):
        super(FeatureExtractor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size)

        if load_weights != None:
            self.load_state_dict(load_weights) # Must be a dictionary
            # Use nn.Module.state_dict() to acquire dictionary of current weights and biases

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        return x

    def get_output_size(self, input_size):
        if type(input_size) is not tuple:
            input_size = (input_size, input_size, self.in_channels)

        # First conv layer
        h1, w1 = calc_convout_shape(input_size, self.kernel_size)
        h1 = floor(h1 / self.kernel_size[0]) # Accounts for maxpool2d layers
        w1 = floor(w1 / self.kernel_size[1])
        
        # Second conv layer
        h2, w2 = calc_convout_shape((h1, w1), self.kernel_size)
        h2 = floor(h2 / self.kernel_size[0]) 
        w2 = floor(w2 / self.kernel_size[1])

        return (self.out_channels, h2, w2)

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