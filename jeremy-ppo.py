import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
import gym


# device = torch.device('cuda')
device = torch.device('cpu')


class DiscretePolicyNet(nn.Module):
    def __init__(self, in_dim, num_actions, num_layers, hidden_size, value_layers=2, value_layer_size=24):
        super(DiscretePolicyNet, self).__init__()
        self.net = self.build_policy_net(in_dim, num_actions, num_layers, hidden_size)
        self.value_net = self.build_value_net(in_dim, value_layers, value_layer_size)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.num_actions = num_actions
        self.optim = None

    def build_policy_net(self, in_dim, num_actions, num_layers, hidden_size):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(0, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def build_value_net(self, in_dim, value_layers, value_layer_size):
        layers = []
        layers.append(nn.Linear(in_dim, value_layer_size))
        layers.append(nn.ReLU())
        for i in range(0, value_layers):
            layers.append(nn.Linear(value_layer_size, value_layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(value_layer_size, 1))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        # make into vector
        state = torch.from_numpy(state).float().unsqueeze(0)

        # make Variable to include info for Autograd
        distro = self.forward(Variable(state))

        selected_action = np.random.choice(self.num_actions, p=np.squeeze(distro.detach().numpy()))
        action_prob = distro.squeeze(0)[selected_action]
        return selected_action, action_prob

    def evaluate_action(self, state, action):
        state = torch.from_numpy(state).float().unsqueeze(0)
        distro = self.forward(Variable(state))
        action_prob = distro.squeeze(0)[action]
        expected_value = self.value_net(state)
        return action_prob, expected_value

    def build_optim(self, lr=3e-4):
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


def explore(env, model, num_steps=500):
    states = []
    actions = []
    rewards = []
    action_probs = []
    env_state = env.reset()
    done = False
    while not done:
        states.append(env_state)
        action, action_prob = model.get_action(env_state)
        env_state, reward, done, info = env.step(action)
        rewards.append(reward)
        action_probs.append(action_prob)
        actions.append(action)
    return states, actions, rewards, action_probs


def replay(model, transitions, rewards, action_probs, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.001):
    discounted_rewards = []

    for i in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[i:]:
            Gt = Gt + gamma**pw * r
            pw += 1
        discounted_rewards.append(Gt)

    expected_values = []
    new_probs = []
    for (s, a) in transitions:
        probability, value = model.evaluate_action(s, a)
        expected_values.append(value)
        new_probs.append(probability)

    new_probs = torch.stack(new_probs)
    expected_values = torch.stack(expected_values)

    discounted_rewards = torch.Tensor(discounted_rewards).to(device)
    advantages = discounted_rewards.clone().detach() - expected_values

    # normalize rewards to discourage the lowest-scoring actions
    normed_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    ratios = new_probs / torch.stack(action_probs).detach()

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-epsilon, 1+epsilon) * advantages

    # entropy term to encourage exploration
    entropy = torch.distributions.Categorical(torch.log(new_probs)).entropy()

    loss = -torch.min(surr1, surr2) + c1*F.mse_loss(expected_values.squeeze(), discounted_rewards.squeeze()) - c2*entropy

    model.train().to(device)
    if model.optim is None:
        model.build_optim()
    action_probs = torch.stack(action_probs).to(device)

    model.optim.zero_grad()
    # loss = -torch.sum(torch.log(action_probs) * discounted_rewards)
    loss.sum().backward()
    model.optim.step()

    model.eval().cpu()


def train_model(env, model, num_epochs, minibatch_size=50):
    for i in range(num_epochs):
        old_policy = copy.deepcopy(model)
        old_policy.eval()
        for j in range(minibatch_size):
            states, actions, rewards, action_probs = explore(env, old_policy)
            replay(model, zip(states, actions), rewards, action_probs)


def get_model_fitness(env, model):
    model.eval()
    env_state = env.reset()
    reward = 0
    done = False
    while not done:
        a, _ = model.get_action(env_state)
        env_state, step_reward, done, _ = env.step(a)
        reward += step_reward
    return reward


def main():
    env = gym.make("CartPole-v1")
    num_inputs = 1
    for i in env.observation_space.shape:
        num_inputs *= i
    model = DiscretePolicyNet(num_inputs, env.action_space.n, 2, 24)
    train_model(env, model, 20)
    print(get_model_fitness(env, model))


if __name__ == '__main__':
    main()
