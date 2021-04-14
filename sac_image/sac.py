import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np

from models import ValueNetwork, SoftQNetwork, PolicyNetwork, FeatureExtractor
from replay_buffers import BasicBuffer

class SACAgent:
    def __init__(self, env, gamma, tau, v_lr, q_lr, policy_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]
        #self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] #1

        self.conv_channels = 4
        self.kernel_size = (3, 3)

        self.img_size = (500, 500, 3)

        print("Diagnostics:")
        print(f"action_range: {self.action_range}")
        #print(f"obs_dim: {self.obs_dim}")
        print(f"action_dim: {self.action_dim}")

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        
        # initialize networks 
        self.feature_net = FeatureExtractor(self.img_size[2], self.conv_channels, self.kernel_size).to(self.device)
        print("Feature net init'd successfully")

        input_dim = self.feature_net.get_output_size(self.img_size)
        self.input_size = input_dim[0] * input_dim[1] * input_dim[2]
        print(f"input_size: {self.input_size}")

        self.value_net = ValueNetwork(self.input_size, 1).to(self.device)
        self.target_value_net = ValueNetwork(self.input_size, 1).to(self.device)
        self.q_net1 = SoftQNetwork(self.input_size, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.input_size, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.input_size, self.action_dim).to(self.device)

        print("Finished initing all nets")

        # copy params to target param
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param)

        print("Finished copying targets")

        # initialize optimizers 
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        print("Finished initing optimizers")

        self.replay_buffer = BasicBuffer(buffer_maxlen)
        print("End of init")

    def get_action(self, state):
        if state.shape != self.img_size:
            print(f"Invalid size, expected shape {self.img_size}, got {state.shape}")
            return None

        inp = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        features = self.feature_net(inp)
        features = features.view(-1, self.input_size)
        
        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        mean, log_std = self.policy_net.forward(features)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0
   
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # states and next states are lists of ndarrays, np.stack converts them to
        # ndarrays of shape (batch_size, height, width, num_channels)
        states = np.stack(states)
        next_states = np.stack(next_states)

        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        # Process images
        features = self.feature_net(states) # Properly shaped due to batching
        next_features = self.feature_net(next_states)

        features = torch.reshape(features, (64, self.input_size))
        next_features = torch.reshape(next_features, (64, self.input_size))
        
        next_actions, next_log_pi = self.policy_net.sample(next_features)
        next_q1 = self.q_net1(next_features, next_actions)
        next_q2 = self.q_net2(next_features, next_actions)
        next_v = self.target_value_net(next_features)

        next_v_target = torch.min(next_q1, next_q2) - next_log_pi
        curr_v = self.value_net.forward(features)
        v_loss = F.mse_loss(curr_v, next_v_target.detach())

        # q loss
        expected_q = rewards + (1 - dones) * self.gamma * next_v
        curr_q1 = self.q_net1.forward(features, actions)
        curr_q2 = self.q_net2.forward(features, actions)    
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update value and q networks        
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # delayed update for policy network and target q networks
        if self.update_step % self.delay_step == 0:
            new_actions, log_pi = self.policy_net.sample(features)
            min_q = torch.min(
                self.q_net1.forward(features, new_actions),
                self.q_net2.forward(features, new_actions)
            )
            policy_loss = (log_pi - min_q).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # target networks
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.update_step += 1


        