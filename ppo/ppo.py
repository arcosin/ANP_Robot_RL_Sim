import torch as torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import gym

import time

first_time = True
old_pi = None

def run_training_batch(template, env, epochs=500, clip=50):
    model = template(env)
    scores = train_model(env, model, epochs)
    return model, np.mean(scores[(-1 * clip):])

'''
    creates a pool of agents and returns the best model found
'''
def run_pool(env, template, pool_size=25):
    best_model = None
    best_score = 0
    for _ in range(pool_size):
        model, score = run_training_batch(template, env)
        if score > best_score:
            best_model = model
    return best_model

def kek_main():
    env = gym.make('CartPole-v0')
    #model = run_pool(env, build_model)
    init_model = build_model
    model, score = run_training_batch(env, init_model)
    input("training done")
    test_model(env, model, delay=0.05)

def test_model(env, model, steps=100, delay=0.):
    model.eval()
    rewards = []
    env_state = env.reset()
    for i in range(steps):
        pred = model(torch.from_numpy(env_state).float())
        action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
        env_state, reward, done, _ = env.step(action)
        env.render()
        rewards.append(reward)
        if delay > 0:
            time.sleep(delay)
        if done:
            print("reward: ", sum([r for r in rewards]))
            return sum([r for r in rewards])

def explore(env, model, num_steps):
    #print("Running exploration")
    #env_state will be a structure of shape observation_space
    env_state = env.reset()
    done = False
    transitions = []
    is_terminal = []
    score = []
    
    #For CartPole-v0, this is a np-array [0 1]
    action_space = np.array([i for i in range(env.action_space.n)])
    for t in range(num_steps):
        action_weights = model(torch.from_numpy(env_state).float())
        action = np.random.choice(action_space, p=action_weights.data.numpy())
        prev_state = env_state
        #env.render()
        env_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t + 1))
        if done:
            is_terminal.append(True)
            break
        else:
            is_terminal.append(False)
    score.append(len(transitions))  #number of actions survived is the score
    reward_batch = [r for (s, a, r) in transitions]
    
    return transitions, reward_batch, is_terminal

def run_epoch(env, model, optimizer, gamma, mem_size=500):
    #print("Running epoch")
    global first_time
    global old_pi
    #500 actions max per explore/epoch
    transitions, rewards, is_terminal = explore(env, model, mem_size)
    reward_func = []
    discounted_reward = 0

    #Monte-Carlo learning. See https://zsalloum.medium.com/monte-carlo-in-reinforcement-learning-the-easy-way-564c53010511
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminal)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        reward_func.append(discounted_reward)
    
    reward_func.reverse()
    reward_func = torch.FloatTensor(reward_func)
    reward_func = (reward_func - reward_func.mean()) / (reward_func.std() + 1e-5) #Normalize rewards

    states = torch.Tensor([s for (s, a, r) in transitions])
    actions = torch.tensor([a for (s, a, r) in transitions], dtype=torch.int64)
    num_actions = actions.shape[0]

    if first_time:
        old_pi = copy.deepcopy(model)
        first_time = False

    preds = model(states)
    preds_old = old_pi(states)  #potentially just remember instead of re-running
    
    #print(f"preds: {preds}")
    #print(f"old preds: {preds_old}")
    #print(f"actions: {actions}")
    
    pi = preds[torch.arange(num_actions), actions]
    pi_old = preds_old[torch.arange(num_actions), actions]

    #print(f"pi: {pi}\npi_old = {pi_old}")

    r_theta = torch.div(pi, pi_old) 

    #print(f"r_theta: {r_theta}\nr_theta size: {r_theta.shape}")

    advantage = reward_func
    epsilon = 0.2

    t1 = torch.mul(r_theta, advantage)
    t2 = torch.mul(torch.clamp(r_theta, 1 - epsilon, 1 + epsilon), advantage)

    #Loss for PPO with clipping
    loss = -torch.min(t1, t2)

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    old_pi = copy.deepcopy(model)

    return len(transitions)

def train_model(env, model, epochs, learning_rate=0.003, gamma=0.99, mem_size=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    score = []
    for epoch in range(epochs):
        score.append(run_epoch(env, model, optimizer, gamma, mem_size))

        if epoch % 50 == 0 and epoch > 0:
            print('Trajectory {}\tAverage Score: {:.2f}'.format(epoch, np.mean(score[-50:-1])))
    return score

def build_model(env):
    in_dim = 1
    print(f"Shape of input space: {env.observation_space.shape}")
    print(f"Environment action space: {env.action_space.n}")
    
    # For CartPole-v0, observation space is shape {4,} as follows:
    # 0. Cart Pos
    # 1. Cart Velocity
    # 2. Pole Angle
    # 3. Pole Tip Velocity

    for i in env.observation_space.shape:
        in_dim *= i
    return nn.Sequential(
        nn.Linear(in_dim, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, env.action_space.n),
        nn.Softmax(dim=0)
    )

def main():
    env = gym.make('CartPole-v0')
    # best_model = None
    model = build_model(env)
    train_model(env, model, 10000)
    input("training done")
    test_model(env, model, delay=0.01)
    env.close()


def run_training_batch(template, env, epochs=500, clip=50):
    model = template(env)
    scores = train_model(env, model, epochs)
    return model, np.mean(scores[(-1 * clip):])

'''
    creates a pool of agents and returns the best model found
'''
def run_pool(env, template, pool_size=25):
    best_model = None
    best_score = 0
    for _ in range(pool_size):
        model, score = run_training_batch(template, env)
        if score > best_score:
            best_model = model
    return best_model

def kek_main():
    env = gym.make('CartPole-v0')
    #model = run_pool(env, build_model)
    init_model = build_model
    model, score = run_training_batch(env, init_model)
    input("training done")
    test_model(env, model, delay=0.05)

if __name__ == '__main__':
    main()
