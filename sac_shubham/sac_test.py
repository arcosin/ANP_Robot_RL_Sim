from sac import SACAgent
import gym

# Implementation referenced from https://github.com/cyoon1729/Policy-Gradient-Methods/tree/master/sac
# with minor modifications and changes

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

env = gym.make("Pendulum-v0")

tau = 0.005
gamma = 0.99
value_lr = 3e-3
q_lr = 3e-3
policy_lr = 3e-3
buffer_maxlen = 1000000

state = env.reset()

print("Starting here\n")
agent = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)

# train
episode_rewards = mini_batch_train(env, agent, 50, 500, 64)