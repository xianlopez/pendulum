import dqn
import environment
import numpy as np
import matplotlib.pyplot as plt

n_episodes = 500
n_steps_per_episode = 500
n_sim_steps_per_control_step = 20

rmsprop_learning_rate = 0.00025
rmsprop_decay = 0.95
rmsprop_momentum = 0.95
gamma = 0.99
a_max = 3
n_actions = 5
n_features = 2
batch_size = 32

replay_memory_capacity = 5000

n_steps_to_update_target_network = 10

agent = dqn.dqn(rmsprop_learning_rate, rmsprop_decay, rmsprop_momentum, gamma, a_max, n_actions, n_features, batch_size)

def take_action_from_state(s):
    a = agent.compute_best_action(s)
    return a

def simulate():
    agent.start_session(replay_memory_capacity)
    total_undiscounted_return = 0
    count_steps_to_update_target_network = 0
    episodes_losses = []
    for ep in range(n_episodes):
        episode_undiscounted_return = 0
        episode_mean_loss = 0
        s_prev = [np.random.rand() * 2 * np.pi, 0]
        # environment.plot_state(s_prev, ep + 1, 0)
        for i in range(n_steps_per_episode):
            count_steps_to_update_target_network += 1
            T = i == (n_steps_per_episode - 1)
            action = take_action_from_state(s_prev)
            reward = 0
            for _ in range(n_sim_steps_per_control_step):
                s_next = environment.update_state(s_prev, action)
                reward += environment.reward_from_state(s_next)
                episode_undiscounted_return += reward
            agent.add_to_memory(s_prev, action, reward, s_next, T)
            if agent.is_ready_to_train():
                loss = agent.train_step()
                episode_mean_loss += loss
            if count_steps_to_update_target_network == n_steps_to_update_target_network:
                count_steps_to_update_target_network = 0
                agent.update_target_network_parameters()
            # environment.plot_state(s_next, ep + 1, i + 1, reward)
            s_prev = s_next
        if np.abs(episode_mean_loss) > 1e-6:
            episode_mean_loss /= float(n_steps_per_episode)
        episodes_losses.append(episode_mean_loss)
        print('Episde mean loss: ' + str(episode_mean_loss))
        print('Episde undiscounted return: ' + str(episode_undiscounted_return))
        total_undiscounted_return += episode_undiscounted_return
    print('Total undiscounted return: ' + str(total_undiscounted_return))
    # plt.ion()
    plt.figure('loss')
    plt.plot(np.array(episodes_losses, dtype=np.float32), 'r-')
    plt.show()
    # plt.pause()
    print('Done.')
    return episodes_losses


episodes_losses = simulate()

if __name__ == '__main__':
    simulate()




