## Reinforcement learning for the pendulum problem

import numpy as np
import matplotlib.pyplot as plt
import time
from linear_continuous_q_learning import linear_continuous_q_learning, put_angle_into_range
from neural_network import neural_netowrk_rl

plt.ion()

m = 2  # kg
g = 9.8  # N / kg
l = 0.2  # m

friction = 0.001

I = m * l * l

# theta_0 = np.pi / 2.0  # rad (comeza na dereita)
# theta_0 = np.pi   # rad (comeza arriba)
# theta_0 = 0  # rad (comeza abaixo)
# w_0 = 0  # rad / s
# s_0 = [theta_0, w_0]   # Estado inicial, dado polo angulo e a velocidade angular.

n_episodes = 200
n_steps_per_episode = 500
dt = 0.01  # s

n_sim_steps_per_control_step = 20

before = None

alpha = 0.001
gamma = 0.99
a_max = 3
# q_learning = linear_continuous_q_learning(alpha, gamma, a_max)
q_learning = neural_netowrk_rl(alpha, gamma, a_max, 5)

# def reward_from_state(s):
#     theta = s[0]
#     theta_dif = s[1]
#     reward = 0
#     # We reward a high position (the higher, the better):
#     coef_highness = 1
#     reward += coef_highness * (-1) * np.cos(theta)
#     return reward

def reward_from_state(s):
    theta = s[0]
    theta_dif = s[1]
    reward = 0
    if np.cos(theta) < -0.5:
        reward = 1
    else:
        reward = 0
    reward -= np.abs(theta_dif) / 10
    return reward

def take_action_from_state(s):
    a = q_learning.compute_best_action(s)
    return a
    # return 3

def plot_state(s, episode, step, reward = 0):
    global before
    theta = s[0]
    theta_dif = s[1]
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    plt.figure('Pendulum')
    plt.clf()
    if reward > 1e-5:
        plt.plot(0, 0, '*g', linewidth=3)
    elif reward < -1e-5:
        plt.plot(0, 0, '*r', linewidth=3)
    plt.plot([0, x], [0, y], '-b')
    plt.plot(x, y, '*r', linewidth=3)
    plt.xlim((-1.1 * l, 1.1 * l))
    plt.ylim((-1.1 * l, 1.1 * l))
    plt.xlabel('theta: ' + str(np.round(theta, 1)) + '  theta_dif: ' + str(np.round(theta_dif, 1)))
    # plt.title('Episode: ' + str(episode) + '. Step: ' + str(step) + '. Time: ' + str(step * dt * n_sim_steps_per_control_step))
    plt.title('Episode: ' + str(episode) + '. Step: ' + str(step) + '.')
    plt.show()
    now = time.time()
    if before is None:
        wait = 1e-5
    else:
        if now - before > dt:
            wait = 1e-5
        else:
            wait = dt - (now - before)
    plt.pause(wait)
    before = now
    return

def update_state(s_prev, action):
    # action (torque): N * m
    theta_prev = s_prev[0]
    w_prev = s_prev[1]
    gravity = m * g * l * np.sin(theta_prev)
    alpha = (action - gravity) / I - friction * w_prev * np.abs(w_prev)
    w_next = w_prev + dt * alpha
    theta_next = put_angle_into_range(theta_prev + dt * w_next)
    s_next = [theta_next, w_next]
    return s_next

def simulate():
    total_undiscounted_return = 0
    for ep in range(n_episodes):
        episode_undiscounted_return = 0
        action_prev = None
        s_prev = [np.random.rand() * 2 * np.pi, 0]
        reward = reward_from_state(s_prev)
        # print('Reward: ' + str(reward))
        # plot_state(s_prev, ep + 1, 0)
        for i in range(n_steps_per_episode):
            action = take_action_from_state(s_prev)
            # print('Step ' + str(i + 1) + ' / ' + str(n_steps) +
            #       ': (' + str(s_prev[0]) + ', ' + str(s_prev[1]) + '). Action: ' + str(action))
            reward = 0
            for _ in range(n_sim_steps_per_control_step):
                s_next = update_state(s_prev, action)
                reward += reward_from_state(s_next)
                episode_undiscounted_return += reward
            if action_prev is not None:
                q_learning.update(s_prev, action_prev, s_next, action, reward)
            # print('Reward: ' + str(reward))
            # if ep == n_episodes -1:
            #     plot_state(s_next, ep + 1, i + 1, reward)
            plot_state(s_next, ep + 1, i + 1, reward)
            s_prev = s_next
            action_prev = action
        print('Episde undiscounted return: ' + str(episode_undiscounted_return))
        total_undiscounted_return += episode_undiscounted_return
    print('Total undiscounted return: ' + str(total_undiscounted_return))
    print('Done.')

if __name__ == '__main__':
    simulate()







