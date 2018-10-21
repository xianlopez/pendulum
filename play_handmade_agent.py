import numpy as np
import handmade_agent
import environment

n_episodes = 200
n_steps_per_episode = 500

n_sim_steps_per_control_step = 20

n_actions = 5
a_max = 3

agent = handmade_agent.handmade_agent(a_max, n_actions)

def take_action_from_state(s):
    a = agent.compute_best_action(s)
    return a

def simulate():
    total_undiscounted_return = 0
    for ep in range(n_episodes):
        episode_undiscounted_return = 0
        s_prev = [np.random.rand() * 2 * np.pi, 0]
        environment.plot_state(s_prev, ep + 1, 0)
        for i in range(n_steps_per_episode):
            action = take_action_from_state(s_prev)
            reward = 0
            for _ in range(n_sim_steps_per_control_step):
                s_next = environment.update_state(s_prev, action)
                reward += environment.reward_from_state(s_next)
                episode_undiscounted_return += reward
            environment.plot_state(s_next, ep + 1, i + 1, reward)
            s_prev = s_next
        print('Episde undiscounted return: ' + str(episode_undiscounted_return))
        total_undiscounted_return += episode_undiscounted_return
    print('Total undiscounted return: ' + str(total_undiscounted_return))
    print('Done.')

if __name__ == '__main__':
    simulate()







