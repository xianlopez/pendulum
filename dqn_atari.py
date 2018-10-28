import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt
import os
import replay_memory

class dqn_atari:
    def __init__(self):
        self.env = gym.make('SpaceInvadersDeterministic-v4')
        self.rmsprop_learning_rate = 0.00025
        self.rmsprop_decay = 0.95
        self.rmsprop_momentum = 0.95
        self.gamma = 0.99
        self.batch_size = 32
        self.img_width = 84
        self.img_height = 84
        # self.action_repeat = 4  # For all games except Space Invaders
        self.action_repeat = 3  # For Space Invaders
        self.epsilon_policy = 0.1
        self.n_episodes_evaluation = 30
        self.agent_history_length = 4
        self.update_frequency = 4
        self.no_op_max = 30
        self.no_op_action = 0
        self.initial_exploration = 1
        self.final_exploration = 0.1

        # self.n_steps_initialization = 200
        # self.n_steps_evaluate = 250
        # self.target_network_update_frequency = int(1e1)
        # self.n_steps_train = int(1e3) * 2
        # self.replay_memory_capacity = int(1e2) * 4
        # self.n_steps_display = 100
        # self.final_exploration_frame = int(1e2)

        self.n_steps_initialization = 50000
        self.n_steps_evaluate = 250000
        self.target_network_update_frequency = int(1e4)
        self.n_steps_train = int(1e7)
        self.replay_memory_capacity = int(1e6)
        self.n_steps_display = 1000
        self.final_exploration_frame = int(1e6)

        self.replay_memory = replay_memory.replay_memory(self.replay_memory_capacity, self.batch_size, self.img_width, self.img_height, self.agent_history_length)

        self.render = False
        self.render_evaluation = False
        self.terminate_on_life_loss = True
        self.n_actions = self.env.action_space.n
        self.define_graph()
        self.sess = None
        self.best_score = None
        self.steps_and_scores = []
        self.last_env_screen = None

    def start_session(self):
        print('Starting session')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def take_action_epislon_greedy(self, s):
        if s is None:
            a = self.env.action_space.sample()
        else:
            # print(self.epsilon_policy)
            if np.random.rand() < self.epsilon_policy:
                a = self.env.action_space.sample()
            else:
                a = self.compute_best_action(s)
        return a

    def evaluate(self, step):
        print('Evaluating')
        self.epsilon_policy = 0.05
        continue_playing = True
        scores = np.zeros(shape=(self.n_episodes_evaluation), dtype=np.float32)
        for ep in range(self.n_episodes_evaluation):
            screen_t = self.initialize_episode()  # Get screen of time t.
            while True:
                if self.render_evaluation:
                    self.env.render()
                state = self.replay_memory.build_state_with_new_screen(screen_t)
                # self.replay_memory.plot_state(state, 'state_' + str(step))
                # cv2.waitKey(0)
                action = self.take_action_epislon_greedy(state)
                screen_tp1, reward_t, terminal_t = self.take_agent_step(action)  # Get screen of time t plus 1, and reward and terminal of time t.
                scores[ep] += reward_t
                if terminal_t:
                    break
        mean_score = np.mean(scores)
        print('Mean score: ' + str(mean_score))
        self.steps_and_scores.append([step, mean_score])
        # Plot scores history:
        self.plot_scores()
        # Save the network weights if we improved the score:
        save_weights = False
        if self.best_score is None:
            save_weights = True
        elif self.best_score < mean_score:
            save_weights = True
        if save_weights:
            self.best_score = mean_score
            print('Saving network weights')
            this_file_folder = os.path.dirname(os.path.abspath(__file__))
            save_path = self.saver.save(self.sess, os.path.join(this_file_folder, 'model'), global_step=step)
            print('Model saved to ' + save_path)
        return mean_score

    def take_agent_step(self, action):
        reward_accum = 0
        for _ in range(self.action_repeat):
            screen, reward, done, info = self.env.step(action)
            self.new_env_screen = screen
            if self.last_env_screen is not None:
                screen = np.maximum(screen, self.last_env_screen)
            self.last_env_screen = self.new_env_screen
            remaining_lives = info['ale.lives']
            if self.initial_lives == None:
                self.initial_lives = remaining_lives
            elif remaining_lives < self.initial_lives and self.terminate_on_life_loss:
                done = True
            reward_accum += reward
            if done:
                break
        reward = min(max(reward_accum, -1), 1)  # Clipping reward to [-1, 1] to handle different reward sizes in different games.
        # if np.abs(reward_accum) < 1e-4:
        #     reward = 0
        # else:
        #     reward = np.sign(reward_accum)
        return screen, reward, done

    def initialize_training(self):
        print('Initializing training')
        continue_playing = True
        step = 0
        while continue_playing:
            self.initial_lives = None
            screen_t = self.initialize_episode()  # Get screen of time t.
            terminal_t = False
            while True:
                action_t = self.env.action_space.sample()
                screen_tp1, reward_t, terminal_tp1 = self.take_agent_step(action_t)  # Get screen of time t plus 1, and reward and terminal of time t.
                self.replay_memory.add_to_memory(screen_t, action_t, reward_t, terminal_t)
                screen_t = screen_tp1
                terminal_t = terminal_tp1
                step += 1
                if step == self.n_steps_initialization:
                    print('End of initialization')
                    continue_playing = False
                    break
                if terminal_tp1:
                    break

    def initialize_episode(self):
        # print('Initializing episode')
        n_noop_steps = np.random.randint(self.no_op_max + 1)
        screen = self.env.reset()
        self.last_env_screen = screen
        for _ in range(n_noop_steps):
            screen, reward, done, info = self.env.step(self.no_op_action)
            self.new_env_screen = screen
            screen = np.maximum(screen, self.last_env_screen)
            self.last_env_screen = self.new_env_screen
        return screen

    def adjust_policy_epsilon(self, step):
        if step < self.final_exploration_frame:
            self.epsilon_policy = float(step) / self.final_exploration_frame * (self.final_exploration - self.initial_exploration) + self.initial_exploration
        else:
            self.epsilon_policy = self.final_exploration

    # def perceive_and_act(self, screen_new, reward_new, terminal_new):
    #     self.replay_memory.add_to_memory(self.screen_last, self.action_last, reward_new, self.terminal_last)
    #     state = self.replay_memory.build_state_with_new_screen(screen_new)
    #     action_new = self.take_action_epislon_greedy(state)
    #     self.screen_last = screen_new
    #     self.action_last = action_new
    #     self.terminal_last = terminal_new
    #     return action_new

    def train(self):
        print('Starting training')
        self.start_session()
        self.initialize_training()
        continue_playing = True
        step = 0
        loss = None
        while continue_playing:
            self.initial_lives = None
            screen_t = self.initialize_episode()  # Get screen of time t.
            terminal_t = False
            while True:
                step += 1
                if self.render:
                    self.env.render()
                self.adjust_policy_epsilon(step)
                state = self.replay_memory.build_state_with_new_screen(screen_t)
                # self.replay_memory.plot_state(state, 'state_' + str(step))
                # self.replay_memory.plot_state(state, 'state')
                # cv2.waitKey(0)
                action_t = self.take_action_epislon_greedy(state)
                screen_tp1, reward_t, terminal_tp1 = self.take_agent_step(action_t)  # Get screen of time t plus 1, and reward and terminal of time t.
                self.replay_memory.add_to_memory(screen_t, action_t, reward_t, terminal_t)
                screen_t = screen_tp1
                terminal_t = terminal_tp1
                if self.replay_memory.is_ready() and step % self.update_frequency == 0:
                    loss = self.train_step()
                    if step % self.target_network_update_frequency == 0:
                        self.update_target_network_parameters()
                    if step % self.n_steps_display == 0 and loss is not None:
                        print('step = ' + str(step) + ', loss = ' + str(loss))
                    if step % self.n_steps_evaluate == 0:
                        self.evaluate(step)
                if step == self.n_steps_train:
                    print('End of training')
                    continue_playing = False
                    break
                if terminal_t:
                    break

    def plot_scores(self):
        if len(self.steps_and_scores) > 1:
            steps_grid = []
            scores_values = []
            for i in range(len(self.steps_and_scores)):
                steps_grid.append(self.steps_and_scores[i][0])
                scores_values.append(self.steps_and_scores[i][1])
            plt.figure('scores')
            plt.plot(steps_grid, scores_values, 'r-')
            # plt.show()
            this_file_folder = os.path.dirname(os.path.abspath(__file__))
            plt.savefig(os.path.join(this_file_folder, 'scores_history.png'))

    def train_step(self):
        if self.sess is None:
            raise Exception('Session not initialized')
        if not self.replay_memory.is_ready():
            raise Exception('Memory buffer not filled yet')
        # Sample mini-batch from replay memory:
        batch_states_t, batch_actions_t, batch_rewards_t, batch_states_tp1, batch_terminals = self.replay_memory.sample_batch()
        feed_dict = {
            self.state_t: batch_states_t,
            self.action_t_id: batch_actions_t,
            self.reward_t: batch_rewards_t,
            self.state_tp1: batch_states_tp1,
            self.terminal: batch_terminals
        }
        # Take training step:
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def q_net(self, state):
        # state: (None, 84, 84, 4)
        print('state')
        print(state)

        # state = tf.Print(state, [tf.shape(state)], 'state shape')

        state = tf.cast(state, tf.float32)
        print('state')
        print(state)

        net = slim.conv2d(state, num_outputs=16, kernel_size=8, stride=4, padding='SAME', activation_fn=tf.nn.relu, biases_initializer=None, scope='conv1')
        # (None, 21, 21, 16)
        print('after conv1')
        print(net)

        net = slim.conv2d(net, num_outputs=32, kernel_size=4, stride=2, padding='SAME', activation_fn=tf.nn.relu, biases_initializer=None, scope='conv2')
        # (None, 11, 11, 32)
        print('after conv2')
        print(net)

        net = slim.conv2d(net, num_outputs=256, kernel_size=11, stride=1, padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None, scope='dense1')
        # (None, 1, 1, 16)
        print('after dense1')
        print(net)

        net = slim.conv2d(net, num_outputs=self.n_actions, kernel_size=1, stride=1, padding='VALID', activation_fn=None, biases_initializer=None, scope='output')
        # (None, 1, 1, 16)
        print('after output')
        print(net)

        net = tf.squeeze(net, axis=[1, 2])
        # (None, self.n_actions)
        print('after squeeze')
        print(net)

        for variable in tf.trainable_variables():
            print(variable)

        return net

    def update_target_network_parameters(self):
        print('Updating target network parameters')
        if self.sess is None:
            raise Exception('Session not initialized')
        self.sess.run(self.update_target_op)

    def create_update_target_op(self):
        q_net_variables = [var for var in tf.trainable_variables() if 'q_net/' in var.name]
        q_net_target_variables = [var for var in tf.trainable_variables() if 'q_net_target/' in var.name]
        update_ops = []
        for var in q_net_variables:
            name = var.name[len('q_net'):]
            target_var = [v for v in q_net_target_variables if name in v.name][0]
            update_ops.append(tf.assign(target_var, var))
        self.update_target_op = tf.group(update_ops)

    def compute_best_action(self, state):
        if self.sess is None:
            raise Exception('Session not initialized')
        state_exp = np.expand_dims(state, axis=0)  # Add batch dimension (with size 1)
        # state_batched = np.tile(np.expand_dims(state, axis=0), [self.batch_size, 1, 1, 1])
        q_allactions = self.sess.run(self.q_st_allactions, feed_dict={self.state_t: state_exp})
        action_id = np.argmax(q_allactions[0, :])
        return action_id

    def define_graph(self):
        # self.state_t = tf.placeholder(dtype=tf.uint8, shape=(self.batch_size, self.img_width, self.img_height, self.agent_history_length))
        # self.action_t_id = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))
        # self.reward_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size))
        # self.state_tp1 = tf.placeholder(dtype=tf.uint8, shape=(self.batch_size, self.img_width, self.img_height, self.agent_history_length))
        # self.terminal = tf.placeholder(dtype=tf.bool, shape=(self.batch_size))
        self.state_t = tf.placeholder(dtype=tf.uint8, shape=(None, self.img_width, self.img_height, self.agent_history_length))
        self.action_t_id = tf.placeholder(dtype=tf.int32, shape=(None))
        self.reward_t = tf.placeholder(dtype=tf.float32, shape=(None))
        self.state_tp1 = tf.placeholder(dtype=tf.uint8, shape=(None, self.img_width, self.img_height, self.agent_history_length))
        self.terminal = tf.placeholder(dtype=tf.bool, shape=(None))
        with tf.variable_scope('q_net'):
            self.q_st_allactions = self.q_net(self.state_t)  # (batch_size, n_actions)
        q_st_at = tf.gather(self.q_st_allactions, self.action_t_id, axis=-1)
        with tf.variable_scope('q_net_target'):
            q_stp1_allactions = self.q_net(self.state_tp1)  # (batch_size, n_actions)
        q_stp1_best_action_tp1 = tf.reduce_max(q_stp1_allactions, axis=-1)  # (batch_size)
        case_not_terminal = self.reward_t + self.gamma * q_stp1_best_action_tp1
        y = tf.where(self.terminal, self.reward_t, case_not_terminal)
        self.loss = tf.reduce_sum(tf.square(y - q_st_at))
        # self.loss = tf.Print(self.loss, [self.loss], 'loss')
        optimizer = tf.train.RMSPropOptimizer(self.rmsprop_learning_rate, self.rmsprop_decay, self.rmsprop_momentum)
        self.train_op = optimizer.minimize(self.loss)
        self.create_update_target_op()
        self.saver = tf.train.Saver(name='net_saver')







