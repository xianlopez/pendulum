import tensorflow as tf
import numpy as np
import cv2
import gym

class dqn_atari:
    def __init__(self, rmsprop_learning_rate, rmsprop_decay, rmsprop_momentum, gamma, a_max, n_actions, n_features, batch_size):
        self.rmsprop_learning_rate = rmsprop_learning_rate
        self.rmsprop_decay = rmsprop_decay
        self.rmsprop_momentum = rmsprop_momentum
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_features = n_features
        self.batch_size = batch_size
        self.env = gym.make('SpaceInvadersDeterministic-v4')
        self.define_graph()
        self.sess = None
        self.memory_position = None

    def start_session(self, replay_memory_capacity):
        self.replay_memory_capacity = replay_memory_capacity
        self.D_state_t = np.zeros(shape=(self.replay_memory_capacity, self.n_features), dtype=np.float32)
        self.D_action_t_id = np.zeros(shape=(self.replay_memory_capacity), dtype=np.int32)
        self.D_reward_t = np.zeros(shape=(self.replay_memory_capacity), dtype=np.float32)
        self.D_state_tp1 = np.zeros(shape=(self.replay_memory_capacity, self.n_features), dtype=np.float32)
        self.D_terminal = np.zeros(shape=(self.replay_memory_capacity), dtype=np.bool)
        self.memory_position = 0
        self.memory_count = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def preprocessing(self, img):
        # The original images from the Atari environment 210x160x3 images.
        # We convert to grayscale, and rescale to 84x84.
        img_gray = np.mean(img, axis=-1).astype(np.uint8)
        img_prep = cv2.resize(img_gray, [84, 84])
        return img_prep

    def build_state_from_sequence_of_observations(self, observations):
        state = np.zeros(shape=(self.img_width, self.img_height, self.n_sim_steps_per_control_step), dtype=np.float32)
        for i in range(self.n_sim_steps_per_control_step):
            ob = observations[i]
            img_prep = self.preprocessing(ob)
            state[:, :, i] = img_prep
        return state

    def take_action_epislon_greedy(self, s):
        if np.random.rand() < self.epsilon_policy:
            a = self.env.action_space.sample()
        else:
            a = self.compute_best_action(s)
        return a

    def evaluate(self):
        pass

    def take_agent_step(self, action):
        reward_accum = 0
        observations = []
        for _ in range(self.n_sim_steps_per_control_step):
            observation_one_frame, reward, done, info = self.env.step(action)
            observations.append(observation_one_frame)
            reward_accum += reward
            if done:
                break
        reward = np.sign(reward_accum)
        if done and (len(observations) < self.n_sim_steps_per_control_step):
            completed = False
        else:
            completed = True
        return observations, reward, done, completed

    def initialize_episode(self):
        n_noop_steps = np.random.randint(31)
        for _ in n_noop_steps:
            self.env.step(0) # 0 is the No-Op action

    def simulate(self):
        self.start_session(self.replay_memory_capacity)
        count_steps_to_update_target_network = 0

        self.initialize_training()
        continue_playing = True
        step = 0
        while continue_playing:
            self.initialize_episode()
            observation_prev = self.env.reset()
            t = 0
            while True:
                # env.render()
                action = self.take_action_epislon_greedy(observation_prev)
                observations, reward, done, completed = self.take_agent_step(action)
                state_next = self.build_state_from_sequence_of_observations(observations)
                self.add_to_memory(state_prev, action, reward, state_next, done)
                if self.is_ready_to_train():
                    loss = self.train_step()
                if count_steps_to_update_target_network == self.n_steps_to_update_target_network:
                    count_steps_to_update_target_network = 0
                    self.update_target_network_parameters()
                state_prev = state_next
                t += 1
                step += 1
                if step % self.n_steps_evaluate == 0:
                    self.evaluate()
                if step == self.n_steps:
                    print('End of training')
                    continue_playing = False
                    break
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def add_to_memory(self, s_t, a_t, r_t, s_tp1, T):
        if self.memory_position is None:
            raise Exception('Memory buffer not initialized')
        self.D_state_t[self.memory_position, :] = s_t
        self.D_action_t_id[self.memory_position] = a_t
        self.D_reward_t[self.memory_position] = r_t
        self.D_state_tp1[self.memory_position, :] = s_tp1
        self.D_terminal[self.memory_position] = T
        # self.D[self.memory_position, :self.n_features] = s_t
        # self.D[self.memory_position, self.n_features:(self.n_features+1)] = a_t
        # self.D[self.memory_position, self.n_features+1] = r_t
        # self.D[self.memory_position, (self.n_features+2):(2*self.n_features+2)] = s_tp1
        # self.D[self.memory_position, 2*self.n_features+2] = np.float32(T)
        self.memory_position += 1
        if self.memory_position == self.replay_memory_capacity:
            self.memory_position = 0
        if self.memory_count < self.replay_memory_capacity:
            self.memory_count += 1

    def is_ready_to_train(self):
        if self.memory_count < self.replay_memory_capacity:
            return False
        else:
            return True

    def train_step(self):
        if self.sess is None:
            raise Exception('Session not initialized')
        if self.memory_count < self.replay_memory_capacity:
            raise Exception('Memory buffer not filled yet')
        # Sample mini-batch from replay memory:
        indices = np.random.choice(self.replay_memory_capacity, self.batch_size, False)
        feed_dict = {
            self.state_t: self.D_state_t[indices, :],
            self.action_t_id: self.D_action_t_id[indices],
            self.reward_t: self.D_reward_t[indices],
            self.state_tp1: self.D_state_tp1[indices, :],
            self.terminal: self.D_terminal[indices]
        }
        # batch = self.D[indices, :]
        # Take training step:
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def q_net(self, state):
        # state: (None, 84, 84, 4)
        net = tf.nn.conv2d(state, filter=[8, 8, 4, 32], strides=[1, 4, 4, 1], padding='SAME', name='conv1')
        net = tf.nn.relu(net, 'relu1')
        net = tf.nn.conv2d(net, filter=[4, 4, 32, 64], strides=[1, 2, 2, 1], padding='SAME', name='conv2')
        net = tf.nn.relu(net, 'relu2')
        net = tf.nn.conv2d(net, filter=[3, 3, 642, 64], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        net = tf.nn.relu(net, 'relu3')
        net = tf.layers.dense(net, 512, activation=tf.nn.relu, name='dense1')
        net = tf.layers.dense(net, self.n_actions, activation=None, name='output')
        return net

    def update_target_network_parameters(self):
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
        state_batched = np.tile(np.expand_dims(state, axis=0), [self.batch_size, 1])
        q_allactions = self.sess.run(self.q_st_allactions, feed_dict={self.state_t: state_batched})
        action_id = np.argmax(q_allactions)
        return action_id

    def define_graph(self):
        self.state_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.action_t_id = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))
        self.reward_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size))
        self.state_tp1 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.terminal = tf.placeholder(dtype=tf.bool, shape=(self.batch_size))
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







