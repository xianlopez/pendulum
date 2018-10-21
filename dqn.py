import tensorflow as tf
import numpy as np
import tools


class dqn:
    def __init__(self, rmsprop_learning_rate, rmsprop_decay, rmsprop_momentum, gamma, a_max, n_actions, n_features, batch_size):
        self.rmsprop_learning_rate = rmsprop_learning_rate
        self.rmsprop_decay = rmsprop_decay
        self.rmsprop_momentum = rmsprop_momentum
        self.gamma = gamma
        self.a_max = a_max
        self.n_actions = n_actions
        self.n_features = n_features
        self.batch_size = batch_size
        self.all_actions = np.linspace(-a_max, a_max, self.n_actions)
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
        # self.D = np.zeros(shape=(self.replay_memory_capacity, self.n_features * 2 + 3), dtype=np.float32)
        # Each sample has the form [s_t, a_t, r_t, s_tp1, T]
        self.memory_position = 0
        self.memory_count = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def add_to_memory(self, s_t, a_t, r_t, s_tp1, T):
        if self.memory_position is None:
            raise Exception('Memory buffer not initialized')
        a_t_id = self.get_action_id_from_action(a_t)
        self.D_state_t[self.memory_position, :] = s_t
        self.D_action_t_id[self.memory_position] = a_t_id
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
        net = tf.layers.dense(state, 10, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1')
        net = tf.layers.dense(net, 10, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2')
        net = tf.layers.dense(net, self.n_actions, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output')
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
        return self.all_actions[action_id]

    def get_action_id_from_action(self, action):
        return np.argmin(np.abs(self.all_actions - action))

    def get_action_from_action_id(self, action_id):
        return self.all_actions[action_id]

    def define_graph(self):
        # # all_actions_tf = tf.constant(self.all_actions, dtype=tf.float32)
        # # all_actions_tf_exp = tf.tile(tf.expand_dims(tf.constant(self.all_actions, dtype=tf.float32), axis=0), [self.batch_size, 1])  # (batch_size, n_actions)
        # self.input = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 2 * self.n_features + 3))
        # # Each sample has the form [s_t, a_t, r_t, s_tp1, T]
        # state_t = self.input[:, :self.n_features]
        # # action_t = self.input[:, self.n_features]
        # action_t_id = tf.cast(tf.round(self.input[:, self.n_features]), tf.int32)
        # reward_t = self.input[:, self.n_features+1]
        # state_tp1 = self.input[:, (self.n_features+1):(2*self.n_features+2)]
        # terminal = self.input[:, 2*self.n_features+2]


        self.state_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.action_t_id = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))
        self.reward_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size))
        self.state_tp1 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.terminal = tf.placeholder(dtype=tf.bool, shape=(self.batch_size))




        with tf.variable_scope('q_net'):
            self.q_st_allactions = self.q_net(self.state_t)  # (batch_size, n_actions)
        print('jau')
        print('self.q_st_allactions')
        print(self.q_st_allactions)
        print('self.action_t_id')
        print(self.action_t_id)
        # q_st_at = self.q_st_allactions[:, self.action_t_id]
        # self.q_st_allactions = tf.Print(self.q_st_allactions, [self.q_st_allactions], 'self.q_st_allactions')
        # self.action_t_id = tf.Print(self.action_t_id, [self.action_t_id], 'self.action_t_id')
        q_st_at = tf.gather(self.q_st_allactions, self.action_t_id, axis=-1)
        # q_st_at = tf.Print(q_st_at, [q_st_at], 'q_st_at')
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





# loss negativa???


