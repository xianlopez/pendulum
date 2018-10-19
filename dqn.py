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
        self.n_features = n_features
        self.batch_size = batch_size
        self.all_actions = np.linspace(-a_max, a_max, n_actions)
        self.define_graph()
        self.sess = None

    def start_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self, D):
        # D: List experiences given by [s_t, a_t, r_t, s_tp1, T]
        if self.sess is None:
            raise Exception('Session not initialized')

    def q_net(self, state, action):
        inputs = tf.concat([state, tf.expand_dims(action, axis=-1)], axis=-1)
        net = tf.layers.dense(inputs, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1')
        net = tf.layers.dense(net, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2')
        net = tf.layers.dense(net, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output')
        return net

    def q_net_target(self, state, action, reuse):
        inputs = tf.concat([state, tf.expand_dims(action, axis=-1)], axis=-1)
        net = tf.layers.dense(inputs, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1', reuse=reuse)
        net = tf.layers.dense(net, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2', reuse=reuse)
        net = tf.layers.dense(net, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output', reuse=reuse)
        return net

    def update_target_network_parameters(self):
        pass

    def define_graph(self):
        self.state_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.state_tp1 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.n_features))
        self.action_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size))
        self.reward_t = tf.placeholder(dtype=tf.float32, shape=(self.batch_size))
        self.terminal = tf.placeholder(dtype=tf.bool, shape=(self.batch_size))
        q_st_at = self.q_net(self.state_t, self.action_t)
        q_stp1_allactions = tf.zeros(shape=(self.batch_size, 0))
        all_actions_tf = tf.constant(self.all_actions, dtype=tf.float32)
        reuse = False
        for i in range(len(self.all_actions)):
            a = all_actions_tf[i]
            a_over_batch = tf.tile(tf.expand_dims(a, axis=0), [self.batch_size])
            q_st_a = self.q_net_target(self.state_tp1, a_over_batch, reuse)
            q_stp1_allactions = tf.concat([q_stp1_allactions, q_st_a], axis=-1)
            reuse = True
        q_stp1_best_action_tp1 = tf.reduce_max(q_stp1_allactions, axis=-1)  # (batch_size)
        case_not_terminal = self.reward_t + self.gamma * q_stp1_best_action_tp1
        y = tf.where(self.terminal, self.reward_t, case_not_terminal)
        loss = tf.square(y - q_st_at)
        optimizer = tf.train.RMSPropOptimizer(self.rmsprop_learning_rate, self.rmsprop_decay, self.rmsprop_momentum)
        self.train_op = optimizer.minimize(loss)



a = dqn(0.001, 0.9, 0.9, 0.99, 3, 5, 2, 10)


