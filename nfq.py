import tensorflow as tf
import numpy as np



def put_angle_into_range(angle):
    if angle > 2 * np.pi:
        n = np.floor(angle / (2 * np.pi))
        angle = angle - n * 2 * np.pi
    elif angle < 0:
        n = np.floor(-angle / (2 * np.pi))
        angle = angle + n * 2 * np.pi
        angle = angle + 2 * np.pi
    # print('angle = ' + str(angle))
    return angle

set_of_actions = []

def make_network(input, reuse):
    net = tf.layers.dense(input, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1', reuse=reuse)
    net = tf.layers.dense(net, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2', reuse=reuse)
    net = tf.layers.dense(net, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output', reuse=reuse)
    return net

def compute_q(state, action, reuse):
    # state: (batch_size, n_states)
    # action: (batch_size)
    input = tf.concat([state, tf.expand_dims(action, axis=-1)], axis=-1)
    q = make_network(input, reuse)
    return q

def cost(state, action):


def compute_error(s_prev, action, s_next):
    q_prev = compute_q(s_prev, action, False)
    all_next_qs = []
    for a in set_of_actions:
        q_next_a = compute_q(s_next, a, True)
        all_next_qs.append(q_next_a)
    all_next_qs = tf.stack(all_next_qs, axis=-1)
    best_next_q = tf.maximum(all_next_qs, axis=-1)
    error = tf.square(q_prev - (cost(s_prev, action, s_next) + best_next_q))
    return error




def train(D):





