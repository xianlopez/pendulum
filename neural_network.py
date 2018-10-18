import tensorflow as tf
import numpy as np



def define_network(batch_size, n_features):
    inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_features))
    net = tf.layers.dense(inputs, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1')
    net = tf.layers.dense(net, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2')
    net = tf.layers.dense(net, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output')
    gradients = tf.gradients(net, tf.trainable_variables())
    return inputs, net, gradients

def assign_to_variables():
    new_values_list = []
    assign_ops_list = []
    for variable in tf.trainable_variables():
        new_values = tf.placeholder(dtype=tf.float32, shape=variable.shape)
        assign_op = tf.assign(variable, new_values)
        new_values_list.append(new_values)
        assign_ops_list.append(assign_op)
    assign_op = tf.group(assign_ops_list)
    return assign_op, new_values_list



def put_angle_into_range(angle):
    if angle > 2 * np.pi:
        n = np.floor(angle / (2 * np.pi))
        angle = angle - n * 2 * np.pi
    elif angle < 0:
        n = np.floor(-angle / (2 * np.pi))
        angle = angle + n * 2 * np.pi
        angle = angle + 2 * np.pi
    return angle

class neural_netowrk_rl:
    def __init__(self, alpha, gamma, a_max, n_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.a_max = a_max
        self.all_actions = np.linspace(-a_max, a_max, n_actions)
        self.inputs, self.net, self.gradients = define_network(1, 3)
        self.assign_variables_op, self.new_values_list = assign_to_variables()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def compute_q(self, s, a):
        theta = put_angle_into_range(s[0])
        theta_dif = s[1]
        inputs = np.array([theta, theta_dif, a], dtype=np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        q_hat = self.sess.run(fetches=self.net, feed_dict={self.inputs: inputs})[0]
        # print('q_hat')
        # print(q_hat)
        # print(q_hat[0])
        q_hat = q_hat[0]
        return q_hat

    def compute_gradient_q(self, s, a):
        theta = put_angle_into_range(s[0])
        theta_dif = s[1]
        inputs = np.array([theta, theta_dif, a], dtype=np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        gradient = self.sess.run(fetches=self.gradients, feed_dict={self.inputs: inputs})
        return gradient

    def update(self, s_t, a_t, s_tp1, a_tp1, R_tp1):
        w = self.sess.run(fetches=tf.trainable_variables())
        q_prev = self.compute_q(s_t, a_t)
        print('q_prev')
        # print(len(q_prev))
        print(q_prev)
        q_next = self.compute_q(s_tp1, a_tp1)
        print('q_next')
        # print(len(q_next))
        print(q_next)
        q_gradient_prev = self.compute_gradient_q(s_t, a_t)
        print('q_gradient_prev')
        print(len(q_gradient_prev))
        print(q_gradient_prev)
        U = R_tp1 + self.gamma * q_next
        print('U')
        # print(len(U))
        print(U)
        for i in range(len(tf.trainable_variables())):
            increment = self.alpha * (U - q_prev) * q_gradient_prev[i]
            print('increment')
            print(len(increment))
            print(increment)
            print('w')
            print(len(w))
            print(w)
            w[i] = w[i] + increment
            print('w')
            print(len(w))
            print(w)
        feed_dict = {}
        print('creating dictionary')
        for i in range(len(tf.trainable_variables())):
            variable = tf.trainable_variables()[i]
            print(variable)
            feed_dict[variable] = w[i]
        print(feed_dict)
        self.sess.run(self.assign_variables_op, feed_dict=feed_dict)

    def compute_best_action(self, s):
        q_of_all_actions = np.zeros(shape=(len(self.all_actions)), dtype=np.float32)
        for i in range(len(self.all_actions)):
            a = self.all_actions[i]
            q_of_all_actions[i] = self.compute_q(s, a)
        best_action = self.all_actions[np.argmax(q_of_all_actions)]
        if np.random.rand() < 0.1:
            best_action = np.random.choice(self.all_actions)
        print('Action: ' + str(best_action))
        return best_action


##### O problema é que asigno as variables, e tamén as uso como input!!!!
##### Claro, porque as actualizo. Habería q ver como facer esta actualización licitamente.


def add():
    print("add")
    x = a + b
    return x

def last():
    print("last")
    return x

with tf.Session() as s:
    a = tf.Variable(tf.constant(0.),name="a")
    b = tf.Variable(tf.constant(0.),name="b")
    x = tf.constant(-1.)
    calculate= tf.cond(x.eval()==-1.,add,last)
    val = s.run([calculate], {a: 1., b: 2.})
    print(val) # 3
    print(s.run([calculate],{a:3.,b:4.})) # 7
    print(val) # 3