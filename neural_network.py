import tensorflow as tf
import numpy as np



def define_network(batch_size, n_features):
    print('defining network')
    inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_features))
    net = tf.layers.dense(inputs, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense1')
    net = tf.layers.dense(net, 5, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='dense2')
    net = tf.layers.dense(net, 1, activation=tf.sigmoid, kernel_initializer=tf.initializers.random_normal, name='output')
    gradients = tf.gradients(net, tf.trainable_variables())
    for variable in tf.trainable_variables():
        print('')
        print(variable)
    print(gradients)
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
        q_hat = q_hat[0]
        return q_hat

    def compute_gradient_q(self, s, a):
        theta = put_angle_into_range(s[0])
        theta_dif = s[1]
        inputs = np.array([theta, theta_dif, a], dtype=np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        gradient = self.sess.run(fetches=self.gradients, feed_dict={self.inputs: inputs})
        return gradient

    def train(self, D):
        # D: (n_batches, batch_size, n_states + 1 + n_states + 1 + 1). Last dimension: [s_t, a_t, s_tp1, a_tp1, R_tp1]
        n_batches = D.shape[0]
        batch_size = D.shape[1]
        n_states = int((D.shape[2] - 3) / 2)
        for b in range(n_batches):
            print('batch ' + str(b + 1))
            w = self.sess.run(fetches=tf.trainable_variables())
            for i in range(batch_size):
                print('    step ' + str(i + 1))
                current_experience = D[b, i, :]
                s_t = current_experience[:n_states]
                a_t = current_experience[n_states]
                s_tp1 = current_experience[(n_states+1):(2*n_states+1)]
                a_tp1 = current_experience[2*n_states+1]
                R_tp1 = current_experience[(2*n_states+2):]
                q_prev = self.compute_q(s_t, a_t)
                q_next = self.compute_q(s_tp1, a_tp1)
                q_gradient_prev = self.compute_gradient_q(s_t, a_t)
                U = R_tp1 + self.gamma * q_next
                for i in range(len(tf.trainable_variables())):
                    increment = self.alpha * (U - q_prev) * q_gradient_prev[i]
                    w[i] = w[i] + increment
            feed_dict = {}
            for i in range(len(self.new_values_list)):
                variable = self.new_values_list[i]
                feed_dict[variable] = w[i]
            self.sess.run(self.assign_variables_op, feed_dict=feed_dict)

    def update(self, s_t, a_t, s_tp1, a_tp1, R_tp1):
        w = self.sess.run(fetches=tf.trainable_variables())
        q_prev = self.compute_q(s_t, a_t)
        q_next = self.compute_q(s_tp1, a_tp1)
        q_gradient_prev = self.compute_gradient_q(s_t, a_t)
        U = R_tp1 + self.gamma * q_next
        for i in range(len(tf.trainable_variables())):
            increment = self.alpha * (U - q_prev) * q_gradient_prev[i]
            w[i] = w[i] + increment
        feed_dict = {}
        for i in range(len(self.new_values_list)):
            variable = self.new_values_list[i]
            feed_dict[variable] = w[i]
        self.sess.run(self.assign_variables_op, feed_dict=feed_dict)

    def compute_best_action(self, s):
        q_of_all_actions = np.zeros(shape=(len(self.all_actions)), dtype=np.float32)
        for i in range(len(self.all_actions)):
            a = self.all_actions[i]
            q_of_all_actions[i] = self.compute_q(s, a)
        best_action = self.all_actions[np.argmax(q_of_all_actions)]
        if np.random.rand() < 0.1:
            best_action = np.random.choice(self.all_actions)
        # print('Action: ' + str(best_action))
        return best_action


##### O problema é que asigno as variables, e tamén as uso como input!!!!
##### Claro, porque as actualizo. Habería q ver como facer esta actualización licitamente.


# def add():
#     print("add")
#     x = a + b
#     return x
#
# def last():
#     print("last")
#     return x
#
# with tf.Session() as s:
#     a = tf.Variable(tf.constant(0.),name="a")
#     b = tf.Variable(tf.constant(0.),name="b")
#     x = tf.constant(-1.)
#     calculate= tf.cond(x.eval()==-1.,add,last)
#     val = s.run([calculate], {a: 1., b: 2.})
#     print(val) # 3
#     print(s.run([calculate],{a:3.,b:4.})) # 7
#     print(val) # 3