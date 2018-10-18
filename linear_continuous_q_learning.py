import numpy as np

# s = (theta, theta_dif)
# a in [-a_max, a_max]

# def put_angle_into_range(angle):
#     if angle > 0:
#         n = np.floor(angle / (2 * np.pi))
#     else:
#         n = np.floor(-angle / (2 * np.pi))
#     if n > 0.5:
#         angle = angle / (n * np.pi)
#     return angle

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

class linear_continuous_q_learning:
    def __init__(self, alpha, gamma, a_max):
        self.alpha = alpha
        self.gamma = gamma
        self.a_max = a_max
        # self.w = np.zeros(shape=(8), dtype=np.float32)
        # self.w = np.zeros(shape=(11), dtype=np.float32)
        # self.w = np.zeros(shape=(4), dtype=np.float32)
        # self.w = np.zeros(shape=(5), dtype=np.float32)
        # self.w = np.zeros(shape=(13), dtype=np.float32)
        self.w = np.zeros(shape=(10), dtype=np.float32)
        self.all_actions = np.linspace(-a_max, a_max, 5)

    def get_x(self, s, a):
        theta = put_angle_into_range(s[0])
        theta_dif = s[1]
        # x = np.array([theta, theta**2, theta_dif, theta_dif**2, a, a**2, a * theta, a * theta_dif], dtype=np.float32)
        # x = np.array([np.cos(theta), np.cos(theta)**2, np.sin(theta), np.sin(theta)**2, theta_dif, theta_dif**2, a, a**2, a * np.cos(theta), a * np.sin(theta), a * theta_dif], dtype=np.float32)
        # x = np.array([np.cos(theta) * a, np.sin(theta) * a, np.cos(theta) * a**2, np.sin(theta) * a**2], dtype=np.float32)
        # x = np.array([np.cos(theta) * a, np.sin(theta) * a, np.cos(theta) * a**2, np.sin(theta) * a**2, np.cos(theta) * theta_dif * a], dtype=np.float32)
        # x = np.array([np.cos(theta), np.sin(theta), np.cos(theta)**2, np.sin(theta)**2, np.cos(theta) * np.sin(theta), theta_dif, theta_dif**2, a * np.cos(theta), a * np.sin(theta), a * theta_dif, a * np.cos(theta)**2, a * np.sin(theta)**2, a * theta_dif**2], dtype=np.float32)
        x = np.array([np.pi - theta, (np.pi - theta) ** 2, (np.pi - theta) * a, (np.pi - theta) ** 2 * a, theta_dif, theta_dif**2, theta_dif * a, theta_dif**2 * a, theta_dif * (np.pi - theta), theta_dif * (np.pi - theta) * a], dtype=np.float32)
        return x

    def compute_q(self, s, a):
        q_hat = np.inner(self.get_x(s, a), self.w)
        # print('q = ' + str(q_hat))
        return q_hat

    def update(self, s_t, a_t, s_tp1, a_tp1, R_tp1):
        x = self.get_x(s_t, a_t)
        U = R_tp1 + self.gamma * self.compute_q(s_tp1, a_tp1)
        # print('U = ' + str(U))
        increment = self.alpha * (U - self.compute_q(s_t, a_t)) * x
        self.w = self.w + increment
        # print('x = ' + str(x))
        # print('w = ' + str(self.w))

    def compute_best_action(self, s):
        theta = put_angle_into_range(s[0])
        # print('angle = ' + str(theta))
        theta_dif = s[1]
        if np.cos(theta) > 0: # metade inferior
            if np.abs(theta_dif) > 14:
                best_action = 0
            else:
                best_action = np.sign(theta_dif) * 2
            # if np.sin(theta) * theta_dif > 0:
            #     best_action = np.sign(np.sin(theta)) * 2
            # else:
            #     best_action = -np.sign(np.sin(theta)) * 2
        else: # metade superior
            if np.cos(theta) < -0.5: # Moi cerca do tope
                if theta_dif * np.sign(np.sin(theta)) > 2:
                    best_action = -np.sign(theta_dif) * 2
                else:
                    best_action = min(max(np.sin(theta) * 10, -2), 2)
            else: # Cerca da metade
                best_action = 0
        print('action = ' + str(best_action))
        return best_action

    # def compute_best_action(self, s):
    #     theta = put_angle_into_range(s[0])
    #     theta_dif = s[1]
    #     q_of_all_actions = np.zeros(shape=(len(self.all_actions)), dtype=np.float32)
    #     for i in range(len(self.all_actions)):
    #         a = self.all_actions[i]
    #         q_of_all_actions[i] = self.compute_q(s, a)
    #     best_action = self.all_actions[np.argmax(q_of_all_actions)]
    #     if np.random.rand() < 0.1:
    #         best_action = np.random.choice(self.all_actions)
    #     # print('Action: ' + str(best_action))
    #     return best_action

    # def compute_best_action(self, s):
    #     theta = put_angle_into_range(s[0])
    #     theta_dif = s[1]
    #     # a = -1.0 / (2.0 * self.w[5] + 1e-6) * (self.w[4] + theta * self.w[6] + theta_dif * self.w[7])
    #     # a = -1.0 / (2.0 * self.w[7] + 1e-6) * (self.w[6] + np.cos(theta) * self.w[8] + np.sin(theta) * self.w[9] + theta_dif * self.w[10])
    #     # a = (-self.w[0] * np.cos(theta) - self.w[1] * np.sin(theta)) / (2.0 * (self.w[3] * np.sin(theta) + self.w[2] * np.cos(theta)) + 1e-6)
    #     a = (-self.w[0] * np.cos(theta) - self.w[1] * np.sin(theta) - self.w[4] * np.cos(theta) * theta_dif) / (2.0 * (self.w[3] * np.sin(theta) + self.w[2] * np.cos(theta)) + 1e-6)
    #     # Apply randomness:
    #     a = a + np.random.normal(0, 0.001)
    #     if a < -self.a_max:
    #         a = -self.a_max
    #     elif a > self.a_max:
    #         a = self.a_max
    #     q_a = self.compute_q(s, a)
    #     print('a = ' + str(a) + '  Q(a) = ' + str(q_a))
    #     return a

