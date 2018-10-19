import numpy as np
import tools

class handmade_agent:
    def __init__(self, a_max, n_actions):
        self.a_max = a_max
        self.all_actions = np.linspace(-a_max, a_max, n_actions)

    def compute_best_action(self, s):
        theta = tools.put_angle_into_range(s[0])
        # print('angle = ' + str(theta))
        theta_dif = s[1]
        if np.cos(theta) > 0:  # metade inferior
            if np.abs(theta_dif) > 14:
                best_action = 0
            else:
                best_action = np.sign(theta_dif) * 2
        else:  # metade superior
            if np.cos(theta) < -0.5:  # Moi cerca do tope
                if theta_dif * np.sign(np.sin(theta)) > 2:
                    best_action = -np.sign(theta_dif) * 2
                else:
                    best_action = min(max(np.sin(theta) * 10, -2), 2)
            else:  # Cerca da metade
                best_action = 0
        print('Action = ' + str(best_action))
        return best_action

