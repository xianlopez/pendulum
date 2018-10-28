import numpy as np
import cv2



class replay_memory:
    def __init__(self, replay_memory_capacity, batch_size, img_width, img_height, agent_history_length):
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.agent_history_length = agent_history_length

        self.D_screens = np.zeros(shape=(self.replay_memory_capacity, self.img_width, self.img_height), dtype=np.uint8)  # Preprocessed screens
        self.D_actions = np.zeros(shape=(self.replay_memory_capacity), dtype=np.int32)
        self.D_rewards = np.zeros(shape=(self.replay_memory_capacity), dtype=np.float32)
        self.D_terminals = np.zeros(shape=(self.replay_memory_capacity), dtype=np.bool)

        self.insert_position = 0
        self.memory_count = 0

    def is_ready(self):
        return self.memory_count == self.replay_memory_capacity

    def add_to_memory(self, screen_raw, action, reward, terminal):
        screen_prep = self.preprocessing(screen_raw)
        # print('inserting at position ' + str(self.insert_position))
        # if self.insert_position > 195:
        #     self.plot_screen(screen_prep)
        #     cv2.waitKey(0)
        self.D_screens[self.insert_position, :] = screen_prep
        self.D_actions[self.insert_position] = action
        self.D_rewards[self.insert_position] = reward
        self.D_terminals[self.insert_position] = terminal
        self.insert_position += 1
        if self.insert_position == self.replay_memory_capacity:
            self.insert_position = 0
        if self.memory_count < self.replay_memory_capacity:
            self.memory_count += 1
            if self.memory_count == self.replay_memory_capacity:
                print('Ready to train!')

    def build_state_with_new_screen(self, new_screen_raw):
        state = np.zeros(shape=(self.img_width, self.img_height, self.agent_history_length), dtype=np.float32)
        new_screen_prep = self.preprocessing(new_screen_raw)
        state[:, :, 0] = new_screen_prep.astype(np.float32)
        for i in range(1, self.agent_history_length):
            memory_pos = self.insert_position - i
            if memory_pos < 0:
                memory_pos += self.replay_memory_capacity
            if self.D_terminals[memory_pos]:
                break
            else:
                # TODO: preprocess?
                state[:, :, i] = self.D_screens[memory_pos, :, :].astype(np.float32)
        return state

    def build_state_from_screens(self, index):
        state = np.zeros(shape=(self.img_width, self.img_height, self.agent_history_length), dtype=np.float32)
        for i in range(self.agent_history_length):
            memory_pos = index - i
            if memory_pos < 0:
                memory_pos += self.replay_memory_capacity
            if self.D_terminals[memory_pos]:
                break
            else:
                state[:, :, i] = self.D_screens[memory_pos, :, :].astype(np.float32)
        return state

    def preprocessing(self, screen_raw):
        # The original images from the Atari environment 210x160x3 images.
        # We convert to grayscale, and rescale to 84x84.
        screen_gray = np.mean(screen_raw, axis=-1)
        screen_prep = cv2.resize(screen_gray, (self.img_width, self.img_height)).astype(np.uint8)
        return screen_prep

    def plot_state(self, state, name='state'):
        # state: (img_width, img_height, agent_history_length)
        margin = 5 # Margin, in pixels, between the different screens of the state.
        nrows = int(np.round(np.sqrt(self.agent_history_length)))
        ncols = self.agent_history_length // nrows
        if nrows * ncols < self.agent_history_length:
            ncols += 1
        mosaic_width = ncols * self.img_width + (ncols - 1) * margin
        mosaic_height = nrows * self.img_height + (nrows - 1) * margin
        mosaic = np.zeros(shape=(mosaic_width, mosaic_height), dtype=np.uint8)
        for i in range(nrows):
            for j in range(ncols):
                screen_pos = i * ncols + j
                if screen_pos < self.agent_history_length:
                    col_ini = j * (self.img_width + margin)
                    col_end = j * col_ini + self.img_width
                    row_ini = i * (self.img_height + margin)
                    row_end = i * row_ini + self.img_height
                    # mosaic[col_ini:col_end, row_ini:row_end] = state[:, :, screen_pos].astype(np.uint8)
                    mosaic[row_ini:row_end, col_ini:col_end] = state[:, :, screen_pos].astype(np.uint8)
        mosaic = cv2.resize(mosaic, (0, 0), fx=3, fy=3)
        cv2.imshow(name, mosaic)

    def plot_screen(self, screen, name='screen'):
        # screen: (img_width, img_height9
        screen_big = cv2.resize(screen, (0, 0), fx=3, fy=3)
        cv2.imshow(name, screen_big)

    def sample_batch(self):
        indices = []
        batch_states_t = np.zeros(shape=(self.batch_size, self.img_width, self.img_height, self.agent_history_length), dtype=np.float32)
        batch_actions_t = np.zeros(shape=(self.batch_size), dtype=np.int32)
        batch_rewards_t = np.zeros(shape=(self.batch_size), dtype=np.float32)
        batch_states_tp1 = np.zeros(shape=(self.batch_size, self.img_width, self.img_height, self.agent_history_length), dtype=np.float32)
        batch_terminals = np.zeros(shape=(self.batch_size), dtype=np.bool)
        for batch_idx in range(self.batch_size):
            index = np.random.randint(self.replay_memory_capacity)
            while index in indices or index == self.insert_position - 1 or self.D_terminals[index]:
                index = np.random.randint(self.replay_memory_capacity)
            indices.append(index)
            if index == self.replay_memory_capacity - 1:
                next_position = 0
            else:
                next_position = index + 1
            batch_states_t[batch_idx, :, :, :] = self.build_state_from_screens(index)
            batch_actions_t[batch_idx] = self.D_actions[index]
            batch_rewards_t[batch_idx] = self.D_rewards[index]
            batch_states_tp1[batch_idx, :, :, :] = self.build_state_from_screens(next_position)
            # batch_terminals[batch_idx] = self.D_terminals[index]
            batch_terminals[batch_idx] = self.D_terminals[next_position]
            # print('state t')
            # self.plot_state(batch_states_t[batch_idx, :, :, :], 'state_t')
            # print('state tp1')
            # name_tp1 = 'state_tp1'
            # if batch_terminals[batch_idx]:
            #     name_tp1 += '-T'
            # self.plot_state(batch_states_tp1[batch_idx, :, :, :], name_tp1)
            # cv2.waitKey(0)
        return batch_states_t, batch_actions_t, batch_rewards_t, batch_states_tp1, batch_terminals







