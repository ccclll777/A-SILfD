import random
import numpy as np
import os
import pickle
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def reset(self):
        self.__init__(capacity =self.capacity)
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return {"state_list": state, "next_state_list": next_state,
                "action_list": action, "reward_list": reward,
                "done_list": done}
        # return state, action, reward, next_state, done
    def get_all_states(self):
        state, _, _, _, _ = map(np.stack, zip(*self.buffer))
        return state
    def distill(self, ratio=0.05):
        # random distill dataset, keep at least 50_000 data points.
        data_size = max(int(ratio * len(self.buffer)), len(self.buffer)) # at least keep 50_000 data
        ind = np.random.randint(0,len(self.buffer), size=data_size)
        self.buffer[:data_size] = self.buffer[ind]
        self.position = data_size % self.capacity
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
