from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)