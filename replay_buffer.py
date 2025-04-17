from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample from batches"""
        reward_idxs = [i for i, (_, _, r, _, _) in enumerate(self.buffer) if r != 0]
        
        # Prioritize transitions with rewards with 1/100 chance
        if len(reward_idxs) > 0 and random.random() < 0.01:
            non_reward_idxs = [i for i in range(len(self.buffer)) if i not in reward_idxs]
            chosen_idxs = (random.sample(reward_idxs, 1)+ random.sample(non_reward_idxs, batch_size - 1))
            random.shuffle(chosen_idxs)
            batch = [self.buffer[i] for i in chosen_idxs]
        else:
            batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states), torch.tensor(actions, dtype=torch.long), torch.tensor(rewards), 
            torch.stack(next_states), torch.tensor(dones)
        )

    def __len__(self):
        return len(self.buffer)