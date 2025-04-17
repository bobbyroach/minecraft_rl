
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset

class MineRLDataset(Dataset):

    def __init__(self, device, capacity, action_manager):
        self.device = device
        self.capacity = capacity
        self.action_manager = action_manager
        self.data = np.array([None] * capacity)

    def put_data_into_dataset(self, data, camera_stacking_amount):
        """Put video and action pairs into dataset and exaggerate camera movements"""
        for episode in data:
            sample_queue = deque(maxlen=camera_stacking_amount)

            for sample_idx, sample in enumerate(episode):
                sample_queue.append(sample)

                if len(sample_queue) == camera_stacking_amount:
                    # Access the oldest sample in the queue
                    base_sample = sample_queue[0]
                    base_action = base_sample[1]['action$camera']

                    # Stack camera actions from subsequent samples
                    for offset in range(1, camera_stacking_amount):
                        next_sample = sample_queue[offset]
                        next_camera = next_sample[1]['action$camera']

                        # Accumulate camera movement into the base sample
                        base_action[0] += next_camera[0]
                        base_action[1] += next_camera[1]
                        if next_sample[1]['reward'] != 0: break

                    self.append_sample(sample_queue.popleft(), False)

            while sample_queue:
                last_episode = sample_idx == len(episode) - 1
                self.append_sample(sample_queue.popleft(), last_episode)
                    
        print(f"Total new transitions added: {len(self.data)}")

    def append_sample(self, sample, done):
        """Append a new sample and map action to id and preprocess image"""

        frame, action_dict  = sample[0], sample[1]
        reward = action_dict['reward']

        action_id = self.action_manager.get_id(action_dict)
        torch_img = torch.from_numpy(frame).permute(2, 0, 1)
        self.data.append((torch_img, action_id, reward, done))

    def sample_line(self, batch_size):
        """Take a batch of random samples"""
        ids = np.random.randint(0, len(self.data), size=batch_size)
        states, actions, rewards, dones = [], [], [], []

        for idx in ids:
            trans = self.data[idx % self.size]
            states.append(trans[0].to(self.device).float().div(255))
            actions.append(torch.tensor(trans[1], device=self.device))
            rewards.append(torch.tensor(trans[2], device=self.device))
            dones.append(torch.tensor(trans[3], device=self.device))

        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones)