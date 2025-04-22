import minerl
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torchvision import transforms

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 1000
MEMORY_CAPACITY = 50000

# Action list (discretized)
ACTIONS = [
    {'forward': 1}, {'attack': 1}, {'jump': 1},
    {'forward': 1, 'attack': 1}, {'camera': [0, 10]}, {'camera': [0, -10]},
    {'camera': [10, 0]}, {'camera': [-10, 0]}
]

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x / 255.0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

def preprocess(obs):
    frame = obs['pov']
    frame = transforms.ToPILImage()(frame).convert('L').resize((84, 84))
    frame = transforms.ToTensor()(frame)
    return frame

# Training loop (DQN only)

def train():
    env = minerl.make('MineRLNavigateDense-v0')
    q_net = DQN(len(ACTIONS)).cuda()
    target_net = DQN(len(ACTIONS)).cuda()
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_CAPACITY)

    epsilon = EPSILON_START
    state_stack = deque(maxlen=4)

    for episode in range(500):
        obs = env.reset()
        state_stack.clear()
        for _ in range(4):
            state_stack.append(preprocess(obs))
        state = torch.cat(list(state_stack)).unsqueeze(0).cuda()

        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action_idx = torch.argmax(q_values).item()

            action = ACTIONS[action_idx]
            next_obs, reward, done, _ = env.step(action)
            next_frame = preprocess(next_obs)
            state_stack.append(next_frame)
            next_state = torch.cat(list(state_stack)).unsqueeze(0).cuda()

            memory.push((state.cpu().numpy(), action_idx, reward, next_state.cpu().numpy(), done))
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                s, a, r, s_next, d = memory.sample(BATCH_SIZE)
                s = torch.tensor(s).float().cuda()
                a = torch.tensor(a).long().cuda()
                r = torch.tensor(r).float().cuda()
                s_next = torch.tensor(s_next).float().cuda()
                d = torch.tensor(d).float().cuda()

                q_val = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_q = target_net(s_next).max(1)[0]
                expected = r + GAMMA * next_q * (1 - d)

                loss = nn.functional.mse_loss(q_val, expected.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

if __name__ == '__main__':
    train()
