import cv2
import gym
from action_manager import ActionManager
from mineRL_dataset import MineRLDataset
from model import ChopTreeAgentNet
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
import torch.optim as optim
import copy
import pickle
import numpy as np
import torch
from tqdm import tqdm

class ChopTreeAgent:
    def __init__(self, num_actions, image_channels, batch_size, hidden_size, lr, gamma,
                 device, action_manager, buffer_capacity):
        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.action_manager = action_manager
        self.gamma = gamma

        self.net = ChopTreeAgentNet(num_actions, image_channels, hidden_size).to(device)
        self.net.train()
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-8, weight_decay=1e-6)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state):
        """
        Get top action given state
        """
        with torch.no_grad():
            logits = self.net(state)
            probs = F.softmax(logits, 1).detach().cpu().numpy()
            return np.argmax(probs)

    def bc_learn(self, dataset):
        states, actions, _, _ = dataset.sample_line(self.batch_size, 1)
        logits = self.net(states)
        loss = F.cross_entropy(logits, actions)

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def dqn_learn(self):
        """
        Perform a single DQN training step using the dataset and a target network.
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q values
        q_values = self.net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Save a transition from the environment"""
        self.replay_buffer.push(state, action, reward, next_state, done)



with open("video_and_actions.pkl", "rb") as f:
    data = pickle.load(f)

BATCH_SIZE = 32
num_epochs = 40
lr = 0.0001
hidden_size = 512
gamma = 0.99
min_replay_size=2500
buffer_capacity=500000
capacity = 2000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_manager = ActionManager(device)
num_outputs = len(action_manager.action_to_id) + 1

dataset = MineRLDataset(device, capacity, action_manager)
dataset.put_data_into_dataset(data)

agent = ChopTreeAgent(num_actions=num_outputs, image_channels=3, batch_size=BATCH_SIZE, hidden_size=hidden_size, lr=lr, gamma=gamma, device=device, action_manager=action_manager, buffer_capacity=buffer_capacity)


# ----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Behavioral cloning training ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


model_path = "minecraft_tree_agent.pth"
optimizer_path = "minecraft_tree_optimizer.pth"

agent.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
agent.optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device('cpu')))
print(f"Model weights loaded from {model_path}")


for epoch in range(num_epochs):

    agent.net.train()
    running_loss = 0.0
    steps = 10000

    for i in tqdm(range(steps), desc=f"Epoch {epoch+1}/{num_epochs}"):

        loss = agent.bc_learn(dataset)
        running_loss += loss.item()

    epoch_loss = running_loss / steps

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")


torch.save(agent.net.state_dict(), model_path)
torch.save(agent.optimizer.state_dict(), optimizer_path)
print(f"Model weights saved to {model_path}")



# ----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Reinforcement training ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


env = gym.make('MineRLObtainDiamondShovel-v0')

ft_model_path = "minecraft_ft_tree_agent.pth"
agent.net.load_state_dict(torch.load(ft_model_path, map_location=torch.device('cpu')))
print(f"Model weights loaded from {ft_model_path}")

def preprocess_obs(obs):
    """Converts the environment observation into a 4D tensor suitable for the model"""
    frame = obs["pov"]
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).to(dtype=torch.float32).div_(255)
    return tensor

action_keys = ['forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack']

def build_action_dict(actions, env):
    """Create a full action dictionary for the MineRL environment"""
    ac = env.action_space.no_op()

    for key in action_keys:
        ac[key] = actions.get("action$" + key, 0)
    ac['camera'] = actions["action$camera"]
    return ac

def reshape_reward(obs, total_reward):
    oak_logs = obs['inventory']['oak_log']
    spruce_logs = obs['inventory']['spruce_log']
    birch_logs = obs['inventory']['birch_log']
    total_logs = oak_logs + spruce_logs + birch_logs
    return total_logs - total_reward



seeds = [6, 6, 6, 6, 6]

for epoch in range(5):
    env.seed(seeds[epoch])
    obs = env.reset()

    agent.net.train()
    running_loss = 0.0
    steps = 2500
    total_reward = 0

    for i in tqdm(range(steps), desc=f"Epoch {epoch+1}/{num_epochs}"):

        state = preprocess_obs(obs)
        output = agent.act(state)

        action = action_manager.get_action(output)
        action = build_action_dict(action, env)

        obs, reward, done, info = env.step(action)
        reward = reshape_reward(obs, total_reward)        
        total_reward += reward

        agent.store_transition(state, output, reward, preprocess_obs(obs), done)

        # Sample from buffer and learn
        if len(agent.replay_buffer.buffer) > min_replay_size and i % 4 == 0:
            loss = agent.dqn_learn()
            running_loss += loss.item()

            if i % 500 == 0:
                agent.update_target_net()
        
        env.render()

    print(total_reward)

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")


torch.save(agent.net.state_dict(), ft_model_path)
print(f"Model weights saved to {ft_model_path}")



