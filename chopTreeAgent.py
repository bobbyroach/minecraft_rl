import copy
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
from model import ChopTreeAgentNet
from replay_buffer import ReplayBuffer


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

        self.model_path = "minecraft_tree_agent.pth"
        self.optimizer_path = "minecraft_tree_optimizer.pth"
        self.rl_model_path = "minecraft_rl_tree_agent.pth"

    def act(self, state):
        """
        Get top action given state
        """
        with torch.no_grad():
            logits = self.net(state)
            probs = F.softmax(logits, 1).detach().cpu().numpy()
            return np.argmax(probs)

    def bc_learn(self, dataset):
        """
        Sample a batch of states and actions from dataset
        """
        states, actions, _, _ = dataset.sample_batch(self.batch_size, 1)
        logits = self.net(states)

        # Cross entropy classification loss
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
        """Update target network periodically for RL training"""
        self.target_net.load_state_dict(self.net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Save a transition from the environment"""
        self.replay_buffer.push(state, action, reward, next_state, done)


    def save_bc_weights(self):
        torch.save(self.net.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        print(f"Model weights saved to {self.model_path}")

    def load_bc_weights(self):
        self.net.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.optimizer.load_state_dict(torch.load(self.optimizer_path, map_location=torch.device('cpu')))
        print(f"Model weights loaded from {self.model_path}")

    def save_rl_weights(self):
        torch.save(self.net.state_dict(), self.rl_model_path)
        print(f"Model weights saved to {self.rl_model_path}")

    def load_rl_weights(self):
        self.net.load_state_dict(torch.load(self.rl_model_path, map_location=torch.device('cpu')))
        print(f"Model weights loaded from {self.rl_model_path}")