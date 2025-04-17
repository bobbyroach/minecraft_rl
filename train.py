import cv2
import gym
from action_manager import ActionController
from chopTreeAgent import ChopTreeAgent
from mineRL_dataset import MineRLDataset
import pickle
import torch
from tqdm import tqdm



# Load in data
with open("video_and_actions.pkl", "rb") as f:
    data = pickle.load(f)

# Set hyperparameters
BATCH_SIZE = 32
num_epochs = 40
lr = 0.0001
hidden_size = 512
gamma = 0.99
min_replay_size=2500
buffer_capacity=500000
capacity = 2000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


action_manager = ActionController(device)
num_outputs = len(action_manager.action_to_id) + 1

dataset = MineRLDataset(device, capacity, action_manager)
dataset.insert_data(data)

agent = ChopTreeAgent(num_actions=num_outputs, image_channels=3, batch_size=BATCH_SIZE, hidden_size=hidden_size, lr=lr, gamma=gamma, device=device, action_manager=action_manager, buffer_capacity=buffer_capacity)


# ---------------------------------------------------------------------------------------------
# -----------------------------Behavioral cloning training ------------------------------------
# ---------------------------------------------------------------------------------------------

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

agent.save_bc_weights()




# --------------------------------------------------------------------------------------------
# ------------------------------ Reinforcement training --------------------------------------
# --------------------------------------------------------------------------------------------

env = gym.make('MineRLObtainDiamondShovel-v0')

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


# Seed 6: oak forest
# Seed 1: spruce forest
seeds = [6, 6, 6, 6, 6]

agent.load_rl_weights()

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


agent.save_rl_weights()

