import cv2
import gym
import torch
import action_manager
from chopTreeAgent import ChopTreeAgent


def preprocess_obs(obs):
    """
    Converts the environment observation into a 4D tensor suitable for the model
    """
    frame = obs["pov"]
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(frame).permute(2, 0, 1).to(dtype=torch.float32).div_(255)
    return tensor.unsqueeze(0)

action_keys = ['forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack']

def build_action_dict(actions, env):
    """Create a full action dictionary for the MineRL environment"""
    ac = env.action_space.no_op()

    for key in action_keys:
        ac[key] = actions.get("action$" + key, 0)
    ac['camera'] = actions["action$camera"]
    return ac

def print_action(action_dict):
    """Print action taken in real-time"""
    display = []
    for key, val in action_dict.items():
        if key == 'action$camera':
            if val[0] < -3:
                display.append("Camera up")
            if val[0] > 3:
                display.append("Camera down")
            if val[1] < -3:
                display.append("Camera left")
            if val[1] > 3:
                display.append("Camera right")
        elif val == 1:
            display.append(key[7:])
    print(sorted(display))



BATCH_SIZE = 32
num_epochs = 40
lr = 0.0001
hidden_size = 512
gamma = 0.99
min_replay_size=2500
buffer_capacity=500000
capacity = 2000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_manager = action_manager.ActionController(device)
num_outputs = len(action_manager.action_to_id) + 1

agent = ChopTreeAgent(num_actions=num_outputs, image_channels=3, batch_size=BATCH_SIZE, hidden_size=hidden_size, lr=lr, gamma=gamma, device=device, action_manager=action_manager, buffer_capacity=buffer_capacity)


agent.net.eval()

env = gym.make('MineRLObtainDiamondShovel-v0')

env.seed(6)
obs = env.reset()

steps = 0
done = False

total_up_down = 0
consec_up_frames = 0

try:
    while not done:
        state = preprocess_obs(obs)
        output = agent.act(state)

        action = action_manager.get_action(output)
        print_action(action)

        action = build_action_dict(action, env)

        obs, reward, done, info = env.step(action)
        env.render()

        steps += 1
        if steps == 10000:
            done = True
finally:
    env.close()