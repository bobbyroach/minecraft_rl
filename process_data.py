from pathlib import Path
import pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def get_and_format_data(path_list, max_len=None, stop_at_reward=False):
    all_data = []

    for i, path in enumerate(path_list):
        action_values = np.load(path + "/rendered.npz", allow_pickle=True)
        action_values = dict(action_values)

        action_values['reward'] = (action_values['reward'] > 0).astype(float)        
        min_len = len(action_values['reward'])

        if stop_at_reward:
            for i, rewards in enumerate(action_values['reward']):
                if rewards == 1:
                    min_len = min(i + 150, min_len)
                    break
                else: min_len = 901
            if min_len > 900: continue

        action_values = {k: v[:max_len] for k, v in action_values.items()}

        frames = extract_frames(path + "/recording.mp4")[:min_len]
        if len(frames) == 0: continue
        if len(frames) < min_len: min_len = len(frames)

        aligned_data = [(frames[i], {k: v[i] for k, v in action_values.items()}) for i in range(min_len)]
        all_data.append(aligned_data)

    return all_data

def filter_actions(data, reference):
    for episode in data:
        for i in range(len(episode)):
            frame, action_dict = episode[i]

            new_action_dict = {}
            for key, val in action_dict.items():
                if key in reference:
                    new_action_dict[key] = val
                if key.replace("_", "$") in reference:
                    new_action_dict[key.replace("_", "$")] = val
            
            episode[i] = (frame, new_action_dict)

directory_path = Path("MineRLTreechop-v0/")
path_list = ["MineRLTreechop-v0/" + f.name for f in directory_path.iterdir()]
data = get_and_format_data(path_list)

directory_path = Path("MineRLObtainDiamond-v0/")
path_list = ["MineRLObtainDiamond-v0/" + f.name for f in directory_path.iterdir()]
data.extend(get_and_format_data(path_list, stop_at_reward=True))

directory_path = Path("MineRLObtainIronPickaxe-v0/")
path_list = ["MineRLObtainIronPickaxe-v0/" + f.name for f in directory_path.iterdir()]
data.extend(get_and_format_data(path_list, stop_at_reward=True))

filter_actions(data, data[0][0][1].keys())

print(f"Amount of episodes: {len(data)}")
print(f"\nEach element in each episode is of form: (frames, action-value dict)")
print(f"Shape of pov frame: {data[0][0][0].shape}")

print("\nActions:")
for key in data[0][0][1].keys():
    print("  " + key)

# Visualize an episode
episode = 32
for idx in range(38, 48):
    frame = data[episode][idx][0]
    plt.imshow(frame)
    plt.show()

with open("video_and_actions.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)