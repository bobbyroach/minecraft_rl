import copy
import numpy as np


class ActionManager:
    """Action wrapper with discrete camera outputs"""

    def __init__(self, device):
        self.device = device
        self.cam_magnitude = 10

        self.zero_action = {
            'action$attack': 0, 'action$forward': 0, 
            'action$sprint': 0, 'action$jump': 0, 
            'action$left': 0, 'action$right': 0, 
            'action$camera': np.array([0., 0.])
        }

        self.camera_dict = {
            'action$turn_up': np.array([-10, 0.]),
            'action$turn_down': np.array([10, 0.]),
            'action$turn_left': np.array([0., -10]),
            'action$turn_right': np.array([0., 10])
        }

        self.two_actions = [('action$forward', 'action$jump'), ('action$attack', 'action$left'), ('action$attack', 'action$right'),
                            ('action$jump', 'action$left'), ('action$jump', 'action$right'), ('action$forward', 'action$sprint'),
                            ('action$forward', 'action$left'), ('action$forward', 'action$right'), ('action$attack', 'action$turn_up'), ('action$attack', 'action$forward'), ('action$attack', 'action$turn_down'), ('action$attack', 'action$turn_left'), ('action$attack', 'action$turn_right'), ('action$forward', 'action$turn_up'), ('action$forward', 'action$turn_down'), ('action$forward', 'action$turn_left'), ('action$forward', 'action$turn_right')]
        self.three_actions = [('action$forward', 'action$sprint', 'action$jump')]

        self.remove_first = ['action$sneak', "action$sprint", 'action$back', 'action$right', 'action$left', 'action$turn_left',
                             'action$turn_right', 'action$turn_up', 'action$turn_down', 'action$jump']

        possible_actions = []
        for key, _ in self.zero_action.items():
            if key != "action$camera":
                possible_actions.append((key,))

        for key, _ in self.camera_dict.items():
            possible_actions.append((key,))

        # Add action combos
        for i in range(len(self.two_actions)):
            self.two_actions[i] = tuple(sorted(self.two_actions[i]))
        for i in range(len(self.three_actions)):
            self.three_actions[i] = tuple(sorted(self.three_actions[i]))
        possible_actions.extend(self.two_actions)
        possible_actions.extend(self.three_actions)

        # Main action dictionaries
        self.id_to_action = {}
        self.action_to_id = {}

        for i in range(1, len(possible_actions) + 1):
            self.action_to_id[possible_actions[i - 1]] = i
            self.id_to_action[i] = possible_actions[i - 1]

        self.id_to_action[0] = copy.deepcopy(self.zero_action)

    def get_action(self, id, camera_noise=True):
        """Retrieve an action by ID"""
        actions = self.id_to_action[int(id)]
        actions = actions if isinstance(actions, tuple) else [actions]
        full_actions = copy.deepcopy(self.zero_action)

        for action in actions:
            if action in self.camera_dict:
                full_actions['action$camera'] = self.camera_dict[action]
            else:
                full_actions[action] = 1

        if camera_noise: full_actions['action$camera'] += np.random.normal(0., 0.4, 2)
        return dict(full_actions)


    def get_id(self, actions):
        """Convert an action into a discrete ID."""
        actions = copy.deepcopy(actions)
        new_actions = []

        for action, val in actions.items():
            if action != 'action$camera' and action != 'reward' and val == 1:
                new_actions.append(action)

        # discretize camera action:
        camera = actions['action$camera']
        if abs(camera[0]) > 3 and abs(camera[0]) > abs(camera[1]):
            new_actions.append('action$turn_up' if camera[0] < 0 else 'action$turn_down')
        elif abs(camera[1]) > 3:
            new_actions.append('action$turn_left' if camera[1] < 0 else 'action$turn_right')

        new_actions = tuple(sorted(new_actions))

        # Remove actions until we get a valid action combo
        while new_actions not in self.action_to_id and len(new_actions) > 0:
            for remove in self.remove_first:
                if remove in new_actions:
                    new_actions = tuple(sorted(x for x in new_actions if x != remove))
                    break

        if new_actions not in self.action_to_id: return 0
        return self.action_to_id[new_actions]
