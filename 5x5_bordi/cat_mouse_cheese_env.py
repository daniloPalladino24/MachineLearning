import gymnasium as gym
import numpy as np
import random

class CatMouseCheeseEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(CatMouseCheeseEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(6,), dtype=np.int32)
        self.walls = self._generate_walls()
        self.reset()

    def _generate_walls(self):
        walls = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                walls[(i, j)] = {"top": False, "bottom": False, "left": False, "right": False}

        walls[(0, 0)]["right"] = True
        walls[(0, 1)]["left"] = True
        walls[(0, 2)]["bottom"] = True
        walls[(1, 2)]["top"] = True
        walls[(1, 0)]["bottom"] = True
        walls[(2, 0)]["top"] = True
        walls[(1, 1)]["bottom"] = True
        walls[(2, 1)]["top"] = True
        walls[(2, 1)]["right"] = True
        walls[(2, 2)]["left"] = True
        walls[(2, 2)]["right"] = True
        walls[(2, 3)]["left"] = True
        walls[(2, 3)]["top"] = True
        walls[(1, 3)]["bottom"] = True
        walls[(3, 0)]["right"] = True
        walls[(3, 1)]["left"] = True
        walls[(3, 1)]["bottom"] = True
        walls[(4, 1)]["top"] = True
        walls[(3, 3)]["bottom"] = True
        walls[(3, 4)]["bottom"] = True
        walls[(4, 3)]["top"] = True
        walls[(4, 4)]["top"] = True

        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mouse_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        while self.cat_pos == self.mouse_pos:
            self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        while self.cat_pos == self.cheese_pos or self.mouse_pos == self.cheese_pos:
            self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.visited_positions = set()
        self.last_distance = self._manhattan_distance(self.mouse_pos, self.cheese_pos)
        state = np.array([self.mouse_pos[0], self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1], self.cat_pos[0], self.cat_pos[1]], dtype=np.int32)
        return state, {}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _can_move(self, pos, action):
        i, j = pos
        if action == 0 and self.walls[(i, j)]["top"]:
            return False
        if action == 1 and self.walls[(i, j)]["bottom"]:
            return False
        if action == 2 and self.walls[(i, j)]["left"]:
            return False
        if action == 3 and self.walls[(i, j)]["right"]:
            return False
        return True

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        old_mouse_pos = self.mouse_pos.copy()
        old_cat_pos = self.cat_pos.copy()
        new_mouse_pos = [self.mouse_pos[0] + moves[action][0], self.mouse_pos[1] + moves[action][1]]

        if (0 <= new_mouse_pos[0] < self.grid_size and 0 <= new_mouse_pos[1] < self.grid_size and
                self._can_move(self.mouse_pos, action)):
            self.mouse_pos = new_mouse_pos

        revisit_penalty = -0.5 if tuple(self.mouse_pos) in self.visited_positions else 0
        self.visited_positions.add(tuple(self.mouse_pos))
        idle_penalty = -0.3 if self.mouse_pos == old_mouse_pos else 0
        current_distance = self._manhattan_distance(self.mouse_pos, self.cheese_pos)
        distance_reward = (self.last_distance - current_distance) * 0.5
        self.last_distance = current_distance

        # Generare una lista di azioni valide per il gatto
        valid_cat_actions = []
        for cat_action in range(4):
            new_cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]
            if (0 <= new_cat_pos[0] < self.grid_size and 0 <= new_cat_pos[1] < self.grid_size and
                    self._can_move(self.cat_pos, cat_action)):
                valid_cat_actions.append(cat_action)

        # Scegliere casualmente un'azione dalla lista di azioni valide
        if valid_cat_actions:
            cat_action = random.choice(valid_cat_actions)
            self.cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]

        distance_to_cat = self._manhattan_distance(self.mouse_pos, self.cat_pos)
        if distance_to_cat <= 2:
            reward = - (3 - distance_to_cat)

        reward = -0.1 + distance_reward + revisit_penalty + idle_penalty
        done = False
        
        if self.mouse_pos == self.cat_pos or (self.mouse_pos == old_cat_pos and self.cat_pos == old_mouse_pos):
            reward = -30
            done = True
        elif self.mouse_pos == self.cheese_pos:
            reward = 30
            done = True

        next_state = np.array([self.mouse_pos[0], self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1], self.cat_pos[0], self.cat_pos[1]], dtype=np.int32)
        return next_state, reward, done, False, {}