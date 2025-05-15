import gymnasium as gym
import numpy as np
import random

class CatMouseCheeseEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(CatMouseCheeseEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(6,), dtype=np.int32)
        self.walls = self._generate_walls()
        self.reset()
        self.last_action = None

    def _generate_walls(self):
        walls = {(i, j): False for i in range(self.grid_size) for j in range(self.grid_size)}
        static_walls = [(1, 1), (2, 2), (1, 4), (1, 5), (2, 7), (1, 8),
                        (4, 3), (5, 3), (4, 6), (5, 6), (7, 2), (8, 1),
                        (8, 4), (8, 5), (7, 7), (8, 8)]
        for wall in static_walls:
            walls[wall] = True
        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        valid_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if not self.walls[(i, j)]]
        positions = random.sample(valid_positions, 3)
        self.mouse_pos, self.cat_pos, self.cheese_pos = map(list, positions)
        self.visited_positions = set()
        self.last_distance_to_cheese = self._manhattan_distance(self.mouse_pos, self.cheese_pos)
        self.last_distance_to_cat = self._manhattan_distance(self.mouse_pos, self.cat_pos)
        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, {}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _can_move(self, pos, action):
        i, j = pos
        if action == 0 and (i == 0 or self.walls[(i-1, j)]):
            return False
        if action == 1 and (i == self.grid_size - 1 or self.walls[(i+1, j)]):
            return False
        if action == 2 and (j == 0 or self.walls[(i, j-1)]):
            return False
        if action == 3 and (j == self.grid_size - 1 or self.walls[(i, j+1)]):
            return False
        return True

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        old_mouse_pos = self.mouse_pos.copy()
        old_cat_pos = self.cat_pos.copy()

        if self._can_move(self.mouse_pos, action):
            self.mouse_pos = [self.mouse_pos[0] + moves[action][0], self.mouse_pos[1] + moves[action][1]]

        self.last_action = action

        valid_cat_actions = [a for a in moves.keys() if self._can_move(self.cat_pos, a)]
        if valid_cat_actions:
            cat_action = random.choice(valid_cat_actions)
            self.cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]

        reward = -0.1

        new_distance_to_cheese = self._manhattan_distance(self.mouse_pos, self.cheese_pos)
        distance_to_cat = self._manhattan_distance(self.mouse_pos, self.cat_pos)

        if new_distance_to_cheese < self.last_distance_to_cheese:
            reward += 10  # Premia il topo se si avvicina al formaggio
        else:
            reward -= 2  # Penalizza il topo se si allontana dal formaggio

        if distance_to_cat < self.last_distance_to_cat:
            reward -= 5  # Penalizza il topo se si avvicina al gatto
        else:
            reward += 2  # Premia il topo se si allontana dal gatto

        self.last_distance_to_cheese = new_distance_to_cheese
        self.last_distance_to_cat = distance_to_cat

        if tuple(self.mouse_pos) in self.visited_positions:
            reward -= 8  # Penalizza il topo se visita una posizione già visitata
        self.visited_positions.add(tuple(self.mouse_pos))

        if self.mouse_pos == old_mouse_pos:
            reward -= 10  # Penalizza il topo se rimane fermo

        done = False
        if self.mouse_pos == self.cat_pos or (self.mouse_pos == old_cat_pos and self.cat_pos == old_mouse_pos):
            reward -= 100  # Penalizza fortemente il topo se viene catturato dal gatto
            done = True
        elif self.mouse_pos == self.cheese_pos:
            reward += 120  # Premia fortemente il topo se raggiunge il formaggio
            done = True

        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, reward, done, False, {}

    def render(self):
        pass