import gym
from gym import spaces
import numpy as np

class RiverCrossingEnv(gym.Env):
    """ Small 3x5 Gridworld with a river in the middle row."""
    def __init__(self):
        self.height = 3
        self.width = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }
        self.start = (self.height-1, 0)
        self.state = self.start

    def reset(self):
        """
        Resets the environment to the first state of the environment.
        Returns
        -------
        state : State
            First state of the episode.
        """
        self.state = self.start
        return self.start

    def step(self, action):
        """
        Performs a single step of the environment given an action.
        i.e. samples s',r from the p(s',r | s,a) distribution.
        Parameters
        ----------
        action: int
            The action performed by the agent
        Returns
        -------
        next_state : state
            State resulting from the transition
        reward : float
            Reward for this transition
        done : bool
            Whether next_state is a terminal state.
        info : dict
            ignore this
        """
        # TODO: code here
        y, x = self.state
        dy, dx = self.moves[action]
        next_x, next_y = x+dx, y+dy

        next_x = np.clip(next_x, 0, self.width-1)  # clip the values to the world
        next_y = np.clip(next_y, 0, self.height-1) # clip the values to the world

        if next_y == 1:
            rand = np.random.uniform()
            if rand < 0.2:
                next_x += 1
            elif rand < 0.7:
                next_x += 2
            else:
                next_x += 3

            next_x = np.clip(next_x, 0, self.width - 1)

        if next_x == 4 and next_y == 1:
            reward = -1
            done = True
        elif next_x == 4 and next_y == 2:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        next_state = (next_y, next_x)
        self.state = next_state

        return next_state, reward, done, {}

    def render(self):
        """
        Optional.
        Prints the environment for visualization.
        """
        grid = [[' ', ' ', ' ', ' ', '+'],
                ['R', 'R', 'R', 'R', '-'],
                [' ', ' ', ' ', ' ', ' ']]

        y, x = self.state

        grid[y][x] = 'X'
        for row in grid:
            print(row)