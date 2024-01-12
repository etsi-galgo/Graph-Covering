import numpy as np


class ACTIONS:
    LEFT = 'left'
    RIGHT = 'right'
    UP = 'up'
    DOWN = 'down'
    UP_LEFT = 'up-left'
    UP_RIGHT = 'up-right'
    DOWN_LEFT = 'down-left'
    DOWN_RIGHT = 'down-right'

    def actions(self):
        return [
            self.LEFT, self.RIGHT, self.UP, self.DOWN, self.UP_LEFT, self.UP_RIGHT, self.DOWN_LEFT, self.DOWN_RIGHT
        ]


def get_random_index(high, low=0, seed=None):
    rs = np.random.RandomState(seed)
    return rs.randint(low=low, high=high)
