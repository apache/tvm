# pylint: disable=abstract-method
"""Grid search tuner and random tuner"""
import pickle

import numpy as np

from .tuner import Tuner


class GridSearchTuner(Tuner):
    """Enumerate the search space in a grid search order"""
    def __init__(self, task):
        super(GridSearchTuner, self).__init__(task)
        self.counter = 0

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            if self.counter >= len(self.task.config_space):
                continue
            index = self.counter
            ret.append(self.task.config_space.get(index))
            self.counter = self.counter + 1
        return ret

    def has_next(self):
        return self.counter < len(self.task.config_space)

    def save_state(self, filename):
        with open(filename, "wb") as fout:
            pickle.dump(self.counter, fout)

    def load_state(self, filename):
        with open(filename, "rb") as fin:
            self.counter = pickle.load(fin)


class RandomTuner(Tuner):
    """Enumerate the search space in a random order"""
    def __init__(self, task):
        super(RandomTuner, self).__init__(task)
        self.visited = set()

    def next_batch(self, batch_size):
        ret = []
        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.task.config_space):
                break
            index = np.random.randint(len(self.task.config_space))
            while index in self.visited:
                index = np.random.randint(len(self.task.config_space))

            ret.append(self.task.config_space.get(index))
            self.visited.add(index)
            counter += 1
        return ret

    def has_next(self):
        return len(self.visited) < len(self.task.config_space)

    def save_state(self, filename):
        with open(filename, "wb") as fout:
            pickle.dump(self.visited, fout)

    def load_state(self, filename):
        with open(filename, "rb") as fin:
            self.visited = pickle.load(fin)
