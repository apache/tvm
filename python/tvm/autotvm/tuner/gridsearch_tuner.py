# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=abstract-method
"""Grid search tuner and random tuner"""

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

    def load_history(self, data_set):
        pass

    def __getstate__(self):
        return {"counter": self.counter}

    def __setstate__(self, state):
        self.counter = state['counter']


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

    def load_history(self, data_set):
        pass

    def __getstate__(self):
        return {"visited": self.counter}

    def __setstate__(self, state):
        self.counter = state['visited']
