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

class IndexBaseTuner(Tuner):
    """Base class for index based tuner
    This type of tuner determine the next batch of configs based on config indices.

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task

    range_idx: Optional[Tuple[int, int]]
        A tuple of index range that this tuner can select from
    """
    def __init__(self, task, range_idx=None):
        super(IndexBaseTuner, self).__init__(task)
        assert range_idx is None or isinstance(range_idx, tuple), \
            "range_idx must be None or (int, int)"

        self.range_length = len(self.task.config_space)
        self.index_offset = 0
        if range_idx is not None:
            assert range_idx[1] > range_idx[0], "Index range must be positive"
            assert range_idx[0] >= 0, "Start index must be positive"
            self.range_length = range_idx[1] - range_idx[0] + 1
            self.index_offset = range_idx[0]
        self.counter = 0

    def has_next(self):
        return self.counter < self.range_length

    def load_history(self, data_set):
        pass


class GridSearchTuner(IndexBaseTuner):
    """Enumerate the search space in a grid search order"""

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            if self.counter >= self.range_length:
                break
            index = self.counter + self.index_offset
            ret.append(self.task.config_space.get(index))
            self.counter = self.counter + 1
        return ret


class RandomTuner(IndexBaseTuner):
    """Enumerate the search space in a random order

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task

    range_idx: Optional[Tuple[int, int]]
        A tuple of index range to random
    """
    def __init__(self, task, range_idx=None):
        super(RandomTuner, self).__init__(task, range_idx)

        # Use a dict to mimic a range(n) list without storing rand_state[i] = i entries so that
        # we can generate non-repetitive random indices.
        self.rand_state = {}
        self.rand_max = self.range_length
        self.visited = []

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            if self.rand_max == 0:
                break

            # Random an indirect index.
            index_ = np.random.randint(self.rand_max)
            self.rand_max -= 1

            # Use the indirect index to get a direct index.
            index = self.rand_state.get(index_, index_) + self.index_offset
            ret.append(self.task.config_space.get(index))
            self.visited.append(index)

            # Update the direct index map.
            self.rand_state[index_] = self.rand_state.get(self.rand_max, self.rand_max)
            self.rand_state.pop(self.rand_max, None)
            self.counter += 1
        return ret
