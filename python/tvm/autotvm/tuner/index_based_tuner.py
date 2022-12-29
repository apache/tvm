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

from .tuner import Tuner


class IndexBaseTuner(Tuner):
    """Base class for index based tuner
    This type of tuner determine the next batch of configs based on config indices.

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task

    range_idx: Optional[Tuple[int, int]]
        A tuple of index range that this tuner can select from [begin_idx, end_idx]
    """

    def __init__(self, task, range_idx=None):
        super(IndexBaseTuner, self).__init__(task)
        assert range_idx is None or isinstance(
            range_idx, tuple
        ), "range_idx must be None or (int, int)"

        self.visited = []
        self.begin_idx, self.end_idx = range_idx or (0, self.space.range_length - 1)
        assert self.begin_idx >= 0, "Start index must be positive"
        self.end_idx += 1  # Further end_idx is exclusive
        assert (
            self.end_idx <= self.space.range_length
        ), "Finish index must be less the space range length "
        self.range_length = self.end_idx - self.begin_idx
        assert self.range_length > 0, "Index range must be positive"
        self.visited_max = self.space.subrange_length(self.begin_idx, self.end_idx)

    def has_next(self):
        return len(self.visited) < self.visited_max

    def load_history(self, data_set, min_seed_records=500):
        pass


class GridSearchTuner(IndexBaseTuner):
    """Enumerate the search space in a grid search order"""

    def __init__(self, task, range_idx=None):
        super(GridSearchTuner, self).__init__(task, range_idx)

        self.index = self.begin_idx
        if not self.space.is_index_valid(self.index):
            self.index = self.space.get_next_index(
                self.index, start=self.begin_idx, end=self.end_idx
            )

    def next_batch(self, batch_size):
        ret = []
        while len(ret) < batch_size and self.has_next():
            self.visited.append(self.index)
            ret.append(self.space.get(self.index))
            self.index = self.space.get_next_index(
                self.index, start=self.begin_idx, end=self.end_idx
            )
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

    def next_batch(self, batch_size):
        ret = []
        while len(ret) < batch_size and self.has_next():
            index = self.space.get_rand_index(self.begin_idx, self.end_idx, to_exclude=self.visited)
            self.visited.append(index)
            ret.append(self.space.get(index))
        return ret
