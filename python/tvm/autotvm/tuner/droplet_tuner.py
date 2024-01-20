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
"""Tuner with droplet algorithm"""

import logging
import os

import numpy as np

from .tuner import Tuner

LOGGER = logging.getLogger("autotvm")


class DropletTuner(Tuner):
    """Tuner with droplet algorithm.

    Parameters
    ----------
    start_position: list of int
        position initial of the space, the default is [0, 0, ..., 0]
    pvalue: float
        statistical value to confidence level, the default is 0.05
    """

    def __init__(self, task, start_position=None, pvalue=0.05):
        super(DropletTuner, self).__init__(task)

        # space info
        self.space = task.config_space
        self.dims = []

        for _, v in self.space.space_map.items():
            self.dims.append(len(v))
        if len(self.dims) == 0:
            self.dims.append(1)

        # start position
        start_position = [0] * len(self.dims) if start_position is None else start_position
        self.best_choice = (-1, [0] * len(self.dims), [99999])
        self.visited = set([self.space.knob2point(start_position)])
        self.execution, self.total_execution, self.pvalue = 1, max(self.dims), pvalue
        self.step, self.iter, self.batch = 1, 0, max(16, os.cpu_count())
        self.next = [(self.space.knob2point(start_position), start_position)]

    def num_to_bin(self, value, factor=1):
        bin_format = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(i) * factor for i in bin_format]

    def search_space(self, factor=1):
        search_space = []
        for i in range(2 ** len(self.dims) - 1, 0, -1):
            search_space += [self.num_to_bin(i, factor)] + [self.num_to_bin(i, -factor)]
        return search_space

    def next_pos(self, new_positions):
        "returns the neighbors of the best solution"
        next_set = []
        for p in new_positions:
            if len(next_set) > self.batch:
                break
            new_p = [
                (x + y) % self.dims[i] if (x + y > 0) else 0
                for i, (x, y) in enumerate(zip(p, self.best_choice[1]))
            ]
            idx_p = self.space.knob2point(new_p)
            if idx_p not in self.visited:
                self.visited.add(idx_p)
                next_set.append((idx_p, new_p))
        return next_set

    def p_value(self, elem_1, elem_2):
        if len(elem_1) <= 1 or len(elem_2) <= 1:
            return True

        from scipy import stats  # pylint: disable=import-outside-toplevel

        return stats.ttest_ind(np.array(elem_1), np.array(elem_2)).pvalue <= self.pvalue

    def next_batch(self, batch_size):
        ret, self.batch = [], batch_size
        for i in range(batch_size):
            if i >= len(self.next):
                break
            if self.space.is_index_valid(self.next[i][0]):
                ret.append(self.space.get(self.next[i][0]))
        return ret

    def speculation(self):
        # Gradient descending direction prediction and search space filling
        while len(self.next) < self.batch and self.execution < self.total_execution:
            self.execution += self.step
            self.next += self.next_pos(self.search_space(self.execution))

    def update(self, inputs, results):
        found_best_pos, count_valids = False, 0
        for i, (_, res) in enumerate(zip(inputs, results)):
            try:
                if np.mean(self.best_choice[2]) > np.mean(res.costs) and self.p_value(
                    self.best_choice[2], res.costs
                ):
                    self.best_choice = (self.next[i][0], self.next[i][1], res.costs)
                    found_best_pos = True
                count_valids += 1
            except TypeError:
                LOGGER.debug("Solution is not valid")
                continue
            else:
                continue

        self.next = self.next[self.batch : -1]
        if found_best_pos:
            self.next += self.next_pos(self.search_space())
            self.execution = 1
        self.speculation()
        # stop, because all neighborhoods are invalid.
        if count_valids == 0 and self.iter > 3:
            self.next = []
            LOGGER.warning(
                f"Warning: early termination due to an all-invalid neighborhood \
                after {self.iter} iterations"
            )

    def has_next(self):
        return len(self.next) > 0

    def load_history(self, data_set, min_seed_records=500):
        pass
