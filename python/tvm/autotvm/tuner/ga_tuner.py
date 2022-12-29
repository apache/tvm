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
# pylint: disable=consider-using-enumerate,invalid-name,abstract-method

"""Tuner with genetic algorithm"""

import numpy as np

from .tuner import Tuner


class GATuner(Tuner):
    """Tuner with genetic algorithm.
    This tuner does not have a cost model so it always run measurement on real machines.
    This tuner expands the :code:`ConfigEntity` as gene.

    Parameters
    ----------
    pop_size: int
        number of genes in one generation
    elite_num: int
        number of elite to keep
    mutation_prob: float
        probability of mutation of a knob in a gene
    """

    def __init__(self, task, pop_size=100, elite_num=3, mutation_prob=0.1):
        super(GATuner, self).__init__(task)

        # algorithm configurations
        self.pop_size = pop_size
        self.elite_num = elite_num
        self.mutation_prob = mutation_prob

        assert elite_num <= pop_size, "The number of elites must be less than population size"

        # random initialization
        self.pop_size = min(self.pop_size, len(self.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        self.visited = set(self.space.sample_ints(self.pop_size))

        # current generation
        self.genes = [self.space.point2knob(idx) for idx in self.visited]
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0

    def next_batch(self, batch_size):
        ret = []
        while len(ret) < batch_size and self.has_next():
            gene = self.genes[self.trial_pt % self.pop_size]
            self.trial_pt += 1
            ret.append(self.space.get(self.space.knob2point(gene)))
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0.0)

        if len(self.scores) >= len(self.genes) and len(self.visited) < len(self.space):
            next_genes = []
            # There is no reason to crossover or mutate since the size of the unvisited
            # is no larger than the size of the population.
            if len(self.space) - len(self.visited) <= self.pop_size:
                for idx in range(self.space.range_length):
                    if self.space.is_index_valid(idx) and idx not in self.visited:
                        next_genes.append(self.space.point2knob(idx))
                        self.visited.add(idx)
            else:
                genes = self.genes + self.elites
                scores = np.array(self.scores[: len(self.genes)] + self.elite_scores)

                # reserve elite
                self.elites, self.elite_scores = [], []
                elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num :]
                for ind in elite_indexes:
                    self.elites.append(genes[ind])
                    self.elite_scores.append(scores[ind])

                indices = np.arange(len(genes))
                scores += 1e-8
                scores /= np.max(scores)
                probs = scores / np.sum(scores)
                while len(next_genes) < self.pop_size:
                    # cross over
                    p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                    p1, p2 = genes[p1], genes[p2]
                    point = np.random.randint(len(self.space.dims))
                    tmp_gene = p1[:point] + p2[point:]
                    # mutation
                    for j, dim in enumerate(self.space.dims):
                        if np.random.random() < self.mutation_prob:
                            tmp_gene[j] = np.random.randint(dim)

                    if self.space.is_index_valid(self.space.knob2point(tmp_gene)):
                        next_genes.append(tmp_gene)
                        self.visited.add(self.space.knob2point(tmp_gene))
            self.genes = next_genes
            self.trial_pt = 0
            self.scores = []

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def load_history(self, data_set, min_seed_records=500):
        pass
