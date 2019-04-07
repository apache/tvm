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
from .model_based_tuner import knob2point, point2knob


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

        # space info
        self.space = task.config_space
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.visited = set([])

        # current generation
        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0

        # random initialization
        self.pop_size = min(self.pop_size, len(self.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        for _ in range(self.pop_size):
            tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)
            while knob2point(tmp_gene, self.dims) in self.visited:
                tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)

            self.genes.append(tmp_gene)
            self.visited.add(knob2point(tmp_gene, self.dims))

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            gene = self.genes[self.trial_pt % self.pop_size]
            self.trial_pt += 1
            ret.append(self.space.get(knob2point(gene, self.dims)))

        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0.0)

        if len(self.scores) >= len(self.genes) and len(self.visited) < len(self.space):
            genes = self.genes + self.elites
            scores = np.array(self.scores[:len(self.genes)] + self.elite_scores)

            # reserve elite
            self.elites, self.elite_scores = [], []
            elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
            for ind in elite_indexes:
                self.elites.append(genes[ind])
                self.elite_scores.append(scores[ind])

            # cross over
            indices = np.arange(len(genes))
            scores += 1e-8
            scores /= np.max(scores)
            probs = scores / np.sum(scores)
            tmp_genes = []
            for _ in range(self.pop_size):
                p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                p1, p2 = genes[p1], genes[p2]
                point = np.random.randint(len(self.dims))
                tmp_gene = p1[:point] + p2[point:]
                tmp_genes.append(tmp_gene)

            # mutation
            next_genes = []
            for tmp_gene in tmp_genes:
                for j, dim in enumerate(self.dims):
                    if np.random.random() < self.mutation_prob:
                        tmp_gene[j] = np.random.randint(dim)

                if len(self.visited) < len(self.space):
                    while knob2point(tmp_gene, self.dims) in self.visited:
                        j = np.random.randint(len(self.dims))
                        tmp_gene[j] = np.random.randint(self.dims[j])
                    next_genes.append(tmp_gene)
                    self.visited.add(knob2point(tmp_gene, self.dims))
                else:
                    break

            self.genes = next_genes
            self.trial_pt = 0
            self.scores = []

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def load_history(self, data_set):
        pass
