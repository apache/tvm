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
    def __init__(self, task, pop_size, elite_num=3, mutation_prob=0.1):
        super(GATuner, self).__init__(task)

        # algorithm configurations
        self.pop_size = pop_size
        self.elite_num = elite_num
        self.mutation_prob = mutation_prob

        assert elite_num <= pop_size, "number of elite must be less than population size"

        # space info
        self.space = task.config_space
        self.n_subspace = len(task.config_space.space_map)
        self.dim_subspaces = [len(x) for x in task.config_space.space_map.values()] + [1]

        self.visited = set([])

        # current generation
        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0

        # random initialization
        for _ in range(self.pop_size):
            tmp_gene = None
            while (tmp_gene is None or self._gene2index(tmp_gene) in self.visited
                   and len(self.visited) < len(self.space)):
                tmp_gene = []
                for j in range(self.n_subspace):
                    tmp_gene.append(np.random.randint(self.dim_subspaces[j]))
            self.genes.append(tmp_gene)
            self.visited.add(self._gene2index(tmp_gene))

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            gene = self.genes[self.trial_pt % self.pop_size]
            self.trial_pt += 1
            ret.append(self.space.get(self._gene2index(gene)))

        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0)

        if len(self.scores) >= len(self.genes):
            genes = self.genes + self.elites
            scores = np.array(self.scores[:len(self.genes)] + self.elite_scores)
            assert len(genes) == len(scores), "%d vs %d" % (len(genes), len(scores))
            indices = np.arange(len(genes))

            # reserve elite
            self.elites, self.elite_scores = [], []
            elites = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
            for ind in elites:
                self.elites.append(genes[ind])
                self.elite_scores.append(scores[ind])

            # cross over
            scores /= np.max(scores)
            probs = scores / np.sum(scores)
            tmp_genes = []
            for _ in range(self.pop_size):
                p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                p1, p2 = genes[p1], genes[p2]
                point = np.random.randint(self.n_subspace)
                tmp_gene = p1[:point] + p2[point:]
                tmp_genes.append(tmp_gene)

            # mutation
            next_genes = []
            for tmp_gene in tmp_genes:
                for j in range(self.n_subspace):
                    if np.random.random() < self.mutation_prob:
                        tmp_gene[j] = np.random.randint(self.dim_subspaces[j])

                if len(self.visited) < len(self.space):
                    while self._gene2index(tmp_gene) in self.visited:
                        j = np.random.randint(self.n_subspace)
                        tmp_gene[j] = np.random.randint(self.dim_subspaces[j])
                    next_genes.append(tmp_gene)
                    self.visited.add(self._gene2index(tmp_genes))
                else:
                    break

            self.genes = next_genes
            self.trial_pt = 0

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def _gene2index(self, gene):
        ind = 0
        for j in range(self.n_subspace):
            ind = (ind + gene[j]) * self.dim_subspaces[j+1]
        return ind
