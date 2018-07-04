
import heapq
import logging
import time

import numpy as np

from ..util import sample_ints
from .model_based_tuner import ModelOptimizer

class SimulatedAnnealingOptimizer(ModelOptimizer):
    def __init__(self, task, n_iter=500, temp=(1, 0), persistent=True, parallel_size=128,
                 early_stop=30, verbose=50):
        super(ModelOptimizer, self).__init__()

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = parallel_size
        self.early_stop = early_stop
        self.verbose = verbose
        self.points = None

    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        temp, n_iter, early_stop, verbose = self.temp, self.n_iter, self.early_stop, self.verbose

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size))

        scores = model.predict(points)

        # build heap and insert initial points
        heap_items = [(float('-inf'), -i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([-i for i in range(num)])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while k < n_iter and k < k_last_modify + early_stop:
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                new_points[i] = random_walk(p, self.dims)

            new_scores = model.predict(new_points)

            ac_prob = np.exp((new_scores - scores) / t)
            ac_index = np.random.random(len(ac_prob)) < ac_prob

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool

            if verbose >= 1 and k % verbose == 0:
                t_str = "%.2f" % t
                logging.info("SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                             "elapsed: %.2f",
                             k, k_last_modify, heap_items[0][0],
                             np.max([heap_items[i][0] for i in range(len(heap_items))]), t_str,
                             time.time() - tic)

        heap_items.sort(key=lambda item: -item[0])
        if verbose:
            logging.info("SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\telapsed: %.2f",
                         k, k_last_modify, heap_items[-1][0], heap_items[0][0], time.time() - tic)
            logging.info("%s", heap_items)

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]

def random_walk(p, dims):
    """random walk as local transition

    Parameters
    ----------
    p: int
        index of the ConfigEntity
    dims: Array of int
        sizes of each dimension

    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # transform to knob form
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    old = knob
    new = list(old)

    # mutate
    while new == old:
        ii = np.random.randint(len(old))
        to = np.random.randint(dims[ii])
        new[ii] = to

    # transform to index form
    p = 0
    for j in range(len(new)):
        p += new[j] * int(np.prod(dims[:j]))
    return p
