# pylint: disable=invalid-unary-operand-type
""" Optimization utilities"""

import heapq
import logging
import time

import numpy as np

def sa_find_maximum(init_points, trans_func, eval_func, n_iter, n_keep, temp,
                    parallel_temp=False,
                    early_stop=None, exclusive=None, verbose=False):
    """parallel simulated annealing optimization algorithm

    Parameters
    ----------
    init_points: Array of point
        initial points
    trans_func: Func(Array of point) -> Array of point
        local transition function
    eval_func: Func(Array of point) -> Array of float
        score function
    n_iter: int
        maximum iteration
    n_keep: int

    temp: float or Array of float
        if is a single float, then use a constant temperature
        if is an Array, then the perform linear cooling from temp[0] to temp[1]
    parallel_temp: bool
        whether use parallel tempering (replica exchange)

    early_stop: int, optional
        stop iteration if the optimal set do not change in early_stop rounds
    exclusive: set, optional
        exclusive points for optimal set (return value)
    verbose: bool, optional
        whether print log info

    Returns
    -------
    opt_set : Array of points
        optimal set
    """
    tic = time.time()
    early_stop = early_stop or 1e9

    scores = eval_func(init_points)
    points = np.array(init_points)

    # build heap and insert initial points
    heap_items = [(float('-inf'), -i) for i in range(n_keep)]
    heapq.heapify(heap_items)
    in_heap = set(exclusive)
    in_heap.update([-i for i in range(n_keep)])

    for s, p in zip(scores, points):
        if s > heap_items[0][0] and p not in in_heap:
            pop = heapq.heapreplace(heap_items, (s, p))
            in_heap.remove(pop[1])
            in_heap.add(p)

    k = 0
    k_last_modify = 0

    if parallel_temp:
        t = np.arange(temp[0], temp[1] - 1e-10, 1.0 * (temp[1] - temp[0]) / (len(points) - 1))
        cool = 0
    else:
        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

    while k < n_iter and k < k_last_modify + early_stop:
        new_points = np.array(trans_func(points))
        new_scores = eval_func(new_points)

        ac_prob = np.exp((new_scores - scores) / t)
        ac_index = np.random.random(len(ac_prob)) < ac_prob

        points[ac_index] = new_points[ac_index]
        scores[ac_index] = new_scores[ac_index]

        if parallel_temp:
            # use parallel tempering: exchange temperature between neighbours
            for j in range(len(points) - 1):
                k = j + 1
                ac_prob = np.exp((scores[k] - scores[j]) * (1 / t[j] - 1 / t[k]))
                if np.random.random() < ac_prob:
                    points[j], points[k] = points[k], points[j]
                    scores[j], scores[k] = scores[k], scores[j]

        for s, p in zip(new_scores, new_points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)
                k_last_modify = k

        k += 1
        t -= cool

        if verbose >= 1 and k % verbose == 0:
            if parallel_temp:
                t_str = "(%.2f, %.2f)" % (np.max(t), np.min(t))
            else:
                t_str = "%.2f" % t
            logging.info("SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                         "elapsed: %.2f",
                         k, k_last_modify, heap_items[0][0],
                         np.max([heap_items[i][0] for i in range(len(heap_items))]), t_str,
                         time.time() - tic)

    heap_items.sort(key=lambda x: -x[0])
    if verbose:
        logging.info("SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\telapsed: %.2f",
                     k, k_last_modify, heap_items[-1][0], heap_items[0][0], time.time() - tic)
        logging.info("%s", heap_items)

    return [x[1] for x in heap_items], points


def submodular_pick(scores, knobs, n_pick, knob_weight=1.0):
    """run greedy optimization to pick points with regard to both score and diversity
    DiversityScore = knob_weight * number of unique knobs in the selected set
    Obj = sum(scores[i] for i in pick) + DiversityScore
    Note that this objective function is a monotone submodular function

    scores: Array of float
        score of every points
    knobs: Array of Array of int
        feature vector (tunable knobs) of every points
    n_keep: int
        number of points to pick
    knob_weight: float
        weight of an unique knob feature
    """
    n = len(scores)
    assert n == len(knobs)
    n_knobs = len(knobs[0])

    knobs_set = [set() for _ in range(n_knobs)]

    ret = []
    remain = list(range(len(scores)))

    for _ in range(n_pick):
        max_x = -1
        max_delta = -1e9

        for x in remain:
            tmp_delta = scores[x]
            for i in range(n_knobs):
                if knobs[x][i] not in knobs_set[i]:
                    tmp_delta += knob_weight

            if tmp_delta > max_delta:
                max_delta, max_x = tmp_delta, x

        ret.append(max_x)
        remain.remove(max_x)
        for i in range(n_knobs):
            knobs_set[i].add(knobs[max_x][i])

    return ret
