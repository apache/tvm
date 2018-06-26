# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Base class for model-based tuner
This type of tuner will fit a cost model and use simulated annealing to
find optimums in space according to the cost model.
"""

import numpy as np

from .tuner import Tuner


class ModelBasedBaseTuner(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use simulated annealing to
    optimize acquisition function

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    batch_size: int
        Tuner will re-fit model per `batch_size` new measure samples

    sa_n_iter: int
        The maximum number of iterations of simulated annealing after refitting a new cost model
    sa_temp: float or list of float
        Temperature config of simulated annealing
    sa_persistent: bool
        Whether keep persistent states of SA points among different models
    sa_parallel_size: int
        Number of parallel Markov chains when doing parallel simulated annealing
    """

    def __init__(self, task, batch_size,
                 sa_n_iter, sa_temp, sa_persistent, sa_parallel_size):
        super(ModelBasedBaseTuner, self).__init__(task)

        self.batch_size = batch_size
        self.space_len = len(task.config_space)

        # space
        self.task = task
        self.target = task.target
        self.space = task.config_space
        self.dims = [len(x) for x in self.space.space_map.values()]

        # trial planning
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.y_max = 0

        # simulated annealing for optimizing acquisition function
        self.sa_n_iter = sa_n_iter
        self.sa_temp = sa_temp
        self.sa_persistent = sa_persistent
        self.sa_parallel_size = min(sa_parallel_size, len(self.space))
        self.sa_points = None
        self.sa_trans_func = None
        self.sa_eval_func = None

        self.train_ct = 0

    def next_batch(self, batch_size):
        ret = []

        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials):  # trial list is empty, choose randomly
                index = np.random.randint(len(self.space))
                while index in self.visited:
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))
            self.visited.add(index)

            counter += 1
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0)

        self._update_model()

    def _update_model(self):
        """refit a new model if the tuner collects enough new samples"""
        pass

    def has_next(self):
        return len(self.visited) < len(self.space)

    def __getstate__(self):
        state = {"visited": self.visited,
                 "trials": self.trials,
                 "trial_pt": self.trial_pt,
                 "xs": self.xs,
                 "ys": self.ys,
                 "train_ct": self.train_ct,
                 "flops_max": self.flops_max}
        return state

    def __setstate__(self, state):
        self.visited = state["visited"]
        self.trials = state["trials"]
        self.trial_pt = state["trial_pt"]
        self.xs = state["xs"]
        self.ys = state["ys"]
        self.train_ct = state["train_ct"]
        self.flops_max = state["flops_max"]

def point2knob(p, dims):
    """convert point form (single integer) to knob form (multi dimension)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob

def knob2point(knob, dims):
    """convert knob form (multi dimension) to point form (single integer)"""
    p = 0
    for j in range(len(knob)):
        p += knob[j] * int(np.prod(dims[:j]))
    return p

def random_walk(arg):
    """random walk as local transition

    Parameters
    ----------
    args[0]: int
        index of the ConfigEntity
    args[1]: Array of int
        sizes of each dimension
    """
    # transform to knob form
    p, dims = arg
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
