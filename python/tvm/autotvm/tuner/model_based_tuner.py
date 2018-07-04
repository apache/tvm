# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Base class for model-based tuner
This type of tuner will fit a cost model and use simulated annealing to
find optimums in space according to the cost model.
"""

import numpy as np

from .tuner import Tuner
from .submodular import submodular_pick


class CostModel(object):
    def __init__(self):
        pass

    def fit(self, xs, ys, plan_size):
        raise NotImplementedError()

    def fit_log(self, records, plan_size):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()

    def set_feature_cache(self, feature_cache):
        pass

    def load_basemodel(self, base_model):
        raise NotImplementedError()

    def clone_new(self):
        raise NotImplementedError()


class ModelOptimizer(object):
    def __init__(self):
        pass

    def find_maximums(self, model, num, exclusive):
        raise NotImplementedError()


class ModelBasedTuner(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use an optimizer to
    find the maximums of the cost model as next trials

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    plan_size: int
        Tuner will re-fit model per `batch_size` new measure samples
    cost_model:
    model_optimizer:

    """

    def __init__(self, task, cost_model, model_optimizer, plan_size, diversity_filter_ratio=None):
        super(ModelBasedTuner, self).__init__(task)

        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size
        self.space = task.config_space
        self.space_len = len(task.config_space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0

        # feature cache (it can be reused by multiple cost models)
        self.fea_cache = {}
        self.cost_model.set_feature_cache(self.fea_cache)

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

        # if we have enough new training samples
        if len(self.xs) >= self.plan_size * (self.train_ct + 1) \
                and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            maximums = self.model_optimizer.find_maximums(self.cost_model, self.plan_size, self.visited)

            if self.diversity_filter_ratio:
                assert self.diversity_filter_ratio >= 1, "Diversity ratio must be larger than one"
                scores = self.cost_model.predict(maximums)
                knobs = [point2knob(x, self.dims) for x in maximums]
                pick_index = submodular_pick(0 * scores, knobs, self.plan_size, knob_weight=1)
                maximums = np.array(maximums)[pick_index]

            self.trials = maximums
            self.trial_pt = 0

    def load_history(self, data_set):
        base_model = self.cost_model.clone_new()
        base_model.fit_log(data_set, self.plan_size)

        if not self.trials:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(self.cost_model, self.visited)
            self.trials = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)

    def has_next(self):
        return len(self.visited) < len(self.space)

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
