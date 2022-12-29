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
# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Base class for model-based tuner
This type of tuner will fit a cost model and use some optimization methods to
find optimums points of cost model in space.
"""
import gc

import numpy as np

from .tuner import Tuner
from ..env import GLOBAL_SCOPE


class FeatureCache(object):
    """Feature cache manager for cache sharing between different cost models"""

    def __init__(self):
        self.feature_cache = {}

    def get(self, key):
        """Get feature cache dictionary for a key

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        fea_cache: dict
            cache dictionary
        """
        if key not in self.feature_cache:
            self.feature_cache[key] = {}

        return self.feature_cache[key]

    def size(self, key):
        """ " Get the size of a feature cache dictionary

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        n: int
        """
        return len(self.feature_cache.get(key, tuple()))

    def clear(self, key):
        """Clear feature cache for a key

        Parameters
        ----------
        key: str
            The key of a feature type
        """
        del self.feature_cache[key]
        self.feature_cache[key] = {}
        gc.collect()


class CostModel(object):
    """Cost model to predict the speed of a config"""

    def __init__(self):
        pass

    def fit(self, xs, ys, plan_size):
        """Fit to training data

        Parameters
        ----------
        xs: Array of int
            indexes of configs in the config space
        ys: Array of float
            The speed (flop, float number operations per second)
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def fit_log(self, records, plan_size, min_seed_records=500):
        """Fit training data from log.

        Parameters
        ----------
        records: Array of Tuple(MeasureInput, MeasureResult)
            The tuning records
        plan_size: int
            The plan size of tuner
        min_seed_records: int
            Defaults to 500. Indicates the minimum number of records to
            train the tuner with. If there are less than `min_seed_records`
            number of records in `data_set`, no training of the tuner
            will be done.
        """
        raise NotImplementedError()

    def predict(self, xs, output_margin=False):
        """Predict the speed of configs

        Parameters
        ----------
        xs: Array of int
            The indexes of configs to predict
        output_margin: bool, optional
            Whether output the untransformed margin.
            When a model is used as base model, it should output untransformed margin

        Returns
        -------
        preds: Array of float
            The prediction
        """
        raise NotImplementedError()

    def load_basemodel(self, base_model):
        """Load base model for transfer learning

        Parameters
        ----------
        base_model: CostModel
                base model
        """
        raise NotImplementedError()

    def spawn_base_model(self):
        """Clone a base model with the same parameters.
        The base model is used to fit history data in transfer learning.

        Returns
        -------
        model: CostModel
            A model with the same hyperparameter (argument)
        """
        raise NotImplementedError()


class ModelOptimizer(object):
    """Optimizer used to find optimal points of cost model"""

    def __init__(self):
        pass

    def find_maximums(self, model, num, exclusive):
        """Find maximum of a cost model

        Note we use cost model to predict GFLOPS, so we should find the maximum

        Parameters
        ----------
        model: CostModel
            Cost model
        num: int
            The number of returned maximum points
        exclusive: set, optional
            The excluded set of this optimizer. Return results won't include any
            elements in this set.
        """
        raise NotImplementedError()


class ModelBasedTuner(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use an optimizer to
    find the maximums of the cost model as next trials

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    cost_model: CostModel
        The cost model that predicts the speed of a config (IR)
    model_optimizer:
        The optimizer to find local optimum points of cost model in tuning search space
    plan_size: int
        Tuner will re-fit model per `plan_size` new measure samples
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick plan_size of them according to the diversity metric.
    """

    def __init__(self, task, cost_model, model_optimizer, plan_size, diversity_filter_ratio=None):
        super(ModelBasedTuner, self).__init__(task)

        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, (
                "Diversity filter ratio " "must be larger than one"
            )

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0

    def next_batch(self, batch_size):
        ret = []
        while len(ret) < batch_size and self.has_next():
            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited and self.space.is_index_valid(index):
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = self.space.get_rand_index(to_exclude=self.visited)
            ret.append(self.space.get(index))
            self.visited.add(index)
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
                self.ys.append(0.0)
            # Usually the update function is called during the tune loop
            # after the index is already added to the visited set.
            # However, adding the index to visited again here enables us
            # to also use this update function to resume tuning progress in
            # case of interruption.
            assert self.space.is_index_valid(index)
            self.visited.add(index)
        # if we have enough new training samples
        if len(self.xs) >= self.plan_size * (self.train_ct + 1) and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size * self.diversity_filter_ratio, self.visited
                )
                scores = self.cost_model.predict(candidate)
                knobs = [self.space.point2knob(x) for x in candidate]
                pick_index = submodular_pick(0 * scores, knobs, self.plan_size, knob_weight=1)
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited
                )

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def load_history(self, data_set, min_seed_records=500):
        # set in_tuning as True to make the feature extraction consistent
        GLOBAL_SCOPE.in_tuning = True

        # fit base model
        base_model = self.cost_model.spawn_base_model()
        success = base_model.fit_log(data_set, self.plan_size, min_seed_records)

        if not success:
            GLOBAL_SCOPE.in_tuning = False
            return

        # use base model to select initial points
        if not self.trials:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
            self.trials = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)
        GLOBAL_SCOPE.in_tuning = False

    def has_next(self):
        return len(self.visited) < len(self.space)


def submodular_pick(scores, knobs, n_pick, knob_weight=1.0):
    """Run greedy optimization to pick points with regard to both score and diversity.
    DiversityScore = knob_weight * number of unique knobs in the selected set
    Obj = sum(scores[i] for i in pick) + DiversityScore
    Note that this objective function is a monotone submodular function.

    Parameters
    ----------
    scores: Array of float
        score of every points
    knobs: Array of Array of int
        feature vector (tunable knobs) of every points
    n_pick: int
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
