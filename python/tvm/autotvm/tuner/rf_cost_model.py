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
# pylint: disable=invalid-name
"""RandomForestRegressor+ExpectedImprovement as a cost model"""

import multiprocessing
import logging
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from .. import feature
from ..utils import get_rank
from .metric import max_curve, recall_curve, cover_curve
from .model_based_tuner import CostModel, FeatureCache

logger = logging.getLogger("autotvm")


class RFEICostModel(CostModel):
    """RandomForestRegressor+ExpectedImprovement as a cost model

    Parameters
    ----------
    task: Task
        The tuning task
    feature_type: str, optional
        If is 'itervar', use features extracted from IterVar (loop variable).
        If is 'knob', use flatten ConfigEntity directly.
        If is 'curve', use sampled curve feature (relation feature).

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' are good.
                                'itervar' is more accurate but 'knob' is much faster.
                                There are some constraints on 'itervar', if you meet
                                problems with feature extraction when using 'itervar',
                                you can switch to 'knob'.

        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability,
                               'knob' is faster.
        For cross-device or cross-operator tuning, you can use 'curve' only.
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    upper_model: RFEICostModel, optional
        The upper model used in transfer learning
    n_estimators: int, optional
        The number of estimators of the RandomForestRegressor
    random_state: int, optional
        The random state of initializing the RandomForestRegressor
    max_features: int, optional
        The max features of the RandomForestRegressor
    """

    def __init__(
        self, task, feature_type, num_threads=None, log_interval=25, upper_model=None, n_estimators=10, random_state=2, max_features=10
    ):
        self.task = task
        self.target = task.target
        self.space = task.config_space
        self.prior = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_features=max_features)
        self.fea_type = feature_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        if feature_type == "itervar":
            self.feature_extract_func = _extract_itervar_feature_index
        elif feature_type == "knob":
            self.feature_extract_func = _extract_knob_feature_index
        elif feature_type == "curve":
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + feature_type)

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.best_flops = 0.0
        self.pool = None
        self.base_model = None

        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        # Use global variable to pass common arguments. This is only used when
        # new processes are started with fork. We have to set the globals
        # before we create the pool, so that processes in the pool get the
        # correct globals.
        global _extract_space, _extract_target, _extract_task
        _extract_space = space
        _extract_target = target
        _extract_task = task
        self.pool = multiprocessing.Pool(self.num_threads)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def _base_model_discount(self):
        return 1.0 / (2 ** (self._sample_size / 64.0))

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)
        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        self.best_flops = max(ys)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6

        self._sample_size = len(x_train)
        self.prior.fit(x_train, ys)

        logger.debug(
            "RFEI train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
            time.time() - tic,
            len(xs),
            len(xs) - np.sum(valid_index),
            self.feature_cache.size(self.fea_type),
        )

    def fit_log(self, records, plan_size):
        tic = time.time()

        # filter data, only pick the data with a same task
        data = []
        for inp, res in records:
            if inp.task.name == self.task.name:
                data.append((inp, res))

        logger.debug("RFEI load %d entries from history log file", len(data))

        # extract feature
        self._reset_pool(self.space, self.target, self.task)
        pool = self._get_pool()
        if self.fea_type == "itervar":
            feature_extract_func = _extract_itervar_feature_log
        elif self.fea_type == "knob":
            feature_extract_func = _extract_knob_feature_log
        elif self.fea_type == "curve":
            feature_extract_func = _extract_curve_feature_log
        else:
            raise RuntimeError("Invalid feature type: " + self.fea_type)
        res = pool.map(feature_extract_func, data)

        # filter out feature with different shapes
        fea_len = len(self._get_feature([0])[0])

        xs, ys = [], []
        for x, y in res:
            if len(x) == fea_len:
                xs.append(x)
                ys.append(y)

        if len(xs) < 500:  # no enough samples
            return False

        xs, ys = [], []
        for x, y in res:
            if len(x) == fea_len:
                xs.append(x)
                ys.append(y)

        if len(xs) < 500:  # no enough samples
            return False

        xs, ys = np.array(xs), np.array(ys)

        self.best_flops = max(ys)
        self.prior.fit(xs, ys)

        logger.debug("RFEI train: %.2f\tobs: %d", time.time() - tic, len(xs))

        return True

    def predict(self, xs):
        predicts, _ = self._prediction_variation(xs)
        return predicts

    def _prediction_variation(self, x_to_predict):
        """Use Bayesian Optimization to predict the y and get the prediction_variation"""
        feas = self._get_feature(x_to_predict)
        preds = np.array([tree.predict(feas) for tree in self.prior]).T
        eis = []
        variances = []
        for pred in preds:
            mu = np.mean(pred)
            sigma = pred.std()
            best_flops = self.best_flops
            variances.append(sigma)
            with np.errstate(divide='ignore'):
                Z = (mu - best_flops) / sigma
                ei = (mu - best_flops) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] == max(0.0, mu-best_flops)
            eis.append(ei)
        prediction_variation = sum(variances)/len(variances)
        return np.array(eis), prediction_variation

    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return RFEICostModel(
            self.task, self.fea_type, self.num_threads, self.log_interval, self
        )

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)
        fea_cache = self.feature_cache.get(self.fea_type)
        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]
        if need_extract:
            pool = self._get_pool()
            # If we are forking, we can pass arguments in globals for better performance
            if multiprocessing.get_start_method(False) == "fork":
                feas = pool.map(self.feature_extract_func, need_extract)
            else:
                args = [(self.space.get(x), self.target, self.task) for x in need_extract]
                feas = pool.map(self.feature_extract_func, args)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret

    def __del__(self):
        self._close_pool()


# Global variables for passing arguments to extract functions.
_extract_space = None
_extract_target = None
_extract_task = None


def _extract_itervar_feature_index(args):
    """extract iteration var feature for an index in extract_space"""
    try:
        if multiprocessing.get_start_method(False) == "fork":
            config = _extract_space.get(args)
            with _extract_target:
                sch, fargs = _extract_task.instantiate(config)
        else:
            config, target, task = args
            with target:
                sch, fargs = task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, fargs, take_log=True)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return fea
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_knob_feature_index(args):
    """extract knob feature for an index in extract_space"""
    try:
        if multiprocessing.get_start_method(False) == "fork":
            config = _extract_space.get(args)
        else:
            config = args[0]
        return config.get_flatten_feature()
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        x = config.get_flatten_feature()

        if res.error_no == 0:
            with inp.target:  # necessary, for calculating flops of this task
                inp.task.instantiate(config)
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_curve_feature_index(args):
    """extract sampled curve feature for an index in extract_space"""
    try:
        if multiprocessing.get_start_method(False) == "fork":
            config = _extract_space.get(args)
            with _extract_target:
                sch, fargs = _extract_task.instantiate(config)
        else:
            config, target, task = args
            with target:
                sch, fargs = task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, fargs, sample_n=20)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return np.array(fea)
    except Exception:  # pylint: disable=broad-except
        return None


def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None








