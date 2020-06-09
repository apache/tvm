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

"""Cost model based on xgboost"""
from typing import List
import multiprocessing
import logging
import time
from collections import defaultdict

import numpy as np
import xgboost as xgb

from ...autotvm.tuner.xgboost_cost_model import get_rank, recall_curve, max_curve
from .cost_model import PythonBasedModel
from ..feature import get_per_stmt_features_from_measure_pairs, get_per_stmt_features_from_states
from ..serialization import LogReader

logger = logging.getLogger('ansor')

class XGBDMatrixContext:
    """Context to hold additional attributes of xgb.DMatrix"""
    def __init__(self):
        self.context_dict = defaultdict(dict)

    def get(self, key, matrix, default=None):
        return self.context_dict[key].get(matrix.handle.value, default)

    def put(self, key, matrix, value):
        self.context_dict[key][matrix.handle.value] = value

dmatrix_context = XGBDMatrixContext()

class XGBModel(PythonBasedModel):
    """Train a XGBoost model to predict the runtime cost of a program.
    The cost of a program = the sum of the costs of all stages in this program.
    i.e. Cost(p) = cost_s0 + cost_s1 + ... + cost_sn, where cost_si is the cost of Stage i

    The xgboost model makes prediction per stage, then we sum them up.
    The final predction made by this class is normalized throughtput (from 0 to 1, larger is better)

    To support this stage decomposition, we have to implement a custom loss function for
    XGBoost, which is the `pack_sum` in the code below.
    """
    def __init__(self, verbose_eval=25, num_warmup_sample=100, seed=None):
        self.xgb_params = {
            'max_depth': 10,
            'gamma': 0.001,
            'min_child_weight': 0,
            'eta': 0.2,
            # todo(lmzheng): automatically decrease learning rate when the loss is too large

            'n_gpus': 0,
            'n_threads': multiprocessing.cpu_count() / 2,
            'silent': 0,
            'seed': seed or 43,
            'disable_default_eval_metric': 1
        }
        self.bst = None
        self.plan_size = 32
        self.num_warmup_sample = num_warmup_sample
        self.verbose_eval = verbose_eval

        super().__init__()

        # measurement input/result pairs
        self.inputs = []
        self.results = []
        self.inputs_feature_cache = []

    def update(self, inputs, results):
        if len(inputs) <= 0:
            return

        self.inputs.extend(inputs)
        self.results.extend(results)

        # extract feature
        n_cached = len(self.inputs_feature_cache)
        features, normalized_throughputs, task_ids = \
            get_per_stmt_features_from_measure_pairs(self.inputs, self.results,
                                                     skip_first_n_feature_extraction=n_cached)
        if n_cached > 0:
            features = list(features)
            features[:n_cached] = self.inputs_feature_cache
            features = np.array(features)
        self.inputs_feature_cache = features
        dtrain = pack_sum_xgbmatrix(features, normalized_throughputs,
                                    task_ids, normalized_throughputs)

        # train xgb model
        self.bst = xgb.train(self.xgb_params, dtrain,
                             num_boost_round=10000,
                             obj=pack_sum_square_error,
                             callbacks=[custom_callback(
                                 stopping_rounds=50,
                                 metric='tr-p-rmse',
                                 fevals=[
                                     pack_sum_rmse, pack_sum_average_peak_score(self.plan_size),
                                 ],
                                 evals=[(dtrain, 'tr')],
                                 maximize=False,
                                 verbose_eval=self.verbose_eval)])

    def predict(self, task, states):
        features = get_per_stmt_features_from_states(states, task)
        if self.bst is not None and len(self.inputs) > self.num_warmup_sample:
            dtest, pack_ids = pack_sum_xgbmatrix_for_prediction(features)
            raw_preds = self.bst.predict(dtest)
            ret = pack_sum_predict_throughput(raw_preds, pack_ids)
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret

    def predict_stages(self, task, states):
        # Format: (s0 score, ..., sN score, s0 n_stage, s0 stage 0, ..., s1 n_stage, s1 stage 0,)
        features = get_per_stmt_features_from_states(states, task)
        if self.bst is not None and len(self.inputs) > self.num_warmup_sample:
            dtest, pack_ids = pack_sum_xgbmatrix_for_prediction(features)
            raw_preds = self.bst.predict(dtest)
            breakdown = pack_sum_predict_throughput(raw_preds, pack_ids)
            stage_scores = [[] for _ in range(len(states))]
            for pred, pack_id in zip(raw_preds, pack_ids):
                stage_scores[pack_id].append(pred)
            for idx, stage_score in enumerate(stage_scores):
                breakdown = np.append(breakdown, len(stage_score))
                breakdown = np.concatenate((breakdown, -np.array(stage_score)))
        else:
            breakdown = np.concatenate(
                (np.random.uniform(0, 1, (len(states), )), np.zeros(len(states), )))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                breakdown[idx] = float('-inf')

        return breakdown

    def load_log_file(self, file_name, n_lines=-1):
        inputs, results = LogReader(file_name).read_lines(n_lines)
        logger.info("XGBModel: Loaded %s lines of history log from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        self.bst.save_model(file_name)

    def load(self, file_name: str):
        if self.bst is None:
            self.bst = xgb.Booster(self.xgb_params)
        self.bst.load_model(file_name)
        self.num_warmup_sample = -1


def pack_sum_xgbmatrix_for_prediction(xs):
    x_flatten = []
    pack_ids = []

    for ct, x in enumerate(xs):
        for row in x:
            x_flatten.append(row)
            pack_ids.append(ct)

    return xgb.DMatrix(x_flatten), pack_ids


def pack_sum_xgbmatrix(xs, ys, gids=None, weights=None):
    if gids is not None:
        # sort by group
        indices = gids.argsort()
        xs, ys = xs[indices], ys[indices]
        group_sizes = np.bincount(gids)
        if weights is not None:
            weights = weights[indices]
    else:
        # assume it has only one group
        group_sizes = [len(xs)]

    x_flatten = []
    y_flatten = []
    weights_flatten = []
    pack_ids = []

    if weights is not None:
        for ct, (x, y, w) in enumerate(zip(xs, ys, weights)):
            for row in x:
                x_flatten.append(row)
                y_flatten.append(y)
                weights_flatten.append(w)
                pack_ids.append(ct)
    else:
        for ct, (x, y) in enumerate(zip(xs, ys)):
            for row in x:
                x_flatten.append(row)
                y_flatten.append(y)
                pack_ids.append(ct)

    ret = xgb.DMatrix(x_flatten, y_flatten)
    if weights is not None:
        ret.set_weight(weights_flatten)
    dmatrix_context.put('pack_ids', ret, np.array(pack_ids))
    dmatrix_context.put('group_sizes', ret, group_sizes)
    return ret

LOSS_TYPE = 3

# Type 0
# The model predicts cost. Use square error of throughput as loss
# loss = 1/2 * (1 / sum(x_i) - y) ^ 2
#
# Type 1
# The model predicts cost. Use square error of cost as loss
# loss = 1/2 * (sum(x_i) - 1 / y) ^ 2
#
# Type 2
# The model predicts throughput. Use square error of throughput as loss.
# loss = 1/2 * (1 / sum(1 / x_i) - y) ^ 2
#
# Type 3
# The model predicts throughput. Use square error of throughput as loss.
# But approximate 1 / (1 / a_1 + 1 / a_2 + ... + 1 / a_n) with -(b_1 + b_2 + b_3)
# loss = 1/2 * (-sum(x_i) - y) ^ 2
#
# Type 4
# The model predicts throughput. Use square error of throughput as loss.
# But approximate 1 / (1 / a_1 + 1 / a_2 + ... + 1 / a_n) with -(b_1 + b_2 + b_3)
# Also add a sigmoid to force the prediction to be within the range of (0, 1)
# loss = 1/2 * (sigmoid(-sum(x_i)) - y) ^ 2
#

def pack_sum_predict_throughput(raw_preds, pack_ids):
    if LOSS_TYPE == 0:
        sum_pred = np.bincount(pack_ids, weights=raw_preds)
        return 1 / sum_pred
    elif LOSS_TYPE == 1:
        sum_pred = np.bincount(pack_ids, weights=raw_preds)
        return 1 / sum_pred
    elif LOSS_TYPE == 2:
        sum_inverse_preds = np.bincount(pack_ids, weights=1 / raw_preds)
        return 1 / sum_inverse_preds
    elif LOSS_TYPE == 3:
        sum_pred = np.bincount(pack_ids, weights=raw_preds)
        return - sum_pred # pylint: disable=invalid-unary-operand-type
    elif LOSS_TYPE == 4:
        sum_pred = np.bincount(pack_ids, weights=raw_preds)
        return 1 / (1 + np.exp(sum_pred))
    else:
        raise ValueError("Invalid loss type: " + LOSS_TYPE)

def pack_sum_square_error(preds, dtrain):
    pack_ids = dmatrix_context.get("pack_ids", dtrain)
    weight = dtrain.get_weight()

    if LOSS_TYPE == 0:
        sum_pred = np.bincount(pack_ids, weights=preds)
        x = sum_pred[pack_ids]
        y = dtrain.get_label()
        gradient = (x * y - 1) / np.power(x, 3)
        hessian = (3 - 2 * x * y) / np.power(x, 4)
    elif LOSS_TYPE == 1:
        sum_pred = np.bincount(pack_ids, weights=preds)
        x = sum_pred[pack_ids]
        y = dtrain.get_label()
        gradient = x - 1 / np.minimum(y, 1e6)
        hessian = np.ones_like(gradient)
    elif LOSS_TYPE == 2:
        sum_inverse_preds = np.bincount(pack_ids, weights=1 / preds)[pack_ids]
        y = dtrain.get_label()
        gradient = (1 / sum_inverse_preds - y) / (np.power(preds * sum_inverse_preds, 2))
        hessian = (2 * preds * y * np.power(sum_inverse_preds, 2) - 2 * y * sum_inverse_preds - 2 * preds * sum_inverse_preds + 3) / (np.power(preds * sum_inverse_preds, 4))
    elif LOSS_TYPE == 3:
        sum_pred = np.bincount(pack_ids, weights=preds)
        x = sum_pred[pack_ids]
        y = dtrain.get_label()
        gradient = x + y
        hessian = np.ones_like(gradient)
    elif LOSS_TYPE == 4:
        sum_pred = np.bincount(pack_ids, weights=preds)
        exp_x = np.exp(sum_pred[pack_ids])
        exp_2x = np.power(exp_x, 2)
        y = dtrain.get_label()
        gradient = exp_x * (exp_x * y + y - 1) / np.power(exp_x + 1, 3)
        hessian = exp_x * (-exp_2x * y + 2 * exp_x + y - 1) / np.power(exp_x + 1, 4)
    else:
        raise ValueError("Invalid loss type: " + LOSS_TYPE)

    if len(weight) == 0:
        return gradient, hessian
    else:
        return gradient * weight, hessian * weight

def pack_sum_rmse(raw_preds, dtrain):
    pack_ids = dmatrix_context.get("pack_ids", dtrain)
    preds = pack_sum_predict_throughput(raw_preds, pack_ids)[pack_ids]
    return 'p-rmse', np.sqrt(np.mean(np.square((preds - dtrain.get_label()))))

def pack_sum_average_peak_score(N):
    """Evaluate pack sum average peak score for xgb"""

    def feval(preds, labels):
        group_sizes = dmatrix_context.get('group_sizes', labels, [len(preds)])
        pack_ids = dmatrix_context.get("pack_ids", labels)

        preds = pack_sum_predict_throughput(preds, pack_ids)
        labels = (np.bincount(pack_ids, weights=labels.get_label())
                  / np.unique(pack_ids, return_counts=True)[1])

        scores = []
        offset = 0
        for size in group_sizes:
            preds_group = preds[offset:offset + size]
            labels_group = labels[offset:offset + size]
            offset += size

            trials = np.argsort(preds_group)[::-1][:N]
            trial_scores = labels_group[trials]
            curve = max_curve(trial_scores) / np.max(labels_group)
            scores.append(np.mean(curve))
        return "a-peak@%d" % N, np.mean(scores)
    return feval

def pack_sum_average_recall_score(N):
    """Evaluate average recall score for xgb"""

    def feval(preds, labels):
        group_sizes = dmatrix_context.get('group_sizes', labels, [len(preds)])
        pack_ids = dmatrix_context.get("pack_ids", labels)

        preds = pack_sum_predict_throughput(preds, pack_ids)
        labels = (np.bincount(pack_ids, weights=labels.get_label())
                  / np.unique(pack_ids, return_counts=True)[1])

        scores = []
        offset = 0
        for size in group_sizes:
            preds_group = preds[offset:offset + size]
            labels_group = labels[offset:offset + size]
            offset += size

            trials = np.argsort(preds_group)[::-1]
            ranks = get_rank(labels_group[trials])[:N]
            curve = recall_curve(ranks)
            scores.append(np.mean(curve))
        return "a-recall@%d" % N, np.mean(scores)
    return feval


def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    maximize=False, verbose_eval=True, skip_every=2):
    """Callback function for xgboost to support multiple custom evaluation functions"""
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric
    from xgboost.training import aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state['maximize_score'] = maximize
        state['best_iteration'] = 0
        if maximize:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        if i % skip_every == 1:
            return

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(':') for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            infos = ["XGB iter: %3d" % i]
            for item in eval_res:
                if 'null' in item[0]:
                    continue
                infos.append("%s: %.6f" % (item[0], item[1]))

            logger.debug("\t".join(infos))
            if log_file:
                with open(log_file, "a") as fout:
                    fout.write("\t".join(infos) + '\n')

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d] %s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in eval_res]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback
