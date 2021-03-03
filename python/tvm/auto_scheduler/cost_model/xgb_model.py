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

"""Cost model based on xgboost"""
import multiprocessing
import logging
from collections import defaultdict

import numpy as np

from tvm.autotvm.tuner.metric import max_curve
from .cost_model import PythonBasedModel
from ..feature import get_per_store_features_from_measure_pairs, get_per_store_features_from_states
from ..measure_record import RecordReader

xgb = None

logger = logging.getLogger("auto_scheduler")


class XGBDMatrixContext:
    """A global context to hold additional attributes of xgb.DMatrix"""

    def __init__(self):
        self.context_dict = defaultdict(dict)

    def get(self, key, matrix, default=None):
        """
        Get an attribute of a xgb.DMatrix
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: xgb.DMatrix
            The matrix
        default: Optional[Any]
            The default value if the item does not exist
        """
        return self.context_dict[key].get(matrix.handle.value, default)

    def set(self, key, matrix, value):
        """
        Set an attribute for a xgb.DMatrix
        Parameters
        ----------
        key: str
            The name of the attribute
        matrix: xgb.DMatrix
            The matrix
        value: Optional[Any]
            The new value
        """
        self.context_dict[key][matrix.handle.value] = value


dmatrix_context = XGBDMatrixContext()


class XGBModel(PythonBasedModel):
    """Train a XGBoost model to predict the normalized throughputs of programs.
    Let the normalized throughput be the score of a program (higher is better). We predict
    the (approximate) score of a program = the sum of the scores of all stages in this program.
    i.e. score(P) = score_s0 + score_s1 + ... + score_sn,
    where score_si is the score of Stage i in Program P.
    We extract feature for each stage and let the xgboost predict the score for each stage.
    We then sum up the predictions as the score of the whole program.
    We use RMSE as the loss function.  i.e. loss(P, y) = 1/2 * (score(P) - y)^2,
    where P is the program and y is the normalized throughput according to
    the ground truth (measurement).
    XGBoost does not support this loss function because `score(P)` is a sum of the prediction
    of several samples, so we implemented a custom loss function and call it pack-sum-rmse.
    It is called "pack-sum" because we combine several samples into a "pack" and sum up
    their predictions.

    Parameters
    ----------
    verbose_eval: int = 25
        Print training log every `verbose_eval` iterations.
    num_warmup_sample: int = 100
        The minimum number of samples to start to use the trained model.
        If the number of samples is less than this number, the model outputs random predictions.
    seed: Optional[int]
        The random seed
    model_file: Optional[str]
        If is not None, save model to this file after every update.
    adapative_training: bool = False
        Whether to use adapatie training, which reduces the training frequency when there are
        too many logs.
    """

    def __init__(
        self,
        verbose_eval=25,
        num_warmup_sample=100,
        seed=None,
        model_file=None,
        adapative_training=False,
    ):
        global xgb
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        except ImportError:
            # add "from Node" to silence
            # "During handling of the above exception, another exception occurred"
            raise ImportError(
                "XGBoost is required for XGBModel. "
                "Please install its python package first. "
                "Help: (https://xgboost.readthedocs.io/en/latest/) "
            ) from None

        self.xgb_params = {
            "max_depth": 10,
            "gamma": 0.001,
            "min_child_weight": 0,
            "eta": 0.2,
            # todo(merrymercy): automatically decrease learning rate when the loss is too large
            "n_gpus": 0,
            "nthread": multiprocessing.cpu_count() // 2,
            "verbosity": 0,
            "seed": seed or 43,
            "disable_default_eval_metric": 1,
        }
        self.bst = None
        self.plan_size = 32
        self.num_warmup_sample = num_warmup_sample
        self.verbose_eval = verbose_eval
        self.model_file = model_file
        self.adapative_training = adapative_training

        super().__init__()

        # cache measurement input/result pairs and extracted features
        self.inputs = []
        self.results = []
        self.last_train_length = 0
        self.inputs_feature_cache = []

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).
        XGBoost does not support incremental training, so we re-train a new model every time.
        Parameters
        ----------
        inputs : List[MeasureInput]
            The measurement inputs
        results : List[MeasureResult]
            The measurement results
        """
        if len(inputs) <= 0:
            return
        assert len(inputs) == len(results)

        self.inputs.extend(inputs)
        self.results.extend(results)

        if (
            self.adapative_training
            and len(self.inputs) - self.last_train_length < self.last_train_length / 5
        ):
            # Set a training threshold related to `last_train_length` to reduce the training
            # overhead when there're too many logs
            return
        self.last_train_length = len(self.inputs)

        # extract feature
        n_cached = len(self.inputs_feature_cache)
        features, normalized_throughputs, task_ids = get_per_store_features_from_measure_pairs(
            self.inputs, self.results, skip_first_n_feature_extraction=n_cached
        )
        if n_cached > 0:
            features = list(features)
            features[:n_cached] = self.inputs_feature_cache
            features = np.array(features, dtype=object)
        self.inputs_feature_cache = features
        dtrain = pack_sum_xgbmatrix(
            features, normalized_throughputs, task_ids, normalized_throughputs
        )

        # train xgb model
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=10000,
            obj=pack_sum_square_error,
            callbacks=[
                custom_callback(
                    stopping_rounds=50,
                    metric="tr-p-rmse",
                    fevals=[
                        pack_sum_rmse,
                        pack_sum_average_peak_score(self.plan_size),
                    ],
                    evals=[(dtrain, "tr")],
                    maximize=False,
                    verbose_eval=self.verbose_eval,
                )
            ],
        )

        # Update the model file if it has been set
        if self.model_file:
            self.save(self.model_file)

    def predict(self, task, states):
        """Predict the scores of states
        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        statse : List[State]
            The input states
        Returns
        -------
        scores: List[float]
            The predicted scores for all states
        """
        features = get_per_store_features_from_states(states, task)
        if self.bst is not None and len(self.inputs) > self.num_warmup_sample:
            dtest, pack_ids = feature_to_pack_sum_xgbmatrix(features)
            raw_preds = self.bst.predict(dtest)
            ret = predict_throughput_pack_sum(raw_preds, pack_ids)
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict -inf for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float("-inf")

        return ret

    def predict_stages(self, task, states):
        """Predict the scores of all stages in states. This is the breakdown version of `predict`.

        Parameters
        ----------
        search_task : SearchTask
            The search task of states
        statse : List[State]
            The input states

        Returns
        -------
        scores: List[float]
            The predicted scores for all stages in all states in the packed format

        Note
        ----
        For faster data copy between c++ and python, the python part returns scores in a
        single flatten array using a packed format. The c++ part then unpacks the flatten array.
        The packed format is:
        {

          float  scores[N];                 // scores[i] is the score for states[i].
          int    n_stage_0;                 // the number of stages in states[0]
          float  stage_scores_0[[n_stage_0] // the scores for all stages in states[0]
          int    n_stage_1;                 // the number of stages in states[1]
          float  stage_scores_1[n_stage_1]; // the scores for all stages in states[1]
          ...
          int    n_stage_i;                 // the number of stages in states[i]
          float  stage_scores_1[n_stage_i]; // the scores for all stages in states[i]
          ...  // untill i == N - 1

        }
        To implement this format, we also store int as float, so we can store all numbers
        into a single float array.
        """
        features = get_per_store_features_from_states(states, task)
        if self.bst is not None and len(self.inputs) > self.num_warmup_sample:
            dtest, pack_ids = feature_to_pack_sum_xgbmatrix(features)
            raw_preds = self.bst.predict(dtest)
            breakdown = predict_throughput_pack_sum(raw_preds, pack_ids)
            stage_scores = [[] for _ in range(len(states))]
            for pred, pack_id in zip(raw_preds, pack_ids):
                stage_scores[pack_id].append(pred)
            for idx, stage_score in enumerate(stage_scores):
                breakdown = np.append(breakdown, len(stage_score))
                breakdown = np.concatenate((breakdown, np.array(stage_score)))
        else:
            breakdown = np.concatenate(
                (
                    np.random.uniform(0, 1, (len(states),)),
                    np.zeros(
                        len(states),
                    ),
                )
            )

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                breakdown[idx] = float("-inf")

        return breakdown

    def update_from_file(self, file_name, n_lines=None):
        """Load measure records from a log file to update the cost model.
        This function can be used to pre-train the cost model with history log files.
        Parameters
        ----------
        file_name: str
            The filename
        n_lines: Optional[int]
            Only load first n lines of the log file
        """
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("XGBModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        """Save the model to a file
        Parameters
        ----------
        file_name: str
            The filename
        """
        self.bst.save_model(file_name)

    def load(self, file_name: str):
        """Load the model from a file
        Parameters
        ----------
        file_name: str
            The filename
        """
        if self.bst is None:
            self.bst = xgb.Booster(self.xgb_params)
        self.bst.load_model(file_name)
        self.num_warmup_sample = -1


def feature_to_pack_sum_xgbmatrix(xs):
    """Convert an extracted multi-stage feature vector to a xgbmatrx in pack-sum format
    Parameters
    ----------
    xs: np.ndarray
        The feature vector
    Returns
    -------
    dmatrix: xgb.DMatrix
        The DMatrix
    pack_ids: List[int]
        pack ids information
    """
    x_flatten = []
    pack_ids = []

    for ct, x in enumerate(xs):
        for row in x:
            x_flatten.append(row)
            pack_ids.append(ct)

    return xgb.DMatrix(np.array(x_flatten)), pack_ids


def pack_sum_xgbmatrix(xs, ys, gids=None, weights=None):
    """Convert (feature, label) pairs into a xgb matrix with pack-sum format
    Parameters
    ----------
    xs: np.ndarray
        The feature vector
    ys: np.ndarray
        The normaizlied throughput
    gids: Optional[List[int]]
        Group id (task id)
    weights: Optional[np.ndarray]
        The weight of samples
    Returns
    -------
    dmatrix: xgb.DMatrix
        The DMatrix with pack-sum information
    """
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

    ret = xgb.DMatrix(np.array(x_flatten), y_flatten)
    if weights is not None:
        ret.set_weight(weights_flatten)
    dmatrix_context.set("pack_ids", ret, np.array(pack_ids))
    dmatrix_context.set("group_sizes", ret, group_sizes)
    return ret


def predict_throughput_pack_sum(raw_preds, pack_ids):
    """Predict the throughputs for predictions in pack-sum format
    Parameters
    ----------
    raw_preds: np.ndarray
        The raw predictions
    pack_ids: List[int]
        The pack id for predictions
    Returns
    -------
    throughputs: np.ndarray
        The throughput
    """
    sum_pred = np.bincount(pack_ids, weights=raw_preds)
    return sum_pred


def pack_sum_square_error(preds, dtrain):
    """Implement square error loss on pack-sum format as
     a custom objective function for xgboost.
    Parameters
    ----------
    preds: np.ndarray
        The predicitons
    dtrain: xgb.DMatrix
        The training set
    Returns
    -------
    gradient: np.ndarray
    hessian: np.ndarray
        gradient and hessian according to the xgboost format
    """
    pack_ids = dmatrix_context.get("pack_ids", dtrain)
    weight = dtrain.get_weight()

    sum_pred = np.bincount(pack_ids, weights=preds)
    x = sum_pred[pack_ids]
    y = dtrain.get_label()
    gradient = x - y
    hessian = np.ones_like(gradient)

    if len(weight) == 0:
        return gradient, hessian

    return gradient * weight, hessian * weight


def pack_sum_rmse(raw_preds, labels):
    """Evaluate RMSE (rooted mean square error) in the pack-sum format
    Parameters
    ----------
    raw_preds: np.ndarray
        The raw prediction
    labels: xgb.DMatrix
        The groud-truth label matrix
    Returns
    -------
    name: str
    score: float
        The name and score of this metric
    """
    pack_ids = dmatrix_context.get("pack_ids", labels)
    preds = predict_throughput_pack_sum(raw_preds, pack_ids)[pack_ids]
    return "p-rmse", np.sqrt(np.mean(np.square((preds - labels.get_label()))))


def pack_sum_average_peak_score(N):
    """Return the evaluation function for average-peak-score@N
    Parameters
    ----------
    N: int
        The "N" in "average-peak-score@N"
    Returns
    -------
    The evaluation function
    """

    def feval(preds, labels):
        """Evaluate average-peak-score@N in the pack-sum format
        Parameters
        ----------
        raw_preds: np.ndarray
            The raw prediction
        labels: xgb.DMatrix
            The groud-truth label matrix
        Returns
        -------
        name: str
        score: float
        The name and score of this metric
        """
        group_sizes = dmatrix_context.get("group_sizes", labels, [len(preds)])
        pack_ids = dmatrix_context.get("pack_ids", labels)

        preds = predict_throughput_pack_sum(preds, pack_ids)
        labels = (
            np.bincount(pack_ids, weights=labels.get_label())
            / np.unique(pack_ids, return_counts=True)[1]
        )

        scores = []
        offset = 0
        for size in group_sizes:
            preds_group = preds[offset : offset + size]
            labels_group = labels[offset : offset + size]
            offset += size

            trials = np.argsort(preds_group)[::-1][:N]
            trial_scores = labels_group[trials]
            curve = max_curve(trial_scores) / np.max(labels_group)
            scores.append(np.mean(curve))
        return "a-peak@%d" % N, np.mean(scores)

    return feval


def custom_callback(
    stopping_rounds,
    metric,
    fevals,
    evals=(),
    log_file=None,
    maximize=False,
    verbose_eval=True,
    skip_every=2,
):
    """Callback function for xgboost to support multiple custom evaluation functions"""
    # pylint: disable=import-outside-toplevel
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric

    try:
        from xgboost.training import aggcv
    except ImportError:
        from xgboost.callback import _aggcv as aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state["maximize_score"] = maximize
        state["best_iteration"] = 0
        if maximize:
            state["best_score"] = float("-inf")
        else:
            state["best_score"] = float("inf")

        if bst is not None:
            if bst.attr("best_score") is not None:
                state["best_score"] = float(bst.attr("best_score"))
                state["best_iteration"] = int(bst.attr("best_iteration"))
                state["best_msg"] = bst.attr("best_msg")
            else:
                bst.set_attr(best_iteration=str(state["best_iteration"]))
                bst.set_attr(best_score=str(state["best_score"]))
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
                res = [x.split(":") for x in bst_eval.split()]
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
                if "null" in item[0]:
                    continue
                infos.append("%s: %.6f" % (item[0], item[1]))

            logger.debug("\t".join(infos))
            if log_file:
                with open(log_file, "a") as fout:
                    fout.write("\t".join(infos) + "\n")

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state["best_score"]
        best_iteration = state["best_iteration"]
        maximize_score = state["maximize_score"]
        if (maximize_score and score > best_score) or (not maximize_score and score < best_score):
            msg = "[%d] %s" % (env.iteration, "\t".join([_fmt_metric(x) for x in eval_res]))
            state["best_msg"] = msg
            state["best_score"] = score
            state["best_iteration"] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(
                    best_score=str(state["best_score"]),
                    best_iteration=str(state["best_iteration"]),
                    best_msg=state["best_msg"],
                )
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state["best_msg"]
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback
