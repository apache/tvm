# pylint: disable=no-else-return,invalid-name
"""Tuner that use xgboost as cost model"""

import multiprocessing
import logging
import time
import gc

import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from ..util import sample_ints, get_rank
from .. import feature
from .model_based_basetuner import ModelBasedBaseTuner, random_walk, point2knob
from .opt_method import sa_find_maximum, submodular_pick
from .metric import max_curve, recall_curve, cover_curve

class XGBTuner(ModelBasedBaseTuner):
    """Tuner that use xgboost as cost model

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    batch_size: int
        Tuner will re-fit the model per `batch_size` new measure samples

    num_bst: int
        The number of booster models to compute empirical mean and variance
        for EI/UCB acquisition function
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    opt_type: str, optional
        Only supports simulated annealing currently
    acq_type: str
        Type of acquisition function if there are multiple boosters
        Supports 'ei', 'ucb' and 'mean'
    transfer_type: str
        Only support 'residual' currently
    feature_type: str
        if is 'itervar', use features extracted from IterVar (loop variable)
        if is 'knob', use flatten ConfigEntity directly
        if is 'curve', use sampled curve feature (relation feature)

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' is good.
                                'itervar' is more accurate but 'knob' is much faster
        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability
                               'knob' is faster
        For cross-device or cross-operator tuning, you can use 'curve' only.

    diversity_filter_ratio: float
        If is not None, the tuner will first select
        top-(batch_size * diversity_filter_ratio) candidates according to the cost model
        and then pick batch_size of them according to the diversity metric

    sa_n_iter: int
        The maximum number of iterations of simulated annealing after refitting a new cost model
    sa_temp: float or list of float
        Temperature config of simulated annealing
    sa_persistent: bool
        Whether keep persistent states of SA points among different models
    sa_parallel_size: int
        Number of parallel Markov chains when doing parallel simulated annealing
    """

    def __init__(self, task, batch_size=32, num_bst=1,
                 loss_type='rank', opt_type='sa', acq_type='mean', feature_type='itervar',
                 transfer_type='residual', diversity_filter_ratio=None,
                 sa_persistent=True, sa_n_iter=500, sa_temp=(1, 0), sa_parallel_size=128):
        super(XGBTuner, self).__init__(task, batch_size,
                                       sa_n_iter, sa_temp, sa_persistent, sa_parallel_size)

        if xgb is None:
            raise RuntimeError("XGBoost is required for XGBTuner. "
                               "Please install its python package first. "
                               "Help: (https://xgboost.readthedocs.io/en/latest/) ")

        self.num_bst = num_bst
        self.sa_early_stopping = 30

        # feature extraction and cache
        global _extract_space, _extract_target, _extract_task
        _extract_space = self.space
        _extract_target = self.target
        _extract_task = self.task
        self.fea_cache = {}

        self.train_ct += 1

        # data for transfer
        self.transfer_type = transfer_type
        self.base_predictor = None

        # cost model (xgboost)
        self.loss_type = loss_type
        if self.loss_type == 'reg':
            self.xgb_params = {
                'max_depth': 3,
                'gamma': 0.0001,
                'min_child_weight': 1,

                'subsample': 1.0,

                'eta': 0.3,
                'lambda': 1.00,
                'alpha': 0,

                'objective': 'reg:linear',
                'silent': 1,
            }
        elif self.loss_type == 'rank':
            self.xgb_params = {
                'max_depth': 3,
                'gamma': 0.0001,
                'min_child_weight': 1,

                'subsample': 1.0,

                'eta': 0.3,
                'lambda': 1.00,
                'alpha': 0,

                'objective': 'rank:pairwise',
                'silent': 1,
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.bst_list = []

        # opt method
        self.opt_type = opt_type
        if diversity_filter_ratio is None:
            self.diversity_filter_size = None
        else:
            self.diversity_filter_size = diversity_filter_ratio * self.batch_size

        def trans_func(points):
            """random walk transition function"""
            args = ((p, self.dims) for p in points)
            return self.pool.map(random_walk, args)

        def eval_func(points):
            """acquisition function"""
            feas = self._get_feature(points)
            dtest = xgb.DMatrix(feas)

            if self.base_predictor is not None:
                pbase = self.base_predictor.predict(dtest, output_margin=True)
                if self.transfer_type == 'residual':
                    dtest.set_base_margin(pbase * self._base_discount())
                elif self.transfer_type == 'stacking':
                    feas = np.concatenate((feas, pbase.reshape((-1, 1))), axis=1)
                    dtest = xgb.DMatrix(feas)

            if self.bst_list:
                preds = np.zeros((self.num_bst, len(feas)), dtype=np.float32)
                for i, bst in enumerate(self.bst_list):
                    preds[i:, ] = bst.predict(dtest)
            else:  # only have transfer base model
                assert self.base_predictor is not None
                preds = [pbase]

            if self.num_bst == 1:
                return preds[0]
            else:
                means = np.mean(preds, axis=0)
                stds = np.std(preds, axis=0)

                if acq_type == 'ei':
                    acq = _ei_max(means, stds, self.y_max, 0)
                elif acq_type == 'ucb':
                    acq = _ucb_max(means, stds)
                else:
                    acq = means

                return acq

        self.sa_trans_func = trans_func
        self.sa_eval_func = eval_func
        self.pool = None
        self.feature_type = feature_type
        if feature_type == 'itervar':
            self.feature_extract_func = _extract_itervar_feature_index
        elif feature_type == 'knob':
            self.feature_extract_func = _extract_knob_feature_index
        elif feature_type == 'curve':
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + feature_type)

        logging.info("feature dim: %d", len(self.feature_extract_func(0)))

    def _base_discount(self):
        """discount factor for the base model used in transfer learning"""
        return 2 ** (-self.train_ct + 1)

    def _update_model(self):
        # need retrain
        if len(self.xs) >= self.batch_size * (self.train_ct + 1) \
                and self.flops_max > 1e-6:
            tic = time.time()
            self.pool = multiprocessing.Pool()

            self.train_ct += 1
            x_train = self._get_feature(self.xs)
            y_train = np.array(self.ys)
            y_train /= np.max(y_train)
            self.y_max = np.max(y_train)

            if self.base_predictor is not None:
                dbase = xgb.DMatrix(x_train)
                pbase = self.base_predictor.predict(dbase, output_margin=True)

            valid_index = y_train > 1e-6

            self.bst_list = []
            self.reg_list = []
            self.binary_list = []

            if self.num_bst > 1:
                pick_ratio = 0.8
            else:
                pick_ratio = 1

            for _ in range(self.num_bst):
                index = np.random.choice(len(x_train), int(len(x_train) * pick_ratio),
                                         replace=False)
                dtrain = xgb.DMatrix(x_train[index], y_train[index])

                if self.base_predictor is not None:
                    if self.transfer_type == 'residual':
                        dtrain.set_base_margin(pbase[index] * self._base_discount())
                    elif self.transfer_type == 'stacking':
                        train_tmp = np.concatenate((x_train[index],
                                                    pbase[index].reshape((-1, 1))), axis=1)
                        dtrain = xgb.DMatrix(train_tmp, y_train[index])

                bst = xgb.train(self.xgb_params, dtrain,
                                num_boost_round=8000,
                                callbacks=[custom_callback(
                                    stopping_rounds=20,
                                    metric='tr-a-recall@%d' % self.batch_size,
                                    evals=[(dtrain, 'tr')],
                                    maximize=True,
                                    fevals=[
                                        xgb_average_recalln_curve_score(
                                            self.batch_size),
                                    ],
                                    verbose_eval=20)])
                self.bst_list.append(bst)

            logging.info("train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
                         time.time() - tic, len(self.xs),
                         len(self.xs) - np.sum(valid_index), len(self.fea_cache))

            tic = time.time()
            n_keep = self.diversity_filter_size if self.diversity_filter_size else self.batch_size

            if not self.sa_persistent or self.sa_points is None:
                self.sa_points = sample_ints(len(self.space), self.sa_parallel_size)

            if self.opt_type == 'sa':
                maximums, next_points = sa_find_maximum(
                    self.sa_points, self.sa_trans_func, self.sa_eval_func,
                    n_iter=self.sa_n_iter, n_keep=n_keep,
                    temp=self.sa_temp, early_stop=self.sa_early_stopping,
                    exclusive=self.visited, verbose=50)
            else:
                raise RuntimeError("invalid opt type " + self.opt_type)

            if self.diversity_filter_size is not None:
                scores = self.sa_eval_func(maximums)
                knobs = [point2knob(x, self.dims) for x in maximums]
                pick_index = submodular_pick(0 * scores, knobs, self.batch_size, knob_weight=1)
                maximums = np.array(maximums)[pick_index]

            logging.info("opt: %.2f\tn_cache: %d", time.time() - tic, len(self.fea_cache))

            self.sa_points = next_points
            self.trials = maximums
            self.trial_pt = 0

            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def load_history(self, data_set):
        tic = time.time()
        args = list(data_set)

        self.pool = multiprocessing.Pool()
        if self.feature_type == 'itervar':
            feature_extract_func = _extract_itervar_feature_log
        elif self.feature_type == 'knob':
            feature_extract_func = _extract_knob_feature_log
        elif self.feature_type == 'curve':
            feature_extract_func = _extract_curve_feature_log
        res = self.pool.map(feature_extract_func, args)

        xs, ys = zip(*res)
        xs, ys = np.array(xs), np.array(ys)

        logging.info("load history data %s, elapsed %.2f",
                     xs.shape, time.time() - tic)

        if self.transfer_type in ['residual', 'stacking']:
            # train base model
            tic = time.time()
            x_train = xs
            y_train = ys
            y_train /= np.max(y_train)
            index = np.random.permutation(len(x_train))
            dtrain = xgb.DMatrix(x_train[index], y_train[index])

            n_recall = self.batch_size * 2
            call_back = custom_callback(
                stopping_rounds=100,
                metric='tr-a-recall@%d' % n_recall,
                evals=[(dtrain, 'tr')],
                maximize=True,
                fevals=[
                    xgb_average_recalln_curve_score(
                        n_recall),
                ],
                verbose_eval=20)

            xgb_params = self.xgb_params
            xgb_params['eta'] = 0.2

            bst = xgb.train(xgb_params, dtrain,
                            num_boost_round=200,
                            callbacks=[call_back])

            self.base_predictor = bst
            logging.info("base predictor done, elapsed %.2f", time.time() - tic)

            # find initial points
            if not self.trials:
                tic = time.time()

                if not self.sa_persistent or self.sa_points is None:
                    self.sa_points = sample_ints(len(self.space), self.sa_parallel_size)

                maximums, next_points = sa_find_maximum(
                    self.sa_points, self.sa_trans_func, self.sa_eval_func,
                    n_iter=self.sa_n_iter, n_keep=self.batch_size,
                    temp=self.sa_temp, early_stop=self.sa_early_stopping,
                    exclusive=self.visited, verbose=50)
                self.sa_points = next_points
                self.trials = maximums
                self.trial_pt = 0

                self.pool.terminate()
                self.pool.join()
                self.pool = None

                logging.info("base sa done, elapsed %.2f", time.time() - tic)

            # cancel the fake addition in __init__
            self.train_ct -= 1

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if len(self.fea_cache) >= 100000:
            del self.fea_cache
            self.fea_cache = {}
            gc.collect()

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in self.fea_cache]

        if need_extract:
            feas = self.pool.map(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                self.fea_cache[i] = fea

        ret = np.empty((len(indexes), self.fea_cache[indexes[0]].shape[-1]), dtype=np.float32)
        for i, ii in enumerate(indexes):
            ret[i, :] = self.fea_cache[ii]
        return ret

try:
    from scipy.stats import norm
except ImportError:
    norm = None

def _ei_max(mean, std, y_max, xi):
    """Acquisition function of Expected Improvement"""
    assert norm, "Python package scipy is required for use EI as acquisition function" \
                 "We need from scipy.stats import norm"
    z = (mean - (y_max + xi)) / std
    return (mean - (y_max + xi)) * norm.cdf(z) + std * norm.pdf(z)

def _ucb_max(mean, std, kappa=2.576):
    """Acquisition function of Upper Confidence Bound"""
    return mean + kappa * std


_extract_space = None
_extract_target = None
_extract_task = None
def _extract_itervar_feature_index(index):
    """extract iteration var feature for an index in extract_space"""
    config = _extract_space.get(index)
    with _extract_target:
        sch, args = _extract_task.instantiate(config)
    fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
    if config.other_option_keys:
        return np.concatenate((fea, list(config.get_other_option().values())))
    else:
        return np.array(fea)

def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, args = inp.task.instantiate(config)
    fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
    if config.other_option_keys:
        x = np.concatenate((fea, list(config.get_other_option().values())))
    else:
        x = np.array(fea)

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0
    return x, y

def _extract_knob_feature_index(index):
    """extract knob feature for an index in extract_space"""
    config = _extract_space.get(index)
    return config.get_flatten_feature()

def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    inp, res = arg
    config = inp.config
    x = config.get_flatten_feature()

    if res.error_no == 0:
        with inp.target:  # necessary, for calculating flops of this task
            inp.task.instantiate(config)
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0
    return x, y

def _extract_curve_feature_index(index):
    """extract sampled curve feature for an index in extract_space"""
    config = _extract_space.get(index)
    with _extract_target:
        sch, args = _extract_task.instantiate(config)
    fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
    if config.other_option_keys:
        return np.concatenate((fea, list(config.get_other_option().values())))
    else:
        return np.array(fea)

def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, args = inp.task.instantiate(config)
    fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
    if config.other_option_keys:
        x = np.concatenate((fea, list(config.get_other_option().values())))
    else:
        x = np.array(fea)

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0
    return x, y

def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    save_file="xgb_checkpoint", save_every=None,
                    maximize=False, verbose_eval=True):
    """callback function for xgboost to support multiple custom evaluation functions"""
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
        infos = ["XGB Iter: %3d" % i]
        for item in eval_res:
            if 'null' in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if not isinstance(verbose_eval, bool) and i % verbose_eval == 0:
            logging.info("\t".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + '\n')

        ##### save model #####
        if save_every and i % save_every == 0:
            filename = save_file + ".%05d.bst" % i
            logging.info("save model to %s ...", filename)
            bst.save_model(filename)

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
                logging.info("Stopping. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback

# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)
    return feval

def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]
    return feval

def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N
    return feval

def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]
    return feval

def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]
    return feval

def xgb_null_score(_):
    """empty score function for xgb"""
    def feval(__, ___):
        return "null", 0
    return feval
