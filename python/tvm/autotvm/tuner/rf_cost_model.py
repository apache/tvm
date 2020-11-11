from .model_based_tuner import CostModel, FeatureCache

from .xgboost_cost_model import _extract_curve_feature_index,_extract_knob_feature_index, _extract_itervar_feature_index

class RFEICostModel(CostModel):
    def __init__(self, task, fea_type="itervar",num_threads=None, log_interval=25):
        super(RFEICostModel, self).__init__()
        self.task = task
        self.target = task.target
        self.space = task.config_space
        
        self.prior = RandomForestRegressor(n_estimators=10, random_state=2, max_features=10)
        self.fea_type = fea_type
        self.num_threads = num_threads
        self.log_interval = log_interval
        if fea_type == 'itervar':
            self.feature_extract_func = _extract_itervar_feature_index
        elif fea_type == 'knob':
            self.feature_extract_func = _extract_knob_feature_index
        elif fea_type == 'curve':
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + fea_type)

        self.feature_cache = FeatureCache()
        self.best_flops = 0.0
        self.pool = None
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        # if self.upper_model:  # base model will reuse upper model's pool,
        #     self.upper_model._reset_pool(space, target, task)
        #     return

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
        # if self.upper_model:
        #     return self.upper_model._get_pool()
        return self.pool

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
        tic = time.time()

        x_train = self._get_feature(xs)
        self.best_flops = max(ys)
        self.prior.fit(x_train, ys)

        logger.debug(
            "RF train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
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

        logger.debug("RF load %d entries from history log file", len(data))

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

        xs, ys = np.array(xs), np.array(ys)

        self.best_flops = max(ys)
        self.prior.fit(xs, ys)

        logger.debug("RF train: %.2f\tobs: %d", time.time() - tic, len(xs))

        return True

    def predict(self, xs, output_margin=False):
        predicts, _ = self._prediction_variation(xs)
        return predicts

    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return RFEICostModel(
            self.task, self.fea_type, self.loss_type, self.num_threads, self.log_interval, self
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

    def _prediction_variation(self, x_to_predict):
        """Use Bayesian Optimization to predict the y and get the prediction_variation
        """
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

    def __del__(self):
        self._close_pool()