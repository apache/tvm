import time

import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel

from test_autotvm_common import get_sample_task, get_sample_records


def test_fit():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    base_model = XGBoostCostModel(task, feature_type='itervar', loss_type='rank')
    base_model.fit_log(records, plan_size=32)

    upper_model = XGBoostCostModel(task, feature_type='itervar', loss_type='rank')
    upper_model.load_basemodel(base_model)

    xs = np.arange(10)
    ys = np.arange(10)

    upper_model.fit(xs, ys, plan_size=32)


def test_tuner():
    task, target = get_sample_task()
    records = get_sample_records(n=100)

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records)


if __name__ == "__main__":
    test_fit()
    test_tuner()

