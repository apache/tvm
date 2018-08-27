"""Test builder and runner"""
import logging
import time

import numpy as np

import tvm
from tvm import autotvm
from test_autotvm_common import get_sample_task, bad_matmul
from tvm.autotvm.measure.measure import Runner, MeasureResult, MeasureErrorNo

def test_task_tuner_without_measurement():
    """test task and tuner without measurement"""
    task, target = get_sample_task()

    class DummyRunner(Runner):
        def __init__(self):
            super(DummyRunner, self).__init__(1, 1)

        def run(self, measure_inputs, build_results):
            return [MeasureResult((np.random.random(),), 0, 0.2, time.time())
                    for _ in range(len(measure_inputs))]

        def get_build_kwargs(self):
            return {}

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=DummyRunner()
    )

    logging.info("%s", task.config_space)

    for tuner_class in [autotvm.tuner.RandomTuner,
                        autotvm.tuner.GridSearchTuner,
                        autotvm.tuner.GATuner,
                        autotvm.tuner.XGBTuner]:
        tuner = tuner_class(task)
        tuner.tune(n_trial=10, measure_option=measure_option)
        assert tuner.best_flops > 1

def test_check_correctness():
    task, target = get_sample_task()

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(check_correctness=True)
    )

    def _callback_correct(tuner, measure_inputs, measure_results):
        for inp, res in zip(measure_inputs, measure_results):
            assert res.error_no == 0

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=2, measure_option=measure_option,
               callbacks=[_callback_correct])

    # a bad template
    n = 128
    target = tvm.target.create("llvm -device=bad_device")
    task = autotvm.task.create(bad_matmul, args=(n, n, n, 'float32'), target=target)

    def _callback_wrong(tuner, measure_inputs, measure_results):
        for inp, res in zip(measure_inputs, measure_results):
            assert res.error_no == MeasureErrorNo.WRONG_ANSWER

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=2, measure_option=measure_option,
               callbacks=[_callback_wrong])


def test_min_repeat_ms():
    task, target = get_sample_task()

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=1, min_repeat_ms=100)
    )

    def _callback(tuner, measure_inputs, measure_results):
        for inp, res in zip(measure_inputs, measure_results):
            if res.error_no != 0:
                continue

            assert 1000 * np.mean(res.costs) * \
                   measure_option['runner'].cur_number >= 100

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=5, measure_option=measure_option,
               callbacks=[_callback])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    test_task_tuner_without_measurement()
    test_check_correctness()
    test_min_repeat_ms()
