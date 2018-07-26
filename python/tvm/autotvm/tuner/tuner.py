# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging

import numpy as np

from ..measure import MeasureInput
from ..measure import create_measure_batch

from ..env import GLOBAL_SCOPE

class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """
        pass

    def tune(self, n_trial, measure_option, early_stopping=None, verbose=1, callbacks=()):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int
            Early stop the tuning when not finding better configs in this number of trials
        verbose: int
            0: silent mode, no output
            1: print every measurement result
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        parallel_num = getattr(measure_batch, 'parallel_num', 1)
        early_stopping = early_stopping or 1e9

        GLOBAL_SCOPE.in_tuning = True
        i = 0
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(parallel_num, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                else:
                    flops = 0
                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                if verbose >= 1:
                    logging.info("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s",
                                 i + k + 1, flops / 1e9, self.best_flops / 1e9,
                                 res, config)

            i += len(results)

            self.update(inputs, results)

            for callback in callbacks:
                callback(self, inputs, results)

            if i > self.best_iter + early_stopping:
                if verbose >= 1:
                    logging.info("Early stopped. Best iter: %d.", self.best_iter)
                break
        GLOBAL_SCOPE.in_tuning = False

        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
