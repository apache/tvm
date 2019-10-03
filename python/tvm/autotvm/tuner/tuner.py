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
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging

import numpy as np

from ..env import GLOBAL_SCOPE
from ..measure import MeasureInput, create_measure_batch

logger = logging.getLogger('autotvm')

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
        self.record_cache = []
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

        # set up dependency if available
        self.tune_depend_only = self.task.depend != self.task and self.task.depend.tuned_configs
        if self.tune_depend_only:
            logger.debug('Reduced the tuning space to the best ones from the dependent task')

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

    def parse_depend_mode(self, mode, total):
        """Parse the mode of using best configs from the dependent task.

        Parameters
        ----------
        mode: str
            It can be either one of the following two formats.
            "topN", where N is a positive integer: Keep the top N best schedules.
            "N%", where 0 < N < 1: Keep the schedules with top N% performance.

        total: int
            The total number of configs from the dependent task.

        Returns
        -------
        a number of the top-N configs this task should depend.
        """
        try:
            if mode.startswith('top'):
                return int(mode[3:])
            elif mode.find('%') != 0:
                return min(1, int(total * (float(mode[:-1]) / 100)))
            else:
                raise ValueError
        except ValueError:
            logger.warning('Unknown mode of using the dependent best configs: %s', mode)
        return 1

    def tune(self, n_trial, measure_option, depend_mode='top1', early_stopping=None, callbacks=()):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        depend_mode: str
            The mode of using best configs from the dependent task. It can be in either
            one of the following two formats.
            "topN", where N is a positive integer: Keep the top N best schedules.
            "N%", where 0 < N < 1: Keep the schedules with top N% performance.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        self.depend_num = self.parse_depend_mode(depend_mode, len(self.task.depend.tuned_configs))
        depend_configs = iter(self.task.depend.tuned_configs)
        i = error_ct = 0
        while i < n_trial or self.best_flops == 0:
            if not self.has_next():
                break

            # take configs from the dependent task
            configs = []
            if self.tune_depend_only:
                for _ in range(min(n_parallel, n_trial - i, self.depend_num - i)):
                    try:
                        configs.append(next(depend_configs))
                    except StopIteration:
                        break

            # fallback to normal search if no dependent task or configs from it are
            # running out
            if not configs:
                if self.tune_depend_only:
                    i = 0  # Reset the trial
                    logger.debug(
                        'Fallback to normal tuning because all dependent configs are not working'
                    )
                self.tune_depend_only = False
                configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                # maintain the tuned cache
                if res.error_no == 0:
                    self.record_cache.append((inp, res))

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s",
                             i + k + 1, flops / 1e9, self.best_flops / 1e9,
                             res, config)

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

            if self.tune_depend_only and self.best_flops > 0 and i >= self.depend_num:
                break

        # sort the tuned configs
        for record in sorted(self.record_cache, key=lambda r: np.mean(r[1].costs)):
            self.task.tuned_configs.append(record[0].config)
        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.record_cache = []
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """Load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
