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
import tempfile

import numpy as np

from ..measure import MeasureInput, create_measure_batch
from ..utils import format_si_prefix

from ..env import GLOBAL_SCOPE

logger = logging.getLogger("autotvm")


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
        self.space = self.task.config_space

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0
        self.error_ct_threshold = 150

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

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

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        si_prefix: str
            One of tvm.autotvm.utils.SI_PREFIXES. The SI prefix to use when reporting FLOPS.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        # Validate si_prefix arg
        format_si_prefix(0, si_prefix)

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        errors = []
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                    result_msg = res
                else:
                    flops = 0
                    error_ct += 1
                    tb, error = res.costs
                    if isinstance(error, str):
                        errors.append(tb + "\n" + error)
                    else:
                        errors.append(tb + "\n" + str(error))
                    result_msg = errors[-1]

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug(
                    "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                    i + k + 1,
                    si_prefix,
                    format_si_prefix(flops, si_prefix),
                    format_si_prefix(self.best_flops, si_prefix),
                    result_msg,
                    config,
                )

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > self.error_ct_threshold:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        if error_ct == i:
            _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
            with open(f, "w") as file:
                file.write("\n".join(errors))
            logging.warning(
                "Could not find any valid schedule for task %s. "
                "A file containing the errors has been written to %s.",
                self.task,
                f,
            )
        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set, min_seed_records=500):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (autotvm.measure.MeasureInput, autotvm.measure.MeasureResult) pair
            Previous tuning records
        min_seed_records: int
            Defaults to 500. Indicates the minimum number of records to
            train the tuner with. If there are less than `min_seed_records`
            number of records in `data_set`, no training of the tuner
            will be done.
        """
        raise NotImplementedError()

    def set_error_threshold(self, threshold):
        """Modify error counter threshold, which controls switch to debug mode

        Parameters
        ----------
        threshold: New threshold value
        """
        self.error_ct_threshold = threshold
