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
"""Tuning Job Class"""
import datetime
import numpy as np

from ..env import GLOBAL_SCOPE
from .callback import log_to_file
from ..record import load_from_file


class TuningJob:
    """A context to hold information during an auto-tuning job.

    The results of a tuning job can optionally be saved to either a
    log file or a config library.
    Parameters
    ----------
    log: str
        A file path where to where the history log should be written.
    target: str
        A TVM target string describing the current target.
    platform: dict
        A dictionary to hold additional platform information.
    content: dict
        A dictionary to hold additional content information.
    config_library: ConfigLibrary
        The config library into which to store results.

    """
    def __init__(
            self,
            log,
            target,
            platform=None,
            content=None,
            config_library=None,
    ):
        self.log = log
        self.target = target
        if platform is None:
            platform = {}
        if content is None:
            content = {}
        self.platform = platform
        self.content = content
        self.config_library = config_library
        self.results_by_workload = {}
        self.start_time = None
        self.end_time = None
        self._logging_callback = log_to_file(self.log)

    def log_configs(self, tuner, inputs, results):
        """Log newly measured configs and update the best results.

        Parameters
        ----------
        tuner: tuner.Tuner
            The tuner used during tuning.
        inputs: list
            The list of tuning inputs measured.
        results: list
            The list of tuning results obtained.

        """
        self._logging_callback(tuner, inputs, results)
        tuner_name = type(tuner).__name__
        for inp, result in zip(inputs, results):
            workload = inp.task.workload
            if workload in self.results_by_workload:
                _, best_result, _, trials = self.results_by_workload[workload]
                if np.mean(result[0]) < np.mean(best_result[0]):
                    self.results_by_workload[workload] = [inp, result, tuner_name, trials]
            else:
                self.results_by_workload[workload] = [inp, result, tuner_name, 0]

            self.results_by_workload[workload][3] += 1  # Increment no. of trials

    def get_records(self):
        """Return the records of the active job."""
        records = load_from_file(self.log)
        return records

    def __enter__(self):
        GLOBAL_SCOPE.tuning_job = self
        self.start_time = datetime.datetime.utcnow().isoformat()

    def __exit__(self, exception_type, exception_value, traceback):
        GLOBAL_SCOPE.tuning_job = None
        self.end_time = datetime.datetime.utcnow().isoformat()
        if self.config_library:
            self.config_library.save_job(self)
