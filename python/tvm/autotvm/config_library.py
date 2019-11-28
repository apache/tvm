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
# under the License.import datetime
#pylint: disable=arguments-differ, method-hidden, inconsistent-return-statements
"""Config Library to store tuning configs"""
import json
import os
from shutil import copyfile
from pathlib import Path

import numpy as np

from . import record
from .task import ApplyHistoryBest


class ConfigLibrary:
    """A library to store auto-tuning results for any number of targets/workloads.

    Parameters
    ----------
    library_dir: str
        Path to the config library directory. If the library does not already
        exist, a new library will be initialised in the directory. This will
        create an 'index.json' file in the directory which contains the location
        of all other files used to store configs in the library.

    """

    LIBRARY_INDEX_FILE_NAME = "index.json"
    JOBS_INDEX_FILE_NAME = "jobs.json"
    JOBS_DIRECTORY_NAME = "jobs"
    BACKUP_DIRECTORY_NAME = "backup"

    def __init__(self, library_dir):
        # Handle if the directory doesn't exist
        if not os.path.isdir(library_dir):
            os.mkdir(library_dir)

        index_file = os.path.join(library_dir, self.LIBRARY_INDEX_FILE_NAME)
        if not os.path.isfile(index_file):
            with open(index_file, "w") as f:
                full_index_path = os.path.abspath(f.name)
                index = {
                    "root": os.path.dirname(full_index_path),
                    "targets": {},
                    "jobs_index": self.JOBS_INDEX_FILE_NAME,
                    "jobs_dir": self.JOBS_DIRECTORY_NAME,
                }
                json.dump(index, f, indent=4)

            with open(os.path.join(library_dir, self.JOBS_INDEX_FILE_NAME), "w") as f:
                json.dump({}, f)

        self.library_dir = library_dir
        self.library_index = index_file
        self.jobs_dir = os.path.join(self.library_dir, self.JOBS_DIRECTORY_NAME)
        self.jobs_index = os.path.join(self.library_dir, self.JOBS_INDEX_FILE_NAME)
        if not os.path.isdir(self.jobs_dir):
            os.makedirs(self.jobs_dir)
        self.backup_dir = os.path.join(self.library_dir, self.BACKUP_DIRECTORY_NAME)
        if not os.path.isdir(self.backup_dir):
            os.makedirs(self.backup_dir)

    def load(self, target):
        """Load the configs for a given TVM target string.

        Returns a DispatchContext with the appropriate configs loaded."""
        target_configs = self._load_target_configs(target)
        return ApplyHistoryBest(target_configs)

    def _load_target_configs(self, target):
        """Yield the configs in the library for a given target."""
        target_file = self.get_config_file(target, create=False)
        if target_file:
            with open(target_file) as f:
                configs = json.load(f)
                for config in configs.values():
                    row = json.dumps(config)
                    yield record.decode(row)

        else:
            yield from []

    def save_job(self, job, save_history=True):
        """Save the results of an auto-tuning job to the library.

        Parameters
        ----------
        job: TuningJob
            The auto-tuning job to save.
        save_history: bool
            Whether to save the history log of the job.

        """
        # Write a backup in case the save fails
        backup_jobs_index = os.path.join(
            self.backup_dir, self.JOBS_INDEX_FILE_NAME + ".backup"
        )
        copyfile(self.jobs_index, backup_jobs_index)
        with open(self.jobs_index, "r+") as f:
            job_index = json.load(f)
            highest_job_id = 0
            for job_id in job_index:
                highest_job_id = max(highest_job_id, int(job_index[job_id]["id"]))

            job_id = str(highest_job_id + 1)
            job_log = None
            if save_history:
                job_log = self._create_job_log(job_id, job.target)
                copyfile(job.log, job_log)

            job_entry = {
                "id": job_id,
                "log": job_log,
                "target": job.target,
                "platform": job.platform,
                "content": job.content,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "tasks": [],
            }

            for workload in job.results_by_workload:
                inp, best_result, tuner_name, trials = job.results_by_workload[workload]
                config_entry_str = record.encode(inp, best_result, "json")
                config_entry = json.loads(config_entry_str)
                config_entry["t"] = [job_id, tuner_name, trials]
                task_entry = json.dumps(config_entry)
                self.save_config(config_entry)
                job_entry["tasks"].append(task_entry)

            job_index[job_id] = job_entry
            f.truncate(0)
            f.seek(0)
            json.dump(job_index, f, indent=4)


    def _create_job_log(self, job_id, target):
        """Returns a path to a job log file.

        This will delete any log that exists with the same name."""
        log_name = job_id + "_" + (
            target.replace(" ", "").replace("-device=", "_").replace("-model=", "_")
        )
        job_log = os.path.join(self.jobs_dir, log_name + ".log")
        if os.path.isfile(job_log):
            os.remove(job_log)

        Path(job_log).touch()
        return job_log

    def save_config(self, new_config):
        """Save a config to the library if it's better than existing entries."""
        target = new_config["i"][0]
        workload = str(new_config["i"][4])
        new_config_key = workload
        config_file = self.get_config_file(target)
        # Write a backup in case the save fails
        backup_config_file = os.path.join(
            self.backup_dir, self._get_target_file_name(target) + ".configs.backup"
        )
        copyfile(config_file, backup_config_file)
        with open(config_file, "r+") as f:
            existing_configs = json.load(f)
            if new_config_key in existing_configs:
                existing_config = existing_configs[new_config_key]
                if np.mean(new_config["r"][0]) < np.mean(existing_config["r"][0]):
                    existing_configs[new_config_key] = new_config
            else:
                existing_configs[new_config_key] = new_config

            f.truncate(0)
            f.seek(0)
            json.dump(existing_configs, f)

    def get_config(self, target, workload):
        """Get a config for a given target/workload from the library.

        Parameters
        ----------
        target: str
            The target string of the config.
        workload: list
            The workload of the config.

        Returns
        -------
        config: Union[dict, None]
            The config for the specified task. Returns None if no config was
            found.

        """
        config_file = self.get_config_file(target, create=False)
        if config_file:
            with open(config_file) as f:
                configs = json.load(f)
                workload_key = str(workload)
                if workload_key in configs:
                    return configs[workload_key]

        return None

    def get_config_file(self, target, create=True):
        """Return the config file path associated with a given target"""
        with open(self.library_index, "r+") as f:
            config_index = json.load(f)

        target_file_name = self._get_target_file_name(target)
        root = config_index["root"]
        config_files = config_index["targets"]
        if target_file_name in config_files:
            return config_files[target_file_name]
        elif create:  # Create the file if it's not already in the index
            with open(self.library_index, "w") as f:
                config_file_name = target_file_name + ".configs"
                config_file = os.path.join(root, config_file_name)
                with open(config_file, "w") as g:
                    json.dump({}, g)

                config_index["targets"][target_file_name] = config_file
                json.dump(config_index, f, indent=4)
                return config_file

        return None

    @staticmethod
    def _get_target_file_name(target):
        """Create a file name from a TVM target string."""
        options = target.split(" ")
        sorted_options = [options[0]] + sorted(options[1:])
        return "-".join(sorted_options)
