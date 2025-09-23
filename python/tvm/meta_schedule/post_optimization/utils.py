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
"""Utils file for exploitation schedule"""

import os
import json
from typing import Dict
import numpy as np  # type: ignore


def write_file(json_list: list, log: str = "/tmp/file.json", mode: str = "w") -> str:
    """Write the log file

    Parameters
    ----------
    json_list: list
        The list input json
    log: Optional[str]
        Path destiny to save the log file
    mode: Optional[str]
        Mode save, "a" means append and "w" means write

    Returns
    -------
    ret: str
        log path file
    """
    with open(log, mode, encoding="utf-8") as outfile:
        for j in json_list:
            outfile.write(json.dumps(j) + "\n")
    return log


def clean_file(filename: str) -> None:
    """Clean temporary files

    Parameters
    ----------
    filename: str
        The filepath with remove from the system
    """
    if os.path.isfile(filename):
        os.remove(filename)


def get_time(log: str) -> list:
    """Get the time from the log file

    Parameters
    ----------
    log: str
        log file

    Returns
    -------
    ret: list
        A list with the best time and the json data
    """
    best_time = [1e10, None]
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            time = params[1]
            if np.mean(best_time[0]) > np.mean(time):
                best_time = [time, data]
    return best_time


def read_cfg_file(path_tuning_file: str, path_workload_file: str) -> Dict[int, list]:
    """Colect the info from meta logfile

    Parameters
    ----------
    log: str
        The input log path with the meta parameter

    Returns
    -------
    ret: dict[layer, Union[time, dict]]
        Returns the best time, total time, and data
    """
    workload_list = []
    with open(path_workload_file, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            workload_list.append(json.loads(line))

    cfg: Dict[int, list] = dict()
    with open(path_tuning_file, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            layer = data[0]
            params = data[1]
            time = params[1]

            if layer not in cfg.keys() or np.mean(cfg[layer][0]) > np.mean(time):
                cfg[layer] = [time, data, workload_list[layer]]
    return cfg
