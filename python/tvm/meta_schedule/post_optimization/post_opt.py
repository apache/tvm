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
"""Post optimization method"""

import numpy as np  # type: ignore

from tvm.target import Target

from .droplet import Droplet
from .utils import clean_file, get_time, read_cfg_file, write_file

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_layer(layer, idx, data, workload, target, work_dir, trials):
    """
    处理单个层的后优化任务：
      1. 生成临时日志文件路径
      2. 创建 Droplet 对象并执行 tune
      3. 从日志文件中获取最佳解
      4. 清理临时日志文件
    返回值：
      dpm_sol: 该层获得的最佳解数据
    """
    temp_log = f"{work_dir}/opt_{idx}.log"
    # 创建 Droplet 对象，并执行 tune
    droplet = Droplet(data, workload, target, temp_log)
    droplet.tune(trials)
    # 提取日志中的最佳解
    dpms_time, dpm_sol = get_time(temp_log)
    # 清理临时日志文件
    clean_file(temp_log)
    return dpm_sol


class PostOpt:
    """PostOpt class
    Parameters
    ----------
    work_dir : str
        The working directory.
    target: Target data
        Target device information
    trials: integer value
        Max number of trials to execute the optimization
    """

    def __init__(
        self, work_dir: str, target: Target, trials: int = 100
    ) -> None:
        self.work_dir = work_dir
        self.target = target
        self.trials = trials

    def run(self) -> None:
        """Execute the post optimization"""

        tuning_file = self.work_dir + "/database_tuning_record.json"
        workload_file = self.work_dir + "/database_workload.json"
        cfg = read_cfg_file(tuning_file, workload_file)
        for idx, layer in enumerate(cfg):

            time, data, workload = cfg[layer]
            ms_time = np.mean(time)

            temp_log = f"{self.work_dir}/opt_{idx}.log"

            # Run the exploitation by Droplet
            droplet = Droplet(data, workload, self.target, temp_log)
            droplet.tune(self.trials)

            dpms_time, dpm_sol = get_time(temp_log)
            # save the best solution
            write_file([dpm_sol], tuning_file, mode="a")
            # clean the temporary files
            clean_file(temp_log)
