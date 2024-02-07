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

""" The class of Space used to optimize the Ansor parameters """

import os
from copy import deepcopy
import tvm
from tvm.auto_scheduler.measure import local_builder_build, local_run


class MeasureResultSpace:
    """Store the results of a measurement.

    Parameters
    ----------
    measureResult: List[MeasureResult]
        A List of MeasureResult.
    """

    def __init__(self, measure_result):
        self._costs = measure_result[0].costs

    @property
    def costs(self):
        return [v.value for v in self._costs]


class Space:
    """Space class

    Parameters
    ----------
    cfg: json data
        A json file template
    task: SearchTask
        The SearchTask of this measurement.
    """

    def __init__(self, cfg, task):
        self.cfg = deepcopy(cfg)
        self.total_dims, self.dims, self.task = 0, [], task
        self.config_space = {}
        self.create_space()

    def create_space(self):
        """Create the space using Ansor's space"""
        sp_space = [4, 8, 16, 24, 32, 48, 64]
        pr_space = [64, 128, 256, 512]
        idx_sp, idx_pos, idx_size, idx_tile = 0, 1, 3, 4
        config = self.cfg["i"][idx_pos][idx_pos]
        for i in range(len(config)):
            opt = config[i]
            if opt[idx_sp] == "SP" and opt[idx_size] != 1:
                for j in range(len(opt[idx_tile])):
                    self.config_space[f"{opt[idx_sp]}_{i}_{j}"] = self.add_space(
                        sp_space, [opt[idx_tile][j]], opt[idx_size]
                    )
            elif opt[idx_sp] == "PR":
                start_value = int(opt[idx_size].split("$")[-1])
                if start_value != 0:
                    self.config_space[f"{opt[idx_sp]}_{i}"] = [
                        f"auto_unroll_max_step${v}" for v in self.add_space(pr_space, [start_value])
                    ]
        self.dims = []
        for key in self.config_space:
            self.dims.append(len(self.config_space[key]))
        self.total_dims = 1
        if len(self.dims) > 0:
            for dim in self.dims:
                self.total_dims *= dim

    def apply_opt(self, vals):
        """Apply the space using Ansor's space"""
        idx_sp, idx_pos, idx_size, idx_tile = 0, 1, 3, 4
        index, config = 0, self.cfg["i"][idx_pos][idx_pos]
        for i in range(len(config)):
            opt = config[i]
            if opt[idx_sp] == "SP" and opt[idx_size] != 1:
                new_value = []
                for j in range(len(opt[idx_tile])):
                    new_value.append(self.get_value(f"{opt[idx_sp]}_{i}_{j}", vals[index]))
                    index += 1
                config[i][idx_tile] = new_value
            elif opt[idx_sp] == "PR":
                if opt[idx_size] != "auto_unroll_max_step$0":
                    config[i][idx_size] = self.get_value(f"{opt[idx_sp]}_{i}", vals[index])
                    index += 1
        return self.cfg

    def run(
        self,
        log,
        final_log,
        timeout=20,
        verbose=0,
        number=3,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0,
        cache=False,
        dev=0,
    ):
        """Execute a log file and save"""
        readlines, _ = tvm.auto_scheduler.RecordReader(log).read_lines()
        inputs, results = [], []
        for i in range(len(readlines)):
            state = self.task.compute_dag.infer_bound_from_state(readlines[i].state)
            inp = [tvm.auto_scheduler.MeasureInput(self.task, state)]
            build_res = local_builder_build(inp, timeout, os.cpu_count(), "default", verbose)
            res = local_run(
                inp,
                build_res,
                timeout,
                number,
                repeat,
                min_repeat_ms,
                cooldown_interval,
                cache,
                verbose,
                dev,
            )
            tvm.auto_scheduler._ffi_api.SaveRecords(final_log, inp, res)
            inputs.append(inp[0])
            results.append(MeasureResultSpace(res))
        return inputs, results

    def get_value(self, key, pos):
        """Return the space"""
        return self.config_space[key][pos]

    def add_space(self, space_list, element, limit=10000):
        """Return a list without repeat and with limited value"""
        new_list = element
        for elem in space_list:
            if elem not in new_list and elem <= limit:
                new_list.append(elem)
        return new_list

    def knob2point(self, arr):
        """Convert a array to point"""
        value = 0
        for i in range(len(arr) - 1):
            value += arr[i] * self.dims[i]
        value += arr[-1]
        return value
