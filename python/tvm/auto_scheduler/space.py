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
        self.jfile, self.cfg = cfg, cfg["i"][1][1]
        self.total_dims, self.dims, self.task = 0, [], task
        self.config_space = {}
        self.create_space()

    def create_space(self):
        """Create the space using Ansor's space"""
        sp_space = [4, 8, 16, 24, 32, 48, 64]
        pr_space = [64, 128, 256, 512]
        for i in range(len(self.cfg)):
            f = self.cfg[i]
            if f[0] == "SP" and f[3] != 1:
                for j in range(len(f[4])):
                    self.config_space[f"{f[0]}_{i}_{j}"] = self.add_space(sp_space, [f[4][j]], f[3])
            elif f[0] == "PR":
                start_value = int(f[3].split("$")[-1])
                if start_value != 0:
                    self.config_space[f"{f[0]}_{i}"] = [
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
        jfile = deepcopy(self.jfile)
        cfg = jfile["i"][1][1]
        index = 0
        for i in range(len(cfg)):
            f = cfg[i]
            if f[0] == "SP" and f[3] != 1:
                new_f = []
                for j in range(len(f[4])):
                    new_f.append(self.get_value(f"{f[0]}_{i}_{j}", vals[index]))
                    index += 1
                cfg[i] = ["SP", f[1], f[2], f[3], new_f, f[5]]
            elif f[0] == "PR":
                if f[3] != "auto_unroll_max_step$0":
                    cfg[i] = ["PR", f[1], f[2], self.get_value(f"{f[0]}_{i}", vals[index])]
                    index += 1
        return jfile

    def run(self, log, final_log):
        """Execute a log file and save"""
        readlines, _ = tvm.auto_scheduler.RecordReader(log).read_lines()
        inputs, results = [], []
        for i in range(len(readlines)):
            state = self.task.compute_dag.infer_bound_from_state(readlines[i].state)
            inp = [tvm.auto_scheduler.MeasureInput(self.task, state)]
            build_res = local_builder_build(inp, 20, os.cpu_count(), "default", 0)
            res = local_run(inp, build_res, 20, 3, 3, 0, 0, False, 0, 0)
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
