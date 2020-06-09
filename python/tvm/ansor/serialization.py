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

"""Tuning log I/O Utilities"""

import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .measure import MeasureCallback, MeasureErrorNo
from . import _ffi_api


@tvm._ffi.register_object("ansor.LogToFile")
class LogToFile(MeasureCallback):
    """
    A measurement callback that writes tuning logs into a file

    Parameters
    ----------
    filename : Str
    """

    def __init__(self, filename="ansor_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.LogToFile, filename)


@tvm._ffi.register_object("ansor.LogReader")
class LogReader(Object):
    """
    Reader of the json log file

    Parameters
    ----------
    filename : Str
    """
    def __init__(self, filename="ansor_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.LogReader, filename)

    def read_lines(self, max_size=-1, skip_size=0):
        inputs, results = _ffi_api.LogReaderReadLines(
            self, max_size, skip_size)
        return inputs, results

    def __iter__(self):
        while True:
            ret = _ffi_api.LogReaderReadNext(self)
            if ret is None or not len(ret):
                break
            yield ret[0], ret[1]  # (input, result)


def write_measure_records_to_file(filename, inputs, results):
    """Write(append) measure records to file"""
    _ffi_api.WriteMeasureRecordsToFile(filename, inputs, results)


def get_states_from_measure_inputs(inputs, task):
    """Get states from measure inputs"""
    return _ffi_api.GetStatesFromMeasureInputs(inputs, task)


def best_measure_pair_in_file(filename, workload_key=None, target=None):
    """ Return best results form log file

    Parameters
    ----------
    filename : Str
    workload_key : Str
    target : Str

    Returns
    -------
    inp : MeasureInput
    res : MeasureResult
    """
    log_reader = LogReader(filename)
    best_cost = 1e30
    best_inp = None
    best_res = None

    for inp, res in log_reader:
        if res.error_no != MeasureErrorNo.NO_ERROR:
            continue
        if workload_key and inp.task.workload_key != workload_key:
            continue
        if target and inp.task.target.target_name != target.target_name:
            continue

        costs = []
        for value in res.costs:
            costs.append(value.value)
        cost = np.mean(costs)
        if cost < best_cost:
            best_cost = cost
            best_inp = inp
            best_res = res

    return best_inp, best_res
