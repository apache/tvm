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

""" Serialization and other I/O support for tuning logs (measurement records). """

import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .measure import MeasureCallback, MeasureErrorNo
from . import _ffi_api


@tvm._ffi.register_object("ansor.LogToFile")
class LogToFile(MeasureCallback):
    """
    A measurement callback that writes measurement records into a file.

    Parameters
    ----------
    filename : str
        File name for this callback to write log to.
    """
    def __init__(self, filename="ansor_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.LogToFile, filename)


@tvm._ffi.register_object("ansor.LogReader")
class LogReader(Object):
    """
    Reader of the json log file.

    Parameters
    ----------
    filename : str = "ansor_tuning.json"
        File name for this reader to load log from.
    """
    def __init__(self, filename="ansor_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.LogReader, filename)

    def read_lines(self, max_lines=None, skip_lines=None):
        """ Read multiple lines from the log file.

        Parameters
        ----------
        max_lines : Optional[int]
            The maximum number of lines. None to read all lines.
        skip_lines : Optional[int]
            Skip the first n lines. None to read all lines.

        Returns
        -------
        inputs : List[MeasureInput]
            The MeasureInputs loaded from the log file.
        results : List[MeasureResult]
            The MeasureResults loaded from the log file.
        """
        inputs, results = _ffi_api.LogReaderReadLines(self, max_lines if max_lines else -1,
                                                      skip_lines if skip_lines else 0)
        return inputs, results

    def __iter__(self):
        while True:
            ret = _ffi_api.LogReaderReadNext(self)
            if not ret:
                break
            yield ret[0], ret[1]  # (input, result)


def load_from_file(filename):
    """
    Load measurement records from a file.

    Parameters
    ----------
    filename : str
        File name to load log from.

    Returns
    -------
    logs : List[MeasureInput, MeasureResult]
    """
    return zip(*LogReader(filename).read_lines())


def append_measure_records_to_file(filename, inputs, results):
    """
    Append measure records to file.

    Parameters
    ----------
    filename : str
        File name to write log to.
    inputs: List[MeasureInputs]
        The MeasureInputs to be written.
    results: List[MeasureResults]
        The MeasureResults to be written.
    """
    _ffi_api.AppendMeasureRecordsToFile(filename, inputs, results)

def best_measure_pair_in_file(filename, workload_key=None, target=None):
    """ Return the best measurement pair form a log file. This may return none results if
    there is no legal measure pair with the specified workload_key/target found from the log file.

    Parameters
    ----------
    filename : str
        File name to load log from.
    workload_key : Optional[str]
        The workload key of the compute declaration.
        With `None`, this retuns the best measure pair of all workloads.
    target : Optional[tvm.target.Target]
        The target device.
        With `None`, this retuns the best measure pair of all target devices.

    Returns
    -------
    input : MeasureInput
        The best State's MeasureInput from this log fine.
    result : MeasureResult
        The best State's MeasureResult from this log fine.
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
        if target and inp.task.target.id.name != target.id.name:
            continue

        costs = [v.value for v in res.costs]
        cost = np.mean(costs)
        if cost < best_cost:
            best_cost = cost
            best_inp = inp
            best_res = res

    return best_inp, best_res
