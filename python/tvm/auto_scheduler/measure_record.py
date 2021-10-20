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
# pylint: disable=invalid-name, pointless-string-statement

""" Serialization and other I/O support for measurement records (tuning logs). """
import argparse
import logging
import os
import itertools

import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .measure import MeasureErrorNo, MeasureCallback
from .utils import calc_workload_dis_factor, decode_workload_key
from . import _ffi_api

logger = logging.getLogger("auto_scheduler")


@tvm._ffi.register_object("auto_scheduler.RecordToFile")
class RecordToFile(MeasureCallback):
    """
    A measurement callback that writes measurement records into a file.

    Parameters
    ----------
    filename : str
        File name for this callback to write log to.
    """

    def __init__(self, filename):
        dirname = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.__init_handle_by_constructor__(_ffi_api.RecordToFile, filename)


@tvm._ffi.register_object("auto_scheduler.RecordReader")
class RecordReader(Object):
    """
    Reader of the json log file.

    Parameters
    ----------
    filename : str
        File name for this reader to load log from.
    """

    def __init__(self, filename):
        if not os.path.exists(filename):
            logger.warning("%s does not exist!", filename)
        # a set to prevent print duplicated message
        self.messages = set()
        self.__init_handle_by_constructor__(_ffi_api.RecordReader, filename)

    def check_workload_key(self, inputs):
        """Check and throw warnings for records with old format workload key.

        Parameters
        ----------
        inputs: List[MeasureInput]
            The measure inputs to be checked.

        Notes
        -----
        This checker could be deprecated in the future.
        """
        for inp in inputs:
            _, args = decode_workload_key(inp.task.workload_key)
            if args is None:
                continue
            if not args:
                msg = (
                    "MeasureInput with old format workload key %s should be updated "
                    "using the script from https://github.com/apache/tvm/pull/7317."
                    % inp.task.workload_key
                )
                if msg not in self.messages:
                    self.messages.add(msg)
                    logger.warning(msg)

    def read_lines(self, max_lines=None, skip_lines=0):
        """Read multiple lines from the log file.

        Parameters
        ----------
        max_lines : Optional[int]
            The maximum number of lines. None to read all lines.
        skip_lines : int = 0
            Skip the first n lines.

        Returns
        -------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The MeasureInputs loaded from the log file.
        results : List[auto_scheduler.measure.MeasureResult]
            The MeasureResults loaded from the log file.

        Notes
        -----
        Some unimportant and expensive fields in the returned MeasureInput are not deserialized
        for faster read speed (e.g. input.task.compute_dag, input.state.stages).
        If you want to use them, you can call the :code:`recover_measure_input` below
        to rebuild these fields.
        """
        inputs, results = _ffi_api.RecordReaderReadLines(
            self, max_lines if max_lines else -1, skip_lines
        )
        self.check_workload_key(inputs)
        return inputs, results

    def __iter__(self):
        while True:
            ret = _ffi_api.RecordReaderReadNext(self)
            if not ret:
                break
            self.check_workload_key([ret[0]])
            yield ret[0], ret[1]  # (input, result)


def load_record_from_string(record):
    """
    Load the measure record from string.

    Parameters
    ----------
    record: str
        A record string, including the serialized MeausreInput and MeasureResult.

    Returns
    -------
    ret: Tuple[MeasureInput, MeasureResult]
        A tuple of MeasureInput, MeasureResult.
    """
    return _ffi_api.ReadMeasureRecord(record)


def dump_record_to_string(inp, res):
    """
    Dump the measure record to a string.

    Parameters
    ----------
    inp: MeasureInput
        The measure input.

    res: MeasureResult
        The measure result.

    Returns
    -------
    ret: str
        The dumped string.
    """
    return _ffi_api.WriteMeasureRecords(inp, res)


def load_records(filename):
    """
    Load measurement records from a file.

    Parameters
    ----------
    filename : str
        File name to load log from.

    Returns
    -------
    logs : List[auto_scheduler.measure.MeasureInput, auto_scheduler.measure.MeasureResult]

    Notes
    -----
    Some unimportant and expensive fields in the returned MeasureInput are not deserialized
    for faster read speed (e.g., input.task.compute_dag, input.state.stages).
    If you want to use them, you can call the :code:`recover_measure_input` below
    to rebuild these fields.
    """
    return zip(*RecordReader(filename).read_lines())


def save_records(filename, inputs, results):
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
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    _ffi_api.SaveRecords(filename, inputs, results)


def load_best_record(filename, workload_key=None, target=None, include_compatible=False):
    """Return the best measurement pair form a log file. This may return none results if
    there is no legal measure pair with the specified workload_key/target found from the log file.

    Parameters
    ----------
    filename : str
        File name to load log from.
    workload_key : Optional[str]
        The workload key of the compute declaration.
        With `None`, this returns the best measure pair of all workloads.
    target : Optional[tvm.target.Target]
        The target device.
        With `None`, this returns the best measure pair of all target devices.
    include_compatible: bool
        When set to True, all compatible records in the log file will be considered.

    Returns
    -------
    input : auto_scheduler.measure.MeasureInput
        The best State's MeasureInput from this log fine.
    result : auto_scheduler.measure.MeasureResult
        The best State's MeasureResult from this log fine.
    """
    log_reader = RecordReader(filename)
    best_cost = 1e30
    best_inp = None
    best_res = None

    for inp, res in log_reader:
        if res.error_no != MeasureErrorNo.NO_ERROR:
            continue
        if target and inp.task.target.kind.name != target.kind.name:
            continue

        costs = [v.value for v in res.costs]
        cost = np.mean(costs)

        if workload_key is not None:
            dis_f = calc_workload_dis_factor(
                decode_workload_key(workload_key), decode_workload_key(inp.task.workload_key)
            )
            if dis_f == float("inf"):
                continue
            if not include_compatible and dis_f != 1:
                continue

            # Since different workloads have different FLOPS, we multiply the factor to
            # eliminate this difference, which is basically the concept of throughput.
            cost *= dis_f

        if cost < best_cost:
            best_cost = cost
            best_inp = inp
            best_res = res

    return best_inp, best_res


def distill_record_file(in_file, out_file):
    """
    Pick the best entries from a record file and store them to another file.
    This function distills the useful log entries from a large log file.
    If out_file already exists, the best entries from both
    in_file and out_file will be saved.

    Parameters
    ----------
    in_file: str
        The filename of input
    out_file: str or file
        The filename of output
    """
    # pylint: disable=import-outside-toplevel
    from .dispatcher import ApplyHistoryBest

    context = load_records(in_file)

    dirname = os.path.dirname(os.path.abspath(out_file))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if os.path.isfile(out_file):
        out_context = load_records(out_file)
        context = itertools.chain(context, out_context)

    def measure_input_str_key(inp):
        return _ffi_api.SerializeMeasureInput(inp)

    # Dict[target key,
    #   Dict[workload hash,
    #     Dict[workload args, (cost, (MeasureInput, MeasureResult))]]]
    # Full type: Dict[str, Dict[str, Dict[Tuple, Tuple[float, Tuple[Measureinput, MeasureResult]]]]]
    best_records = {}

    for inp, res in context:
        if res.error_no != 0:
            continue

        # Keep the best record for each target and workload.
        costs = [x.value for x in res.costs if isinstance(x, tvm.tir.expr.FloatImm)]
        cost = np.mean(costs)
        for k in inp.task.target.keys:
            entry, _, workload_args = ApplyHistoryBest.get_workload_entry(
                best_records, k, inp.task.workload_key
            )
            if workload_args not in entry or cost < entry[workload_args][0]:
                entry[workload_args] = (cost, (inp, res))

    # Remove duplications by multiple target keys.
    out_records = {}
    for target_entry in best_records.values():
        for workload_entry in target_entry.values():
            for _, (inp, res) in workload_entry.values():
                out_records[measure_input_str_key(inp)] = (inp, res)

    inputs = []
    results = []
    for inp, res in out_records.values():
        inputs.append(inp)
        results.append(res)

    # create a new file and save the best records
    open(out_file, "w")
    save_records(out_file, inputs, results)
    logger.info("Extract %d best records from %s to %s", len(inputs), in_file, out_file)


def main():
    """The main function for CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["distill"], default="distill")
    parser.add_argument("-i", "--input", type=str, help="input file")
    parser.add_argument("-o", "--output", type=str, default=None, help="output file")

    args = parser.parse_args()
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    if args.mode == "distill":
        args.output = args.output or args.input + ".best.json"
        distill_record_file(args.input, args.output)


"""
Usage:
* Distill the best entries from a large log file
e.g. python -m tvm.auto_scheduler.measure_record --mode distill -i input.json
"""
if __name__ == "__main__":
    main()
