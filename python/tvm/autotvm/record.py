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
# pylint: disable=superfluous-parens, redefined-outer-name, redefined-outer-name,pointless-string-statement
# pylint: disable=consider-using-enumerate,invalid-name
"""Tuning record and serialization format"""

import argparse
import base64
import logging
import pickle
import json
import time
import os
import itertools
from collections import OrderedDict
import numpy as np

from .. import build, lower
from ..target import Target
from ..contrib import popen_pool
from .. import __version__
from . import task
from .task import ConfigEntity, ApplyHistoryBest
from .measure import MeasureInput, MeasureResult

AUTOTVM_LOG_VERSION = 0.2
_old_version_warning = True
logger = logging.getLogger("autotvm")

try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()

try:
    _long = long
except NameError:
    _long = int


def measure_str_key(inp, include_config=True):
    """get unique str key for MeasureInput

    Parameters
    ----------
    inp: autotvm.measure.MeasureInput
        input for the measure
    include_config: bool, optional
        whether includes config in the str key

    Returns
    -------
    key: str
        The str representation of key
    """
    config_str = str(inp.config) if include_config else ""
    return "".join(
        [str(inp.target), inp.task.name, str(inp.task.args), str(inp.task.kwargs), config_str]
    )


def encode(inp, result, protocol="json"):
    """encode (MeasureInput, MeasureResult) pair to a string

    Parameters
    ----------
    inp: autotvm.measure.MeasureInput
    result: autotvm.measure.MeasureResult
        pair of input/result
    protocol: str
        log protocol, json or pickle

    Returns
    -------
    row: str
        a row in the logger file
    """

    if protocol == "json":
        json_dict = {
            "input": (str(inp.target), inp.task.name, inp.task.args, inp.task.kwargs),
            "config": inp.config.to_json_dict(),
            "result": (
                result.costs if result.error_no == 0 else (1e9,),
                result.error_no,
                result.all_cost,
                result.timestamp,
            ),
            "version": AUTOTVM_LOG_VERSION,
            "tvm_version": __version__,
        }
        return json.dumps(json_dict)
    if protocol == "pickle":
        row = (
            str(inp.target),
            str(
                base64.b64encode(
                    pickle.dumps([inp.task.name, inp.task.args, inp.task.kwargs])
                ).decode()
            ),
            str(base64.b64encode(pickle.dumps(inp.config)).decode()),
            str(base64.b64encode(pickle.dumps(tuple(result))).decode()),
            str(AUTOTVM_LOG_VERSION),
            str(__version__),
        )
        return "\t".join(row)

    raise RuntimeError("Invalid log protocol: " + protocol)


def decode(row, protocol="json"):
    """Decode encoded record string to python object

    Parameters
    ----------
    row : str
        a row in the logger file

    protocol : str
        log protocol, json or pickle

    Returns
    -------
    ret : tuple(autotvm.measure.MeasureInput, autotvm.measure.MeasureResult), or None
        The tuple of input and result, or None if input uses old version log format.
    """
    # pylint: disable=unused-variable
    global _old_version_warning

    if protocol == "json":
        row = json.loads(row)
        if "v" in row and row["v"] == 0.1:
            if _old_version_warning:
                logger.warning("AutoTVM log version 0.1 is no longer supported.")
                _old_version_warning = False
            return None

        tgt, task_name, task_args, task_kwargs = row["input"]
        tgt = str(tgt)
        if "-target" in tgt:
            logger.warning('"-target" is deprecated, use "-mtriple" instead.')
            tgt = tgt.replace("-target", "-mtriple")
        tgt = Target(str(tgt))

        def clean_json_to_python(x):
            """1. Convert all list in x to tuple (hashable)
            2. Convert unicode to str for python2
            """
            if isinstance(x, list):
                return tuple([clean_json_to_python(a) for a in x])
            if isinstance(x, _unicode):
                return str(x)
            if isinstance(x, (_long, int)):
                return int(x)
            return x

        tsk = task.Task(clean_json_to_python(task_name), clean_json_to_python(task_args))
        config = ConfigEntity.from_json_dict(row["config"])
        inp = MeasureInput(tgt, tsk, config)
        result = MeasureResult(*[tuple(x) if isinstance(x, list) else x for x in row["result"]])
        config.cost = np.mean(result.costs)

        return inp, result
    if protocol == "pickle":
        items = row.split("\t")
        if len(items) == 4:
            if _old_version_warning:
                logger.warning("AutoTVM log version 0.1 is no longer supported.")
                _old_version_warning = False
            return None
        tgt = Target(items[0])
        task_tuple = pickle.loads(base64.b64decode(items[1].encode()))
        config = pickle.loads(base64.b64decode(items[2].encode()))
        result = MeasureResult(*pickle.loads(base64.b64decode(items[3].encode())))
        config.cost = np.mean(result.costs)

        tsk = task.Task(task_tuple[0], task_tuple[1])
        return MeasureInput(tgt, tsk, config), result

    raise RuntimeError("Invalid log protocol: " + protocol)


def load_from_file(filename):
    """Generator: load records from file.
    This is a generator that yields the records.

    Parameters
    ----------
    filename: str

    Yields
    ------
    input: autotvm.measure.MeasureInput
    result: autotvm.measure.MeasureResult
    """
    with open(filename) as f:
        for row in f:
            if row and not row.startswith("#"):
                ret = decode(row)
                if ret is None:
                    continue
                yield ret


def split_workload(in_file, clean=True):
    """Split a log file into separate files, each of which contains only a single workload
    This function can also delete duplicated records in log file

    Parameters
    ----------
    in_file: str
        input filename
    clean: bool
        whether delete duplicated items
    """
    tic = time.time()
    lines = list(open(in_file).readlines())

    logger.info("start converting...")
    pool = popen_pool.PopenPoolExecutor()
    lines = [rec for rec in pool.map(decode, lines) if rec is not None]
    logger.info("map done %.2f", time.time() - tic)

    wkl_dict = OrderedDict()
    for inp, res in lines:
        wkl = measure_str_key(inp, False)
        if wkl not in wkl_dict:
            wkl_dict[wkl] = []
        wkl_dict[wkl].append([inp, res])

    if clean:
        for i, (k, v) in enumerate(wkl_dict.items()):
            # clean duplicated items
            added = set()
            cleaned = []
            for inp, res in v:
                str_key = measure_str_key(inp)
                if str_key in added:
                    continue
                added.add(str_key)
                cleaned.append([inp, res])

            # write to file
            logger.info("Key: %s\tValid: %d\tDup: %d\t", k, len(cleaned), len(v) - len(cleaned))
            with open(args.i + ".%03d.wkl" % i, "w") as fout:
                for inp, res in cleaned:
                    fout.write(encode(inp, res) + "\n")
    else:
        for i, (k, v) in enumerate(wkl_dict.items()):
            logger.info("Key: %s\tNum: %d", k, len(v))
            with open(args.i + ".%03d.wkl" % i, "w") as fout:
                for inp, res in v:
                    fout.write(encode(inp, res) + "\n")


def pick_best(in_file, out_file):
    """
    Pick the best entries from a file and store them to another file.
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
    context = load_from_file(in_file)
    if os.path.isfile(out_file):
        out_context = load_from_file(out_file)
        context = itertools.chain(context, out_context)
    context, context_clone = itertools.tee(context)
    best_context = ApplyHistoryBest(context)
    best_set = set()

    for v in best_context.best_by_model.values():
        best_set.add(measure_str_key(v[0]))

    for v in best_context.best_by_targetkey.values():
        best_set.add(measure_str_key(v[0]))

    logger.info("Extract %d best records from the %s", len(best_set), in_file)
    fout = open(out_file, "w") if isinstance(out_file, str) else out_file

    for inp, res in context_clone:
        if measure_str_key(inp) in best_set:
            fout.write(encode(inp, res) + "\n")
            best_set.remove(measure_str_key(inp))


"""
Usage:
This record executable module has three modes.

* Print log file in readable format
e.g. python -m tvm.autotvm.record --mode read --i collect_conv.log --begin 0 --end 5 --ir --code

* Extract history best from a large log file
e.g. python -m tvm.autotvm.record --mode pick --i collect.log

* Split a log file into separate files, each of which contains only a single wkl
e.g. python -m tvm.autotvm.record --mode split --i collect.log
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["read", "pick", "split"], default="read")
    parser.add_argument("--i", type=str, help="input file")
    parser.add_argument("--o", type=str, default=None, help="output file")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--ir", action="store_true")
    parser.add_argument("--code", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == "pick":
        args.o = args.o or args.i + ".best.log"
        pick_best(args.i, args.o)
    elif args.mode == "read":
        for i, (inp, result) in enumerate(load_from_file(args.i)):
            if args.begin <= i < args.end:
                with inp.target:
                    s, arg_bufs = inp.task.instantiate(inp.config)

                print("")
                print(inp.target, inp.task, inp.config)
                print(result)

                if args.ir:
                    with inp.target:
                        print(lower(s, arg_bufs, simple_mode=True))

                if args.code:
                    with inp.target:
                        func = build(s, arg_bufs)
                        print(func.imported_modules[0].get_source())
    elif args.mode == "split":
        split_workload(args.i)
