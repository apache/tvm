# pylint: disable=superfluous-parens, redefined-outer-name, redefined-outer-name,pointless-string-statement
# pylint: disable=consider-using-enumerate,invalid-name
"""Tuning record and serialization format"""

import argparse
import base64
import logging
import multiprocessing
import pickle
import json
import time
from collections import OrderedDict

import numpy as np

from .. import target, build, lower

from . import task
from .task import DispatchContext, ConfigEntity
from .measure import MeasureInput, MeasureResult

AUTOTVM_LOG_VERSION = 0.1

try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()


def measure_str_key(inp, include_config=True):
    """ get unique str key for MeasureInput

    Parameters
    ----------
    inp: MeasureInput
        input for the measure
    include_config: bool, optional
        whether includes config in the str key

    Returns
    -------
    key: str
        The str representation of key
    """
    config_str = str(inp.config) if include_config else ""
    return "".join([str(inp.target), inp.task.name, str(inp.task.args),
                    str(inp.task.kwargs), config_str])


def encode(inp, result, protocol='json'):
    """encode (MeasureInput, MeasureResult) pair to a string

    Parameters
    ----------
    inp: autotvm.tuner.MeasureInput
    result: autotvm.tuner.MeasureResult
        pair of input/result
    protocol: str
        log protocol, json or pickle

    Returns
    -------
    row: str
        a row in the logger file
    """

    if protocol == 'json':
        json_dict = {
            "i": (str(inp.target),
                  inp.task.name, inp.task.args, inp.task.kwargs,
                  inp.task.workload,
                  inp.config.to_json_dict()),

            "r": (result.costs if result.error_no == 0 else (1e9,),
                  result.error_no,
                  result.all_cost,
                  result.timestamp),

            "v": AUTOTVM_LOG_VERSION
        }
        return json.dumps(json_dict)
    elif protocol == 'pickle':
        row = (str(inp.target),
               str(base64.b64encode(pickle.dumps([inp.task.name,
                                                  inp.task.args,
                                                  inp.task.kwargs,
                                                  inp.task.workload])).decode()),
               str(base64.b64encode(pickle.dumps(inp.config)).decode()),
               str(base64.b64encode(pickle.dumps(tuple(result))).decode()))
        return '\t'.join(row)
    else:
        raise RuntimeError("Invalid log protocol: " + protocol)


def decode(row, protocol='json'):
    """Decode encoded record string to python object

    Parameters
    ----------
    row: str
        a row in the logger file
    protocol: str
        log protocol, json or pickle

    Returns
    -------
    input: autotvm.tuner.MeasureInput
    result: autotvm.tuner.MeasureResult
    """
    # pylint: disable=unused-variable
    if protocol == 'json':
        row = json.loads(row)
        tgt, task_name, task_args, task_kwargs, workload, config = row['i']
        tgt = target.create(str(tgt))

        def clean_json_to_python(x):
            """1. convert all list in x to tuple (hashable)
               2. convert unicode to str for python2
            """
            if isinstance(x, list):
                return tuple([clean_json_to_python(a) for a in x])
            if isinstance(x, _unicode):
                return str(x)
            return x

        tsk = task.Task(clean_json_to_python(task_name), clean_json_to_python(task_args))
        tsk.workload = clean_json_to_python(workload)
        config = ConfigEntity.from_json_dict(config)
        inp = MeasureInput(tgt, tsk, config)
        result = MeasureResult(*[tuple(x) if isinstance(x, list) else x for x in row["r"]])

        return inp, result
    elif protocol == 'pickle':
        items = row.split("\t")
        tgt = target.create(items[0])
        task_tuple = pickle.loads(base64.b64decode(items[1].encode()))
        config = pickle.loads(base64.b64decode(items[2].encode()))
        result = pickle.loads(base64.b64decode(items[3].encode()))

        tsk = task.Task(task_tuple[0], task_tuple[1])
        tsk.workload = task_tuple[3]
        return MeasureInput(tgt, tsk, config), MeasureResult(*result)
    else:
        raise RuntimeError("Invalid log protocol: " + protocol)

def load_from_file(filename):
    """Generator: load records from file.
    This is a generator that yields the records.

    Parameters
    ----------
    filename: str

    Yields
    ------
    input: autotvm.tuner.MeasureInput
    result: autotvm.tuner.MeasureResult
    """
    for row in open(filename):
        yield decode(row)


class ApplyHistoryBest(DispatchContext):
    """
    Apply the history best config

    Parameters
    ----------
    records : str or iterator of (MeasureInput, MeasureResult)
        Collection of tuning records.
        if is str, then it should be the filename of a records log file.
                   Each row of this file is an encoded record pair.
        otherwise, it is an iterator
    default: ConfigEntity, optional
        default config to return when no history records
    """
    def __init__(self, records, default=None):
        super(ApplyHistoryBest, self).__init__()

        if isinstance(records, str):
            records = load_from_file(records)

        counter = 0
        best_map = {}
        for inp, res in records:
            counter += 1
            if res.error_no != 0:
                continue
            for k in inp.target.keys:
                key = (k, inp.task.workload)
                if key not in best_map:
                    best_map[key] = (inp, res)
                else:
                    _, other_res = best_map[key]
                    if np.mean(other_res.costs) > np.mean(res.costs):
                        best_map[key] = (inp, res)
        logging.info(
            "Finish load %d records, %d entries selected", counter, len(best_map))
        self._best_map = best_map
        self._default = default

    def query(self, target, workload):
        if target is None:
            raise RuntimeError("Need a target context to find the history best. "
                               "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                               " above the dispatcher call. So does other target. ")

        for k in target.keys:
            key = (k, workload)
            if key in self._best_map:
                return self._best_map[key][0].config

        if self._default:
            return self._default
        raise RuntimeError(
            "Cannot find config for target=%s, workload=%s" % (target, workload))

    def dump_best(self, out_file):
        """Dump the best records for each workload to a file

        Parameters
        ----------
        out_file: str
            filename
        """
        fout = open(out_file, 'a')
        for val in self._best_map.values():
            inp, res = val
            fout.write(encode(inp, res) + '\n')


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

    logging.info("start convert...")
    pool = multiprocessing.Pool()
    lines = pool.map(decode, lines)
    logging.info("map done %.2f", time.time() - tic)

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
            logging.info("Key: %s\tValid: %d\tDup: %d\t", k, len(cleaned), len(v) - len(cleaned))
            with open(args.i + ".%03d.wkl" % i, 'w') as fout:
                for inp, res in cleaned:
                    fout.write(encode(inp, res) + '\n')
    else:
        for i, (k, v) in enumerate(wkl_dict.items()):
            logging.info("Key: %s\tNum: %d", k, len(v))
            with open(args.i + ".%03d.wkl" % i, 'w') as fout:
                for inp, res in v:
                    fout.write(encode(inp, res) + '\n')


"""
Usage:
This record executable module has three modes.

* Print log file in readable format
e.g. python -m autotvm.record --mode read --i collect_conv.tsv --begin 0 --end 5 --ir --code

* Extract history best from a large log file
e.g. python -m autotvm.record --mode best --i collect.tsv

* Split a log file into separate files, each of which contains only a single wkl
e.g. python -m autotvm.record --mode split --i collect.tsv
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['read', 'best', 'split'], default='read')
    parser.add_argument("--i", type=str, help="input file")
    parser.add_argument("--o", type=str, default=None, help='output file')
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--ir", action='store_true')
    parser.add_argument("--code", action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == 'best':
        args.o = args.o or args.i + ".best"
        hist_best = ApplyHistoryBest(load_from_file(args.i))
        hist_best.dump_best(args.o)
    elif args.mode == 'read':
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
    elif args.mode == 'split':
        split_workload(args.i)
