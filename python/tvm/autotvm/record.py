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
import os
from collections import OrderedDict

import numpy as np

from .. import build, lower, target as _target

from . import task
from .task import DispatchContext, ConfigEntity
from .measure import MeasureInput, MeasureResult

AUTOTVM_LOG_VERSION = 0.1

try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()

try:
    _long = long
except NameError:
    _long = int


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
        tgt = _target.create(str(tgt))

        def clean_json_to_python(x):
            """1. convert all list in x to tuple (hashable)
               2. convert unicode to str for python2
            """
            if isinstance(x, list):
                return tuple([clean_json_to_python(a) for a in x])
            if isinstance(x, _unicode):
                return str(x)
            if isinstance(x, (_long, int)):
                return int(x)
            return x

        tsk = task.Task(clean_json_to_python(task_name), clean_json_to_python(task_args))
        tsk.workload = clean_json_to_python(workload)
        config = ConfigEntity.from_json_dict(config)
        inp = MeasureInput(tgt, tsk, config)
        result = MeasureResult(*[tuple(x) if isinstance(x, list) else x for x in row["r"]])

        return inp, result
    elif protocol == 'pickle':
        items = row.split("\t")
        tgt = _target.create(items[0])
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
        If is str, then it should be the filename of a records log file.
                   Each row of this file is an encoded record pair.
        Otherwise, it is an iterator.
    default: ConfigEntity, optional
        The default config to return when no history records
    """
    def __init__(self, records, default=None):
        super(ApplyHistoryBest, self).__init__()

        self.best_by_targetkey = {}
        self.best_by_model = {}
        self._default = default

        self.load(records)

    def load(self, records):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (MeasureInput, MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
                       Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.
        """
        if isinstance(records, str):
            records = load_from_file(records)
        if not records:
            return

        best_by_targetkey = self.best_by_targetkey
        best_by_model = self.best_by_model

        counter = 0
        for inp, res in records:
            counter += 1
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for k in inp.target.keys:
                key = (k, inp.task.workload)
                if key not in best_by_targetkey:
                    best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = best_by_targetkey[key]
                    if np.mean(other_res.costs) > np.mean(res.costs):
                        best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            for opt in inp.target.options:
                if opt.startswith("-model"):
                    model = opt[7:]
                    key = (model, inp.task.workload)
                    if key not in best_by_model:
                        best_by_model[key] = (inp, res)
                    else:
                        _, other_res = best_by_model[key]
                        if np.mean(other_res.costs) > np.mean(res.costs):
                            best_by_model[key] = (inp, res)
                    break

        logging.info("Finish loading %d records", counter)

    def query(self, target, workload):
        if target is None:
            raise RuntimeError("Need a target context to find the history best. "
                               "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                               " above the dispatcher call. So does other target. ")

        # first try matching by model
        for opt in target.options:
            if opt.startswith("-model"):
                model = opt[7:]
                key = (model, workload)
                if key in self.best_by_model:
                    return self.best_by_model[key][0].config

        # then try matching by target key
        for k in target.keys:
            key = (k, workload)
            if key in self.best_by_targetkey:
                return self.best_by_targetkey[key][0].config

        if self._default:
            return self._default
        raise RuntimeError(
            "Cannot find config for target=%s, workload=%s" % (target, workload))


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

    logging.info("start converting...")
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

def pick_best(in_file, out_file):
    """
    Pick best entries from a file and store it to another file.
    This distill the useful log entries from a large log file.

    Parameters
    ----------
    in_file: str
        The filename of input
    out_file:
        The filename of output
    """
    best_context = ApplyHistoryBest(load_from_file(in_file))
    best_set = set()

    for v in best_context.best_by_model.values():
        best_set.add(measure_str_key(v[0]))

    for v in best_context.best_by_targetkey.values():
        best_set.add(measure_str_key(v[0]))

    logging.info("Extract %d best records from the log file", len(best_set))

    fout = open(out_file, 'w')
    for inp, res in load_from_file(in_file):
        if measure_str_key(inp) in best_set:
            fout.write(encode(inp, res) + "\n")


def load_op_param(rootpath=os.path.join(os.path.expanduser('~'), ".tvm", "op_params")):
    """Load pre-tuned parameters of operators.
    This function will load all "*.log" file under root path and select best configs.

    Parameters
    ----------
    rootpath: str
        The root path of stored parameters
    """
    best_context = ApplyHistoryBest([])
    for dirpath, _, filenames in os.walk(rootpath):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.log':
                best_context.load(os.path.join(dirpath, filename))

    assert not DispatchContext.current, "Cannot load pre-tuned parameters inside a dispatch context"
    DispatchContext.current = best_context

"""
Usage:
This record executable module has three modes.

* Print log file in readable format
e.g. python -m autotvm.record --mode read --i collect_conv.log --begin 0 --end 5 --ir --code

* Extract history best from a large log file
e.g. python -m autotvm.record --mode pick --i collect.log

* Split a log file into separate files, each of which contains only a single wkl
e.g. python -m autotvm.record --mode split --i collect.log
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['read', 'pick', 'split'], default='read')
    parser.add_argument("--i", type=str, help="input file")
    parser.add_argument("--o", type=str, default=None, help='output file')
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--ir", action='store_true')
    parser.add_argument("--code", action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == 'pick':
        args.o = args.o or args.i + ".best.log"
        pick_best(args.i, args.o)
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
