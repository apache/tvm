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
# pylint: disable=invalid-name
"""Utilities"""
import logging
import time

import numpy as np
import tvm.arith
from tvm.tir import expr
from tvm.contrib.popen_pool import PopenPoolExecutor

logger = logging.getLogger("autotvm")


class EmptyContext(object):
    """An empty context"""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_rank(values):
    """get rank of items

    Parameters
    ----------
    values: Array

    Returns
    -------
    ranks: Array of int
        the rank of this item in the input (the largest value ranks first)
    """
    tmp = np.argsort(-values)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(tmp))
    return ranks


def pool_map(func, args, batch_size, verbose=False, pool=None):
    """A wrapper of multiprocessing.pool.Pool.map to support small-batch mapping
    for large argument list. This can reduce memory usage

    Parameters
    ----------
    func: Func(arg) -> np.ndarray
        mapping function
    args: List
        list of arguments
    batch_size: int
        batch size in mapping
    verbose: bool, optional
        whether print progress
    pool: multiprocessing.Pool, optional
        pool objection

    Returns
    -------
    converted numpy array
    """

    ret = None
    tic = time.time()
    local_pool = pool or PopenPoolExecutor()
    if verbose:
        logger.info("mapping begin")
    for i in range(0, len(args), batch_size):
        if verbose:
            logger.info("mapping %d/%d elapsed %.2f", i, len(args), time.time() - tic)
        tmp = np.array(local_pool.map(func, args[i : i + batch_size]))
        ret = tmp if ret is None else np.concatenate((ret, tmp))
    if verbose:
        logger.info("mapping done")
    if not pool:
        local_pool.close()
    return ret


def get_func_name(func):
    """Get name of a function

    Parameters
    ----------
    func: Function
        The function
    Returns
    -------
    name: str
        The name
    """

    return func.func_name if hasattr(func, "func_name") else func.__name__


def get_const_int(exp):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    exp : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(exp, int):
        return exp
    if not isinstance(exp, (expr.IntImm,)):
        ana = tvm.arith.Analyzer()
        exp = ana.simplify(exp)
    if not isinstance(exp, (expr.IntImm,)):
        raise ValueError("Expect value to be constant int")
    return exp.value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    for elem in in_tuple:
        if isinstance(elem, expr.Var):
            ret.append(elem)
        elif not isinstance(elem, (expr.IntImm, int)):
            ana = tvm.arith.Analyzer()
            elem = ana.simplify(elem)
            if not isinstance(elem, (expr.IntImm)):
                ret.append(elem)
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


SI_PREFIXES = "yzafpn\xb5m kMGTPEZY"
YOCTO_EXP10 = -24


def format_si_prefix(x, si_prefix):
    exp10 = 10 ** (SI_PREFIXES.index(si_prefix) * 3 + YOCTO_EXP10)
    return float(x) / exp10
