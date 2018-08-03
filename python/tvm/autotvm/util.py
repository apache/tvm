# pylint: disable=invalid-name
"""Utilities"""
import logging
import multiprocessing
import time

import numpy as np

from .. import expr, ir_pass

logger = logging.getLogger('autotvm')

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


def sample_ints(low, high, m):
    """
    Sample m different integer numbers from [low, high) without replacement
    This function is an alternative of `np.random.choice` when (high - low) > 2 ^ 32, in
    which case numpy does not work.

    Parameters
    ----------
    low: int
        low point of sample range
    high: int
        high point of sample range
    m: int
        The number of sampled int

    Returns
    -------
    ints: an array of size m
    """
    vis = set()
    assert m <= high - low
    while len(vis) < m:
        new = np.random.randint(low, high)
        while new in vis:
            new = np.random.randint(low, high)
        vis.add(new)

    return list(vis)


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
    local_pool = pool or multiprocessing.Pool()
    if verbose:
        logger.info("mapping begin")
    for i in range(0, len(args), batch_size):
        if verbose:
            logger.info("mapping %d/%d elapsed %.2f", i, len(args),
                        time.time() - tic)
        tmp = np.array(local_pool.map(func, args[i:i+batch_size]))
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

    return func.func_name if hasattr(func, 'func_name') else func.__name__


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
    if not isinstance(exp, (expr.IntImm, expr.UIntImm)):
        exp = ir_pass.Simplify(expr)
    if not isinstance(exp, (expr.IntImm, expr.UIntImm)):
        raise ValueError("Expect value to be constant int")
    return exp.value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    return tuple(get_const_int(x) for x in in_tuple)
