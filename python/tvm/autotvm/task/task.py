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
# pylint: disable=unused-variable
"""Definition of task function.

Task can be constructed from tuple of func, args, and kwargs.
func is a state-less function, or a string that
registers the standard task.
"""

import numpy as np

from ... import tensor, expr, container, target as _target

from ..util import get_const_int, get_const_tuple, get_func_name
from .dispatcher import DispatchContext, ApplyConfig, dispatcher
from .space import ConfigSpace

def _raise_error(*args, **kwargs):  # pylint: disable=unused-argument
    raise RuntimeError("The function of this task is not found. Possibly the function "
                       "of this task is registered in another python file "
                       "which is not imported in this run")

class Task(object):
    """A Tunable Task

    Parameters
    ----------
    name: str
        The name of the task.
    args: Tuple
        Positional argument of func
    """
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.kwargs = {}  # currently unused

        # init null config space
        self.config_space = None
        self.func = TASK_TABLE.get(name, _raise_error)

        # auxiliary info, available after `init_space` is called
        self.workload = None
        self.flop = None
        self.target = None
        self.target_host = None

    def instantiate(self, config):
        """Instantiate this task function (template) with a config.
        Returns corresponding schedule.

        Parameters
        ----------
        config: template.ConfigEntity
            parameter config for this template

        Returns
        -------
        sch: tvm.schedule.Schedule
            The tvm schedule
        arg_bufs: Array of tvm.tensor.Tensor
            The input/output buffers
        """
        config.flop = 0
        with ApplyConfig(config):
            sch, arg_bufs = self.func(*self.args, **self.kwargs)
        if not self.flop:
            config.flop = config.flop or compute_flop(sch)
            self.flop = config.flop
        return sch, arg_bufs

    def __getstate__(self):
        # custom pickle implementation is required for
        # some unpickable local task functions.
        # So we only pickle the name of the function
        # and restore the function by name when unpickling it.
        return {
            "name": self.name,
            "args": self.args,
            "kwargs": self.kwargs,
            "config_space": self.config_space,
            "workload": self.workload,
            "flop": self.flop,
            "target": self.target,
            "target_host": self.target_host
        }

    def __setstate__(self, state):
        self.name = state["name"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]
        self.config_space = state["config_space"]
        self.func = TASK_TABLE.get(state["name"], _raise_error)
        self.workload = state["workload"]
        self.flop = state["flop"]
        self.target = state["target"]
        self.target_host = state["target_host"]

    def __repr__(self):
        return "Task(func_name=%s, args=%s, kwargs=%s, workload=%s)" % (
            self.name, self.args, self.kwargs, self.workload
        )

TASK_TABLE = {
}

def register(name, func=None, override=False):
    """Register a task function.

    Parameters
    ----------
    name : str
        The name to identify the task.
    func : callable
        The function to be registered.
    override : bool
        Whether override existing registration.

    Returns
    -------
    func: callable
        The registered function
    """
    def _do_reg(myf):
        if name in TASK_TABLE and not override:
            raise ValueError(
                "Key %s is already registered" % name)
        TASK_TABLE[name] = myf
        return myf
    if func:
        return _do_reg(func)
    return _do_reg

def create(func_name, args, target, target_host=None, template_key=None):
    """Create a tuning task and initialize its search space

    Parameters
    ----------
    func_name : str or callable
        The task function
    args : List
        Positional arguments
    target : Target
        The compilation target
    target_host: Target, optional
        The compilation target for host side

    Returns
    -------
    tsk: Task
        a task object
    """
    if callable(func_name):
        # register this function if it is not registered before
        func = func_name
        func_name = func.func_name if hasattr(func, 'func_name') else func.__name__
        if func_name in TASK_TABLE:
            assert func == TASK_TABLE[func_name], "Find name conflict in task registration. " \
                                                  "Consider to choose another name for this task"
        else:
            register(func_name, func=func)

    func = TASK_TABLE[func_name]
    ret = Task(func_name, args)

    if isinstance(target, str):
        target = _target.create(target)

    # init config space
    ret.config_space = ConfigSpace()
    ret.config_space.template_key = template_key or ""

    ctx = ApplyConfig(ret.config_space)
    with ctx:
        with target:
            sch, _ = func(*args)
            ret.config_space.code_hash = getattr(sch, 'code_hash', None)

    ret.workload = ctx.workload
    ret.flop = ret.config_space.flop or compute_flop(sch)
    ret.target = target
    ret.target_host = target_host

    return ret

def args_to_workload(x, topi_compute_func=None):
    """Convert argument list to hashable workload tuple.
    This function will convert list to tuple, tvm node to python value and
    flatten tvm.tensor.Tensor to a tuple

    Parameters
    ----------
    x: primitive hashable types or tensor.Tensor
        The original value
    topi_compute_func: topi compute function
        The function name will be added as first element of the workload tuple

    Returns
    -------
    ret: hashable
        The hashable value
    """
    if isinstance(x, tensor.Tensor):
        workload = get_const_tuple(x.shape) + (x.dtype, )
    elif isinstance(x, (tuple, list, container.Array)):
        workload = tuple([args_to_workload(a) for a in x])
    elif isinstance(x, (str, int, float, np.int, np.float)):
        workload = x
    elif isinstance(x, (expr.StringImm, expr.UIntImm, expr.IntImm, expr.FloatImm)):
        workload = x.value
    elif x is None:
        workload = 0
    else:
        raise RuntimeError('Do not support type "%s" in argument. Consider to use'
                           'primitive types only' % type(x))
    return (get_func_name(topi_compute_func), ) + workload  if topi_compute_func else workload

def template(func):
    """
    Decorate a function as a tunable schedule template

    Parameters
    ----------
    func: callable
        A callable template function.
        Its argument should be hashable values.
        Its return value should be a Tuple(Schedule, Array of Tensor)

    Returns
    -------
    func: callable
        The decorated function

    Examples
    --------
    The following code is a tunable template for a blocked matrix multiplication

    .. code-block:: python

        @autotvm.template
        def matmul(N, L, M, dtype):
            A = tvm.placeholder((N, L), name='A', dtype=dtype)
            B = tvm.placeholder((L, M), name='B', dtype=dtype)

            k = tvm.reduce_axis((0, L), name='k')
            C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
            s = tvm.create_schedule(C.op)

            # schedule
            y, x = s[C].op.axis
            k = s[C].op.reduce_axis[0]

            ##### define space begin #####
            cfg = autotvm.get_config()
            cfg.define_split("tile_y", y, num_outputs=2)
            cfg.define_split("tile_x", x, num_outputs=2)
            ##### define space end #####

            # schedule according to config
            yo, yi = cfg["tile_y"].apply(s, C, y)
            xo, xi = cfg["tile_x"].apply(s, C, x)

            s[C].reorder(yo, xo, k, yi, xi)

            return s, [A, B, C]
    """
    # pylint: disable=unused-variable

    fname = get_func_name(func)

    @register(fname)
    @dispatcher
    def config_dispatcher(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        return (fname, ) + args_to_workload(args)

    @config_dispatcher.register("")
    def template_call(cfg, *args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        with ApplyConfig(cfg):
            return func(*args, **kwargs)

    config_dispatcher.func_name = fname
    return config_dispatcher

def get_config():
    """Get current config object

    Returns
    -------
    cfg: ConfigSpace or ConfigEntity
        The current config
    """
    return DispatchContext.current.query(None, None)

class FlopCalculationError(RuntimeError):
    """Error happens when estimating FLOP for a compute op"""


def compute_flop(sch):
    """Calculate number of FLOP (floating number operations) of the compute ops in a schedule

    Parameters
    ----------
    sch: tvm.schedule.Schedule
        schedule

    Returns
    -------
    flop: int
        number of FLOP in this schedule
    """
    def _prod_length(axes):
        """compute product of the lengths of a list of axes"""
        try:
            num_iter = int(np.prod([get_const_int(axis.dom.extent) for axis in axes]))
        except ValueError:
            raise FlopCalculationError("The length of axis is not constant. ")
        return num_iter

    def _count_flop(exp):
        """compute flop for a single expression"""
        if isinstance(exp, expr.Reduce):
            num_iter = _prod_length(exp.axis)
            combiner = exp.combiner.result
            source = exp.source
            if len(combiner) != 1:
                raise FlopCalculationError("Found multiple output in the combiner of reduce op")
            if len(source) != 1:
                raise FlopCalculationError("Found multiple output in the source of reduce op")
            return num_iter * (_count_flop(combiner[0]) + _count_flop(source[0]))
        if isinstance(exp, (expr.FloatImm, expr.IntImm, expr.UIntImm)):
            return 0
        if isinstance(exp, expr.Cast):
            return _count_flop(exp.value)
        if isinstance(exp, expr.Var):
            return 0
        if isinstance(exp, (expr.Add, expr.Sub, expr.Mul, expr.Div, expr.Mod,
                            expr.Max, expr.Min,
                            expr.EQ, expr.NE, expr.LT, expr.LE, expr.GT, expr.GE,
                            expr.And, expr.Or, expr.Not)):
            base = 1

            if isinstance(exp, expr.Not):  # unary
                return base + _count_flop(exp.a)

            return base + _count_flop(exp.a) + _count_flop(exp.b)
        if isinstance(exp, expr.Select):
            return _count_flop(exp.condition) + max(_count_flop(exp.true_value),
                                                    _count_flop(exp.false_value))
        if isinstance(exp, expr.Call):
            if exp.call_type == expr.Call.Halide:
                # Ignore flops from indexing expressions.
                return 0

            return sum([_count_flop(x) for x in exp.args])

        raise FlopCalculationError("Found unsupported operator in the compute expr")

    def traverse(ops):
        """accumulate flops"""
        ret = 0
        for op in ops:
            if isinstance(op, tensor.ComputeOp):
                num_element = _prod_length(op.axis)

                body = op.body
                if len(body) != 1:
                    raise FlopCalculationError("Found multiple output in the compute")
                exp = body[0]

                ret += num_element * _count_flop(exp)
                ret += traverse([t.op for t in op.input_tensors])

            elif isinstance(op, tensor.PlaceholderOp):
                pass
            else:
                raise FlopCalculationError("Only support tvm.compute currently. "
                                           "Other ops like tvm.scan/tvm.extern is not supported")
        return ret

    try:
        ret = traverse(sch.outputs)
    except FlopCalculationError as exc:
        raise RuntimeError("FLOP estimator fails for this operator. Error msg: "
                           + str(exc) + ". Please use `cfg.add_flop` to manually set "
                                        "FLOP for this operator")

    if ret == 0:
        raise RuntimeError("Cannot find float number operation in this operator. "
                           "Please use `cfg.add_flop` to manually set "
                           "FLOP for this operator")
    return ret
