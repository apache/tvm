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
# pylint: disable=unused-variable,not-callable
"""Definition of task function.

Task can be constructed from tuple of func, args, and kwargs.
func is a state-less function, or a string that
registers the standard task.
"""
import functools

import numpy as np

from tvm import runtime
from tvm.ir import container
from tvm.target import Target
from tvm.te import placeholder, tensor
from tvm.tir import expr


from ..utils import get_const_int, get_const_tuple
from .dispatcher import ApplyConfig, DispatchContext
from .space import ConfigSpace


def _lookup_task(name):
    task = TASK_TABLE.get(name)
    if task is None:
        # Unable to find the given task. This might be because we are
        # creating a task based on a name that has not been imported.
        # Rather than raising an exception here, we return a dummy
        # task which cannot be invoked.
        task = MissingTask(name)
    return task


def serialize_args(args):
    """serialize arguments of a topi function to a hashable tuple.

    Parameters
    ----------
    args: list of hashable or Tensor
    """

    def _encode(x):
        if isinstance(x, tensor.Tensor):
            return ("TENSOR", get_const_tuple(x.shape), x.dtype)
        if isinstance(x, (tuple, list, container.Array)):
            return tuple([_encode(a) for a in x])
        if isinstance(x, (str, int, float, expr.Var, expr.Any)):
            return x
        if isinstance(x, (expr.StringImm, expr.IntImm, expr.FloatImm)):
            return x.value
        if isinstance(x, runtime.container.String):
            return str(x)
        if x is None:
            return None
        raise RuntimeError(
            f'Do not support type "{type(x)}" in argument. Consider to use'
            f"primitive types or tvm.tir.Var only"
        )

    ret = []
    for t in args:
        ret.append(_encode(t))
    return tuple(ret)


def deserialize_args(args):
    """The inverse function of :code:`serialize_args`.

    Parameters
    ----------
    args: list of hashable or Tensor
    """
    ret = []
    for t in args:
        if isinstance(t, tuple) and t[0] == "TENSOR":
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret


def args_to_workload(args, task_name=None):
    """Convert argument list to hashable workload tuple.
    This function will convert list to tuple, tvm node to python value and
    flatten te.tensor.Tensor to a tuple

    Parameters
    ----------
    task_name : str
        The AutoTVM task name

    args : list of args
        The arguments to the function

    Returns
    -------
    ret: hashable
        The hashable value
    """
    return (task_name,) + serialize_args(args) if task_name is not None else serialize_args(args)


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
        self.func = _lookup_task(name)

        # auxiliary info, available after `init_space` is called
        self.flop = None
        self.target = None
        self.target_host = None

    @property
    def workload(self):
        return (self.name,) + serialize_args(self.args)

    def instantiate(self, config):
        """Instantiate this task function (template) with a config.
        Returns corresponding schedule.

        Parameters
        ----------
        config: template.ConfigEntity
            parameter config for this template

        Returns
        -------
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        arg_bufs: Array of te.tensor.Tensor
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
        import cloudpickle  # pylint: disable=import-outside-toplevel

        self.target, self.target_host = Target.canon_target_and_host(self.target, self.target_host)
        return {
            "name": self.name,
            "args": self.args,
            "kwargs": self.kwargs,
            "config_space": self.config_space,
            "flop": self.flop,
            "target": self.target,
            "target_host": self.target_host,
            "func": cloudpickle.dumps(self.func),
        }

    def __setstate__(self, state):
        import cloudpickle  # pylint: disable=import-outside-toplevel

        self.name = state["name"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]
        self.config_space = state["config_space"]
        self.func = cloudpickle.loads(state["func"])
        self.flop = state["flop"]
        self.target, self.target_host = Target.canon_target_and_host(
            state["target"], state["target_host"]
        )

    def __repr__(self):
        return "Task(func_name=%s, args=%s, kwargs=%s, workload=%s)" % (
            self.name,
            self.args,
            self.kwargs,
            self.workload,
        )


TASK_TABLE = {}


class TaskTemplate(object):
    """
    Task template is used to creates a tunable AutoTVM task.

    It can be defined by a pair of compute and schedule function using
    `_register_task_compute` and `_register_task_schedule`,
    or by a customized task creation function that is more flexible using
    `_register_customized_task`.

    Note that when customized func is registered, compute and schedule function
    will be ignored
    """

    def __init__(self):
        self.fcompute = None
        self.fschedule = None
        self.fcustomized = None

    def __call__(self, *args, **kwargs):
        args = deserialize_args(args)
        if self.fcustomized is None:
            return self._default_func(*args, **kwargs)
        assert callable(self.fcustomized)
        return self.fcustomized(*args, **kwargs)

    def _default_func(self, *args, **kwargs):
        assert callable(self.fcompute) and callable(self.fschedule)
        out = self.fcompute(*args, **kwargs)
        arg_bufs = [out] + self._get_inputs(out)
        s = self.fschedule([out])
        return s, arg_bufs

    @staticmethod
    def _get_inputs(out):
        inputs = []
        queue = [out]
        hash_set = set()
        while queue:
            t = queue.pop(0)
            if isinstance(t.op, tensor.PlaceholderOp):
                inputs.append(t)
            else:
                input_tensors = [t for t in t.op.input_tensors if t not in hash_set]
                queue.extend(input_tensors)
                hash_set.update(input_tensors)
        return inputs


class MissingTask(TaskTemplate):
    """
    Dummy task template for a task lookup which cannot be resolved.
    This can occur if the task being requested from _lookup_task()
    has not been imported in this run.
    """

    def __init__(self, taskname: str):
        super().__init__()
        self._taskname = taskname

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Attempting to invoke a missing task {self._taskname}."
            "It is possible that the function is registered in a "
            "Python module that is not imported in this run, or the log is out-of-date."
        )


def _register_task_compute(name, func=None):
    """Register compute function to autotvm task

    Parameters
    ----------
    name: str
        The task name

    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator
    """

    def _do_reg(f):
        if name not in TASK_TABLE:
            TASK_TABLE[name] = TaskTemplate()
        tmpl = TASK_TABLE[name]
        if tmpl.fcompute is not None:
            raise ValueError(f"Compute is already registered in autoTVM task {name}")
        tmpl.fcompute = f
        return f

    if func:
        return _do_reg(func)
    return _do_reg


def _register_task_schedule(name, func=None):
    """Register schedule function to autotvm task

    Parameters
    ----------
    name: str
        The task name

    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator
    """

    def _do_reg(f):
        if name not in TASK_TABLE:
            TASK_TABLE[name] = TaskTemplate()
        tmpl = TASK_TABLE[name]
        if tmpl.fschedule is not None:
            raise ValueError(f"Schedule is already registered in autoTVM task {name}")
        tmpl.fschedule = f
        return f

    if func:
        return _do_reg(func)
    return _do_reg


def _register_customized_task(name, func=None):
    """Register a customized function to AutoTVM task.

    Parameters
    ----------
    name: str
        The task name

    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator
    """

    def _do_reg(f):
        if name not in TASK_TABLE:
            TASK_TABLE[name] = TaskTemplate()
        tmpl = TASK_TABLE[name]
        if tmpl.fcustomized is not None:
            raise ValueError(f"Customized func is already registered in autoTVM task {name}")
        tmpl.fcustomized = f
        return f

    if func:
        return _do_reg(func)
    return _do_reg


def template(task_name, func=None):
    """Decorate a function as a tunable schedule template.

    Parameters
    ----------
    task_name: str
        The task name

    func: None or callable
        A callable template function.
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    func: callable
        The decorated function

    Examples
    --------
    The following code is a tunable template for a blocked matrix multiplication

    .. code-block:: python

        @autotvm.template("matmul")
        def matmul(N, L, M, dtype):
            A = te.placeholder((N, L), name='A', dtype=dtype)
            B = te.placeholder((L, M), name='B', dtype=dtype)

            k = te.reduce_axis((0, L), name='k')
            C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
            s = te.create_schedule(C.op)

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

    def _decorate(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            workload = args_to_workload(args, task_name)
            tgt = Target.current()
            cfg = DispatchContext.current.query(tgt, workload)
            with ApplyConfig(cfg):
                return f(*args, **kwargs)

        _register_customized_task(task_name, f)
        return wrapper

    if func:
        return _decorate(func)
    return _decorate


def create(task_name, args, target, target_host=None):
    """Create a tuning task and initialize its search space

    Parameters
    ----------
    task_name : str
        The AutoTVM task name
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
    args = serialize_args(args)
    ret = Task(task_name, args)

    target, target_host = Target.canon_target_and_host(target, target_host)

    # init config space
    ret.config_space = ConfigSpace()

    ctx = ApplyConfig(ret.config_space)
    with ctx:
        with target:
            sch, _ = ret.func(*args)
            ret.config_space.code_hash = getattr(sch, "code_hash", None)

    ret.flop = ret.config_space.flop or compute_flop(sch)
    ret.target = target
    ret.target_host = target_host

    return ret


def get_config():
    """Get current config object

    Returns
    -------
    cfg: ConfigSpace or ConfigEntity
        The current config
    """
    tgt = Target.current(allow_none=True)
    return DispatchContext.current.query(tgt, None)


class FlopCalculationError(RuntimeError):
    """Error happens when estimating FLOP for a compute op"""


def compute_flop(sch):
    """Calculate number of FLOP (floating number operations) of the compute ops in a schedule

    Parameters
    ----------
    sch: tvm.te.schedule.Schedule
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
        if isinstance(exp, (expr.FloatImm, expr.IntImm)):
            return 0
        if isinstance(exp, expr.Cast):
            return _count_flop(exp.value)
        if isinstance(exp, expr.Var):
            return 0
        if isinstance(
            exp,
            (
                expr.Add,
                expr.Sub,
                expr.Mul,
                expr.Div,
                expr.Mod,
                expr.FloorDiv,
                expr.FloorMod,
                expr.Max,
                expr.Min,
                expr.EQ,
                expr.NE,
                expr.LT,
                expr.LE,
                expr.GT,
                expr.GE,
                expr.And,
                expr.Or,
                expr.Not,
            ),
        ):
            base = 1

            if isinstance(exp, expr.Not):  # unary
                return base + _count_flop(exp.a)

            return base + _count_flop(exp.a) + _count_flop(exp.b)
        if isinstance(exp, expr.Select):
            return _count_flop(exp.condition) + max(
                _count_flop(exp.true_value), _count_flop(exp.false_value)
            )
        if isinstance(exp, expr.ProducerLoad):
            # Ignore flops from indexing expressions.
            return 0

        if isinstance(exp, expr.Call):
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
                raise FlopCalculationError(
                    f"{op.name} is not supported by autotvm. "
                    "Only support te.compute currently. "
                    "Other ops like tvm.te.scan/te.extern is not supported"
                )
        return ret

    try:
        ret = traverse(sch.outputs)
    except FlopCalculationError as exc:
        raise RuntimeError(
            "FLOP estimator fails for this operator. Error msg: "
            + str(exc)
            + ". Please use `cfg.add_flop` to manually set "
            "FLOP for this operator"
        )

    if ret == 0:
        raise RuntimeError(
            "Cannot find float number operation in this operator. "
            "Please use `cfg.add_flop` to manually set "
            "FLOP for this operator"
        )
    return ret
