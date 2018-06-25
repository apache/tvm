"""Definition of task function.

Task can be constructed from tuple of func, args, and kwargs.
func is a state-less function, or a string that
registers the standard task.
"""

from functools import wraps

import numpy as np

from ... import build, nd, tensor, expr, target as _target

from ..util import get_const_tuple, get_const_int
from .dispatcher import DispatchContext, ApplyConfig, dispatcher
from .space import ConfigSpace, ConfigEntity


class Task(object):
    """A Tunable Task

    Parameters
    ----------
    name: str
        The name of the task.
    args: list
        Positional argument of func
    """
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.kwargs = {}  # currently unused

        # init null config space
        self.config_space = None
        self.func = TASK_TABLE[name]

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
            self.flop = config.flop or compute_flop(sch)
        return sch, arg_bufs

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

    ret.workload = ctx.last_workload
    ret.flop = ret.config_space.flop or compute_flop(sch)
    ret.target = target
    ret.target_host = target_host

    return ret

def template(func):
    """
    decorate a function as a template

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
    """
    # pylint: disable=unused-variable

    fname = func.func_name if hasattr(func, 'func_name') else func.__name__

    @wraps(func)
    @register(fname)
    @dispatcher
    def config_dispatcher(*args, **kwargs):
        assert not kwargs, "do not support kwargs in simple_template"
        return (fname, ) + args

    @wraps(func)
    @config_dispatcher.register("")
    def template_call(cfg, *args, **kwargs):
        assert not kwargs, "do not support kwargs in simple_template"
        with ApplyConfig(cfg):
            return func(*args, **kwargs)

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
    pass

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
        elif isinstance(exp, (expr.FloatImm, expr.IntImm, expr.UIntImm)):
            return 0
        elif isinstance(exp, expr.Cast):
            return _count_flop(exp.value)
        elif isinstance(exp, expr.Var):
            return 0
        elif isinstance(exp, (expr.Add, expr.Sub, expr.Mul, expr.Div, expr.Mod,
                              expr.Max, expr.Min,
                              expr.EQ, expr.NE, expr.LT, expr.LE, expr.GT, expr.GE,
                              expr.And, expr.Or, expr.Not)):
            base = 1 if "float" in exp.a.dtype else 0

            if isinstance(exp, expr.Not):  # unary
                return base + _count_flop(exp.a)

            return base + _count_flop(exp.a) + _count_flop(exp.b)
        elif isinstance(exp, expr.Select):
            return _count_flop(exp.condition) + max(_count_flop(exp.true_value),
                                                    _count_flop(exp.false_value))
        elif isinstance(exp, expr.Call):
            return sum([_count_flop(x) for x in exp.args])
        else:
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
                ret += traverse([sch[t].op for t in op.input_tensors])
            elif isinstance(op, tensor.PlaceholderOp):
                pass
            else:
                raise FlopCalculationError("Only support tvm.compute currently. "
                                           "Other ops like tvm.scan is not supported")
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
