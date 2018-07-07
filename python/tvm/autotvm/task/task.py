# pylint: disable=unused-variable
"""Definition of task function.

Task can be constructed from tuple of func, args, and kwargs.
func is a state-less function, or a string that
registers the standard task.
"""

import numpy as np

from ... import tensor, expr, _api_internal, container, target as _target

from ..util import get_const_int, get_const_tuple, get_func_name
from .dispatcher import DispatchContext, ApplyConfig, dispatcher
from .space import ConfigSpace


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

def args_to_workload(x):
    """Convert argument list to hashable workload tuple.
    This function will convert list to tuple, tvm node to python value and
    flatten tvm.tensor.Tensor to a tuple

    Parameters
    ----------
    x: primitive hashable types or tensor.Tensor
        The original value

    Returns
    -------
    ret: hashable
        The hashable value
    """
    if isinstance(x, tensor.Tensor):
        return get_const_tuple(x.shape) + (x.dtype, )
    elif isinstance(x, (tuple, list, container.Array)):
        return tuple([args_to_workload(a) for a in x])
    elif isinstance(x, (str, int, float, np.int, np.float)):
        return x
    elif isinstance(x, (expr.StringImm, expr.IntImm, expr.FloatImm)):
        return x.value
    else:
        raise RuntimeError('Do not support type "%s" in argument. Consider to use'
                           'primitive types only' % type(x))

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

# a table that records all register dispatch for all targets
_REGISTED_DISPATHCER = {
}

def register_topi_compute(topi_compute, target_keys, template_keys):
    """Register a tunable template for a topi compute function

    Parameters
    ----------
    topi_compute: callable
        The overloaded topi compute call
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key

    Returns
    -------
    decorator: callable
        A decorator
    """
    fname = get_func_name(topi_compute)

    def _decorator(func):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTED_DISPATHCER:
                _REGISTED_DISPATHCER[target_key] = {}
            if topi_compute not in _REGISTED_DISPATHCER:
                @topi_compute.register(target_key)
                @dispatcher
                def config_dispatcher(*args, **kwargs):
                    """override topi call as a config dispatcher"""
                    assert not kwargs, "Do not support kwargs in template function call"
                    return (fname, ) + args_to_workload(args)
                _REGISTED_DISPATHCER[target_key][topi_compute] = config_dispatcher

            config_dispatcher = _REGISTED_DISPATHCER[target_key][topi_compute]

            @config_dispatcher.register(template_keys)
            def template_call(cfg, *args, **kwargs):
                """call the topi func and attach workload to compute node"""
                assert not kwargs, "Do not support kwargs in template function call"
                node = func(cfg, *args, **kwargs)

                # attach workload to return op
                op = node.op
                attrs = {}
                for k, v in node.op.attrs.items():
                    attrs[k] = v
                attrs['workload'] = (fname, ) + args_to_workload(args)
                if isinstance(op, tensor.ComputeOp):
                    op = _api_internal._ComputeOp(
                        op.name, op.tag, attrs, op.axis, op.body)
                elif isinstance(op, tensor.ExternOp):
                    op = _api_internal._ExternOp(
                        op.name, op.tag, attrs,
                        op.inputs, op.input_placeholders,
                        op.output_placeholders, op.body)
                else:
                    raise RuntimeError("Unsupported op type: " + type(op))

                if isinstance(node, tensor.Tensor):
                    return op.output(0)
                return [op.output(i) for i in range(len(node))]

    return _decorator

def register_topi_schedule(topi_schedule, target_keys, template_keys):
    """Register a tunable template for a topi schedule function

    Parameters
    ----------
    topi_schedule: callable
        The overloaded topi schedule call
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key

    Returns
    -------
    decorator: callable
        A decorator
    """
    def _decorator(func):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTED_DISPATHCER:
                _REGISTED_DISPATHCER[target_key] = {}
            if topi_schedule not in _REGISTED_DISPATHCER[target_key]:
                @topi_schedule.register(target_key)
                @dispatcher
                def config_dispatcher(outs):
                    """override topi call as a workload dispatcher"""
                    def traverse(tensors):
                        """traverse all ops to find attached workload"""
                        for t in tensors:
                            op = t.op
                            if 'workload' in op.attrs:
                                return op.attrs['workload']
                            wkl = traverse(op.input_tensors)
                            if wkl:
                                return wkl
                        return None

                    outs = [outs] if isinstance(outs, tensor.Tensor) else outs
                    workload = traverse(outs)

                    if workload is None:
                        raise RuntimeError("Cannot find workload in attribute of this schedule")

                    return args_to_workload(workload)

                _REGISTED_DISPATHCER[target_key][topi_schedule] = config_dispatcher

            config_dispatcher = _REGISTED_DISPATHCER[target_key][topi_schedule]

            @config_dispatcher.register(template_keys)
            def template_call(cfg, outs):
                """call the schedule func"""
                return func(cfg, outs)

    return _decorator


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
