# pylint: disable=unused-variable,invalid-name
"""
Decorators for registering tunable templates to TOPI.

These decorators can make your simple implementation be able to use different configurations
for different workloads.
Here we directly use all arguments to the TOPI call as "workload", so make sure all the arguments
(except tvm.Tensor) in you calls are hashable. For tvm.Tensor, we will serialize it to a hashable
tuple.

See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
"""

from ... import _api_internal, tensor

from ..util import get_func_name
from .task import args_to_workload, dispatcher


# A table that records all registered dispatcher for all targets
_REGISTED_DISPATHCER = {
}


def register_topi_compute(topi_compute, target_keys, template_keys, func=None):
    """Register a tunable template for a topi compute function.

    After the registration. This topi compute will become a configuration dispatcher. It uses
    all its argument as workload and dispatches configurations according to the input workload.

    It also stores this "workload" to its final ComputeOp, which can be used to reconstruct
    "workload" in the following topi_schedule call.

    Parameters
    ----------
    topi_compute: GenericFunc
        The topi compute function that will be overloaded
    target_keys: str or list of str
        The compilation target. The same as the argument of GenericFunc.register.
    template_keys: str or list of str
        The template key.
        We might have several strategies for a single operator (e.g. direct, im2col, winograd).
        The template key is used to identity the algorithm strategy.
        Every operator must have a "direct" template, which is used by default.
    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator

    Examples
    --------
    See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
    """
    fname = get_func_name(topi_compute)

    def _decorator(f):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTED_DISPATHCER:
                _REGISTED_DISPATHCER[target_key] = {}
            if topi_compute not in _REGISTED_DISPATHCER[target_key]:
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

                if f == topi_compute.fdefault:
                    node = f(*args, **kwargs)
                else:
                    node = f(cfg, *args, **kwargs)

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
                    raise RuntimeError("Unsupported op type: " + str(type(op)))

                if isinstance(node, tensor.Tensor):
                    return op.output(0)
                return [op.output(i) for i in range(len(node))]

        return f

    if func:
        _decorator(func)

    return _decorator


def register_topi_schedule(topi_schedule, target_keys, template_keys, func=None):
    """Register a tunable template for a topi schedule function.

    After the registration. This topi schedule will become a configuration dispatcher. It dispatches
    configurations according to the input workload.

    Note that this function will try to find "workload" from all the ComputeOp in the input.
    You can attach "workload" to your compute op by using :any:`register_topi_compute`.

    Parameters
    ----------
    topi_schedule: GenericFunc
        The topi schedule function that will be overloaded
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key.
        We might have several strategies for a single operator (e.g. direct, im2col, winograd).
        The template key is used to identity the algorithm strategy.
        Every operator must have a "direct" template, which is used by default.
    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator

    Examples
    --------
    See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
    """
    def _decorator(f):
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
                if f == topi_schedule.fdefault:
                    return f(outs)
                return f(cfg, outs)

        return f

    if func:
        _decorator(func)

    return _decorator
