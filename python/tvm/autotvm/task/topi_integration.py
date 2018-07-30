# pylint: disable=unused-variable,invalid-name
"""
Decorators for registering tunable templates to topi
"""

from ... import _api_internal, tensor

from ..util import get_func_name
from .task import args_to_workload, dispatcher, register


# A table that records all registered dispatcher for all targets
_REGISTED_DISPATHCER = {
}


def register_topi_compute(topi_compute, target_keys, template_keys):
    """Register a tunable template for a topi compute function

    Parameters
    ----------
    topi_compute: GenericFunc
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

    def _decorator(func=None):
        """If call this function without argument, then we will reuse the function body
        of original function"""
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
                if func is None:
                    node = topi_compute.fdefault(*args, **kwargs)
                else:
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
    topi_schedule: GenericFunc
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
