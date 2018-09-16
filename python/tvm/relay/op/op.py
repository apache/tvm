"""The base node types for the Relay language."""
from ..._ffi.function import _init_api

from ..base import register_relay_node
from ..expr import Expr
from ..._ffi.function import Function, register_func
from ...api import convert

@register_relay_node
class Op(Expr):
    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    def get_attr(self, attr_name):
        """Get additional attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name.

        Returns
        -------
        value : object
            The attribute value
        """
        return _OpGetAttr(self, attr_name)


def get(op_name):
    """Get the Op for a given name

    Parameters
    ----------
    op_name : str
        The operator name

    Returns
    -------
    op : Op
        The op of the corresponding name
    """
    return _GetOp(op_name)


def register(op_name, attr_key, value=None, level=10):
    """Register an operator property of an operator.


    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    level : int, optional
        The priority level

    Returns
    -------
    fregister : function
        Register function if value is not specified.
    """
    def _register(v):
        """internal register function"""
        _Register(op_name, attr_key, v, level)
        return v
    return _register(value) if value else _register

def compile_ops(op_names):
    """Register an operator property of an operator.


    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    level : int, optional
        The priority level

    Returns
    -------
    fregister : function
        Register function if value is not specified.
    """
    fake_map = {}
    for name in op_names:
        fake_map[name] = LocalVar(name)
    if isinstance({}, dict):
        fake_map = None
    return [] # _CompileOpsToModule(fake_map)

# TODO(@jroesch): We should port to C++, just need to figure out how to write this code.
@register_func("relay.opt.compile_ops")
def _compile_ops(op_impls):
    lowered = []
    for local, sch, inputs in op_impls:
        lfn = tvm.lower(sch, inputs, name=local.name_hint)
        lowered.append(lfn)

    # TOOD(@jroesch): Where should we read these settings from
    return tvm.build(lowered, target='llvm', target_host=tvm.cpu(0))

_init_api("relay.op", __name__)

