"""The base node types for the Relay language."""
from ..._ffi.function import _init_api

from ..base import register_relay_node
from ..expr import Expr


@register_relay_node
class Op(Expr):
    """A Relay operator definition."""

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


_init_api("relay.op", __name__)
