"""Intrinsics and math functions in TVM."""
from __future__ import absolute_import as _abs

from .expr import Call as _Call
from . import make as _make
from ._ctypes._function import register_func as _register_func
from .api import convert

def call_packed(*args):
    """Build expression by call an external packed function

    Parameters
    ----------
    args : list
        Positional arguments.
    """
    return _make.Call(
        "int32", "tvm_call_packed", args, _Call.Intrinsic, None, 0)


def call_pure_intrin(dtype, func_name, *args):
    """Build expression by calling a pure intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.
    """
    args = convert(args)
    return _make.Call(
        dtype, func_name, convert(args), _Call.PureIntrinsic, None, 0)


def call_pure_extern(dtype, func_name, *args):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.
    """
    return _make.Call(
        dtype, func_name, convert(args), _Call.PureExtern, None, 0)


def exp(x):
    """Take exponetial of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "exp", x)


def log(x):
    """Take log of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "log", x)


# Intrinsic rule related code
def register_intrin_rule(target, intrin, f=None, override=False):
    """Register an intrinsic function generation rule.

    Intrinsic generation rules are callback functions for
    code generator to get device specific calls.
    This function simply translates to.

    :code:`register_func("tvm.intrin.rule.%s.%s" % (target, intrin), f, override)`

    TVM may already pre-register intrinsic rules in the backend.
    However, user can use this function to change the intrinsic translation
    behavior or add new intrinsic rules during runtime.

    Parameters
    ----------
    target : str
        The name of codegen target.

    intrin : str
        The name of the instrinsic.

    f : function, optional
        The function to be registered.

    override: boolean optional
        Whether override existing entry.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    The following code registers exp expansion rule for opencl.

    .. code-block:: python

        register_intrin_rule("opencl", "exp", my_exp_rule, override=True)
    """
    return _register_func("tvm.intrin.rule.%s.%s" % (target, intrin), f, override)


def _rule_float_suffix(op):
    """Intrinsic rule: Add float suffix if it is float32.

    This is an example intrinsic generation rule.

    Parameters
    ----------
    op : Expr
        The call expression of original intrinsic.

    Returns
    -------
    ret : Expr
        The translated intrinsic rule.
        Return same op if no translation is possible.

    See Also
    --------
    register_intrin_rule : The registeration function for intrin rule.
    """
    if op.dtype == "float32":
        return call_pure_extern(op.dtype, "%sf" % op.name, *op.args)
    elif op.dtype == "float64":
        return call_pure_extern(op.dtype, op.name, *op.args)
    else:
        return op


def _rule_float_direct(op):
    """Intrinsic rule: Directly call pure extern function for floats.

    This is an example intrinsic generation rule.

    Parameters
    ----------
    op : Expr
        The call expression of original intrinsic.

    Returns
    -------
    ret : Expr
        The translated intrinsic rule.
        Return same op if no translation is possible.

    See Also
    --------
    register_intrin_rule : The registeration function for intrin rule.
    """
    if str(op.dtype).startswith("float"):
        return call_pure_extern(op.dtype, op.name, *op.args)
    else:
        return None

# opencl pattern for exp
register_intrin_rule("opencl", "exp", _rule_float_direct, override=True)
# default pattern for exp
register_intrin_rule("default", "exp", _rule_float_suffix, override=True)
