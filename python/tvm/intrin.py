"""Expression Intrinsics and math functions in TVM."""
# pylint: disable=redefined-builtin
from __future__ import absolute_import as _abs

from ._ffi.function import register_func as _register_func
from . import make as _make
from .api import convert, const
from .expr import Call as _Call
from .schedule import Buffer as _Buffer

def _pack_buffer(buf):
    """Build intrinsics that packs the buffer.
    """
    assert buf.shape
    shape = _make.Call("handle", "tvm_stack_make_shape", buf.shape,
                       _Call.Intrinsic, None, 0)
    strides = _make.Call("handle", "tvm_stack_make_shape", buf.strides,
                         _Call.Intrinsic, None, 0) if buf.strides else 0
    pack_args = [buf.data,
                 shape,
                 strides,
                 len(buf.shape),
                 const(0, dtype=buf.dtype),
                 buf.elem_offset]
    return _make.Call("handle", "tvm_stack_make_array",
                      pack_args, _Call.Intrinsic, None, 0)

def call_packed(*args):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will recieve an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    Returns
    -------
    call : Expr
        The call expression.

    See Also
    --------
    tvm.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, _Buffer) else x for x in args]
    return _make.Call(
        "int32", "tvm_call_packed", call_args, _Call.Intrinsic, None, 0)


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

    Returns
    -------
    call : Expr
        The call expression.
    """
    args = convert(args)
    return _make.Call(
        dtype, func_name, convert(args), _Call.PureIntrinsic, None, 0)


def call_intrin(dtype, func_name, *args):
    """Build expression by calling an intrinsic function.

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

    Returns
    -------
    call : Expr
        The call expression.
    """
    args = convert(args)
    return _make.Call(
        dtype, func_name, convert(args), _Call.Intrinsic, None, 0)


def call_pure_extern(dtype, func_name, *args):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return _make.Call(
        dtype, func_name, convert(args), _Call.PureExtern, None, 0)


def call_extern(dtype, func_name, *args):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return _make.Call(
        dtype, func_name, convert(args), _Call.Extern, None, 0)


def call_llvm_intrin(dtype, name, *args):
    """Build expression by calling an llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Poistional arguments.

    Returns
    -------
    call : Expr
        The call expression.
    """
    import tvm
    llvm_id = tvm.codegen.llvm_lookup_intrinsic_id(name)
    assert llvm_id != 0, "%s is not an LLVM intrinsic" % name
    return call_pure_intrin(dtype, 'llvm_intrin', tvm.const(llvm_id, 'uint32'), *args)


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


def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "tanh", x)


def sigmoid(x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "sigmoid", x)


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


def sqrt(x):
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
    return call_pure_intrin(x.dtype, "sqrt", x)


def floor(x):
    """Take floor of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "floor", x)


def ceil(x):
    """Take ceil of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "ceil", x)


def trunc(x):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "trunc", x)


def abs(x):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return _make.abs(x)


def round(x):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "round", x)


def power(x, y):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    Returns
    -------
    z : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "pow", x, y)


def popcount(x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(x.dtype, "popcount", x)


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
    return None

# opencl pattern for exp
register_intrin_rule("opencl", "exp", _rule_float_direct, override=True)
# default pattern for exp
register_intrin_rule("default", "exp", _rule_float_suffix, override=True)

# default pattern for sigmoid
register_intrin_rule("default", "sigmoid", lambda op: 1.0 / (1.0 + exp(-op.args[0])))
