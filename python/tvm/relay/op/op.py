"""The base node types for the Relay language."""
from ..._ffi.function import _init_api

from ..base import register_relay_node
from ..expr import Expr
from ..._ffi.function import register_func
from ... import lower, build


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


def compile_ops(op_names):
    """Register an operator property of an operator.


    Parameters
    ----------
    op_names : List[str]
        A list of operator names to compile to machine code.

    Returns
    -------
        A module containing the compiled TVM operators.
    """
    return _CompileOpsToModule(*op_names)

# TODO(@jroesch): We should port to C++, just need to figure out how to write this code.


@register_func("relay.op._compile_ops")
def _compile_ops(op_impls):
    lowered = []
    for local, sch, inputs in op_impls:
        lfn = lower(sch, inputs, name=local.name_hint)
        lowered.append(lfn)

    # TOOD(@jroesch): Where should we read these settings from
    return build(lowered, target='llvm', target_host='llvm')


_init_api("relay.op", __name__)


def specialize_op(op_name, new_op_name, type_args):
    """Specializes an operator to a set of types and assigns it new_op_name.

    The idea is to take operators with generic types such as broadcasting
    addition:

    add : forall (T : Type) (U : Type), (U, T) -> Broadcast(U, T)

    This is a function which is polymorphic over two types `T` and `U` and
    takes a value of type `T` and one of `U` and returns `Broadcast` of U
    and T.

    Broadcast is a type relation which relates U and T to an output type.

    The idea is that the above type is shorthand for:

    add : forall (T : Type) (U : Type) (O : Type), Broadcast(U, T, O) => (U, T) -> O

    That is a function from U and T to O where the typing relation between the values
    is specified by Broadcast.

    We implement a basic Broadcasting rule in `type_relations.h` but users can specify
    their own.

    If we know T=Tensor[(10, 10), dtype], U=Tensor[(10, 10), dtype] then the result
    should be Tensor[(10, 10), dtype].

    We can use SpecializeOp to implement this change of operator.

    Parameters
    ----------
    op_name : str
        The operator to be specialized.

    Returns
    -------
        The specialized operator.
    """
    return _SpecializeOp(op_name, new_op_name, type_args)
