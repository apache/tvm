"""Structured error classes in TVM.

Each error class takes an error message as its input.
See the example sections for for suggested message conventions.
To make the code more readable, we recommended developers to
copy the examples and raise errors with the same message convention.
"""
from ._ffi.base import register_error, TVMError

@register_error
class InternalError(TVMError):
    """Internal error in the system.

    Examples
    --------
    .. code :: c++

        // Example code C++
        LOG(FATAL) << "InternalError: internal error detail.";

    .. code :: python

        # Example code in python
        raise InternalError("internal error detail")
    """
    def __init__(self, msg):
        # Patch up additional hint message.
        if "TVM hint:" not in msg:
            msg += ("\nTVM hint: You hit an internal error. " +
                    "Please open a thread on https://discuss.tvm.ai/ to report it.")
        super(InternalError, self).__init__(msg)


register_error("ValueError", ValueError)
register_error("TypeError", TypeError)


@register_error
class OpError(TVMError):
    """Base class of all operator errors in frontends."""


@register_error
class OpNotImplemented(OpError, NotImplementedError):
    """Operator is not implemented.

    Example
    -------
    .. code:: python

        raise OpNotImplemented(
            "Operator {} is not supported in {} frontend".format(
                missing_op, frontend_name))
    """


@register_error
class OpAttributeRequired(OpError, AttributeError):
    """Required attribute is not found.

    Example
    -------
    .. code:: python

        raise OpAttributeRequired(
            "Required attribute {} not found in operator {}".format(
                attr_name, op_name))
    """


@register_error
class OpAttributeInvalid(OpError, AttributeError):
    """Attribute value is invalid when taking in a frontend operator.

    Example
    -------
    .. code:: python

        raise OpAttributeInvalid(
            "Value {} in attribute {} of operator {} is not valid".format(
                value, attr_name, op_name))
    """


@register_error
class OpAttributeUnimplemented(OpError, NotImplementedError):
    """Attribute is not supported in a certain frontend.

    Example
    -------
    .. code:: python

        raise OpAttributeUnimplemented(
            "Attribute {} is not supported in operator {}".format(
                attr_name, op_name))
    """
