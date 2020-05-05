# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Structured error classes in TVM.

Each error class takes an error message as its input.
See the example sections for for suggested message conventions.
To make the code more readable, we recommended developers to
copy the examples and raise errors with the same message convention.

.. note::

    Please also refer to :ref:`error-handling-guide`.
"""
from tvm._ffi.base import register_error, TVMError

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
register_error("AttributeError", AttributeError)
register_error("KeyError", KeyError)


@register_error
class RPCError(RuntimeError):
    """Error thrown by the remote server handling the RPC call."""


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
class OpAttributeUnImplemented(OpError, NotImplementedError):
    """Attribute is not supported in a certain frontend.

    Example
    -------
    .. code:: python

        raise OpAttributeUnImplemented(
            "Attribute {} is not supported in operator {}".format(
                attr_name, op_name))
    """
