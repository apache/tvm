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
# pylint: disable=invalid-name
"""Primitive operators in the TVM IR."""
import tvm._ffi
from .expr import RelayExpr
from . import _ffi_api


@tvm._ffi.register_object("Op")
class Op(RelayExpr):
    """Primitive operator in the IR."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    @staticmethod
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
        return _ffi_api.GetOp(op_name)

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
        return _ffi_api.OpGetAttr(self, attr_name)

    def set_attr(self, attr_name, value, plevel=10):
        """Set attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name

        value : object
            The attribute value

        plevel : int
            The priority level
        """
        _ffi_api.OpSetAttr(self, attr_name, value, plevel)

    def reset_attr(self, attr_name):
        """Reset attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name
        """
        _ffi_api.OpResetAttr(self, attr_name)


def register_op_attr(op_name, attr_key, value=None, level=10):
    """Register an operator property of an operator by name.

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
        _ffi_api.RegisterOpAttr(op_name, attr_key, v, level)
        return v

    return _register(value) if value is not None else _register
