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

import tvm_ffi

from . import _ffi_api
from .expr import Expr


@tvm_ffi.register_object("ir.Op")
class Op(Expr):
    """Primitive operator in the IR."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    @staticmethod
    def get(op_name):
        """Get a registered operator by name.

        Parameters
        ----------
        op_name : str
            The canonical operator name.

        Returns
        -------
        Op
            A handle to the registered operator.
        """
        return _ffi_api.GetOp(op_name)

    @staticmethod
    def list_op_names():
        """List registered operator names in unspecified order.

        Returns
        -------
        list[str]
            The registered operator names.
        """
        return _ffi_api.ListOpNames()

    def set_attr(self, attr_name, value, override=False):
        """Set an operator attribute.

        Parameters
        ----------
        attr_name : str
            Attribute column name.
        value : object
            Non-None attribute value.
        override : bool, optional
            Replace an existing value if True. Duplicate registration otherwise
            raises ValueError. Cached views observe replacements; no history is kept.

        Returns
        -------
        None
        """
        self._set_attr(attr_name, value, override)


def register_op_attr(op_name, attr_key, value=None, override=False):
    """Register an operator property of an operator by name.

    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    override : bool, optional
        Replace an existing value if True; otherwise duplicate registration raises
        ValueError. Cached views observe replacements; no priority history is kept.

    Returns
    -------
    result : object or function
        The registered value when supplied, or a decorator that registers and
        returns its argument. The named Op is created if it does not exist.
    """

    def _register(v):
        """internal register function"""
        _ffi_api.RegisterOpAttr(op_name, attr_key, v, override)
        return v

    return _register(value) if value is not None else _register
