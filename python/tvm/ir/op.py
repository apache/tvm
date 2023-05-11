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

from . import _ffi_api
from .expr import RelayExpr


@tvm._ffi.register_object("Op")
class Op(RelayExpr):
    """Primitive operator in the IR."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[Object->str]
            Optionally annotate function to provide additional
            information in the comment block.

        Returns
        -------
        text : str
            The text format of the expression.

        Notes
        -----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.
        """
        from tvm.relay import astext  # pylint: disable=import-outside-toplevel

        return astext(self, show_meta_data, annotate)

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

    def has_attr(self, attr_name):
        """Check whether the operator has additional attribute.

        Parameters
        ----------
        attr_name : str
            The attribute name.

        Returns
        -------
        value : bool
            Whether the operator has additional attribute
        """
        return _ffi_api.OpHasAttr(self, attr_name)

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

    def add_type_rel(self, rel_name, type_rel_func=None):
        """Attach the type function corresponding to the return type.

        Parameters
        ----------
        rel_name : str
            The type relation name to register.

        type_rel_func : Optional[function (args: List[Type], attrs: Attrs) -> Type]
            The backing relation function which can solve an arbitrary relation on variables.
            Differences with type_rel_func in C++:

            1) When type_rel_func is not None

               a) OpAddTypeRel on C++ side will adjust type_rel_func with TypeReporter to
                  calling convention of relay type system.

               b) type_rel_func returns output argument's type, return None means can't
                  infer output's type.

               c) only support single output operators for now, the last argument is output tensor.

            2) when type_rel_func is None, will call predefined type_rel_funcs in relay
                   according to ``tvm.relay.type_relation.`` + rel_name.

        """
        _ffi_api.OpAddTypeRel(self, rel_name, type_rel_func)

    def add_argument(self, name, type, description):  # pylint: disable=redefined-builtin
        """Add arguments information to the function.

        Parameters
        ----------
        name : str
            The argument name.
        type : str
            The argument type.
        description : str
            The argument description.
        """
        _ffi_api.OpAddArgument(self, name, type, description)

    def set_support_level(self, level):
        """Set the support level of op.

        Parameters
        ----------
        level : int
            The support level.
        """
        _ffi_api.OpSetSupportLevel(self, level)

    def set_num_inputs(self, n):
        """Set the support level of op.

        Parameters
        ----------
        n : int
            The input number.
        """
        _ffi_api.OpSetNumInputs(self, n)

    def set_attrs_type_key(self, key):
        """Set the attribute type key of op.

        Parameters
        ----------
        key : str
            The type key.
        """
        _ffi_api.OpSetAttrsTypeKey(self, key)

    @staticmethod
    def list_op_names():
        """List all the op names in the op registry.

        Returns
        -------
        value : List[str]
            The registered op names
        """
        return _ffi_api.ListOpNames()


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


def register_intrin_lowering(
    op_name,
    target,
    *,
    f=None,
    level=10,
):
    """Register Op lowering function

    Parameters
    ----------
    op_name : str
        The op name

    target : str
        The target string for given intrinsic lowering function

    f : function, optional
        The function to be registered.

    level : int
        The priority level

    Returns
    -------
    fregister : function
        Register op lowering function if f is not specified.
    """

    def _register(f):
        """internal register function"""
        _ffi_api.RegisterOpLowerIntrinsic(op_name, f, target, level)
        return f

    return _register(f) if f is not None else _register
