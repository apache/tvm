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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, arguments-differ
"""The expression functor of Relax."""
from typing import Callable, Optional

import tvm
from tvm.ir import Op
from tvm.meta_schedule.utils import derived_object
from tvm.runtime import Object

from ..ir.module import IRModule
from . import _ffi_api
from .block_builder import BlockBuilder
from .expr import (
    Binding,
    BindingBlock,
    Call,
    Constant,
    Id,
    DataflowBlock,
    DataflowVar,
    DataTypeImm,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    If,
    MatchCast,
    PrimValue,
    SeqExpr,
    ShapeExpr,
    Span,
    StringImm,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
)
from .struct_info import StructInfo

visitor = derived_object
"""
A decorator to wrap user-customized PyExprVisitor as TVM object _PyExprVisitor.

Parameters
----------
visitor_cls : PyExprVisitor
    The user-customized PyExprVisitor.

Returns
-------
cls : _PyExprVisitor
    The decorated TVM object _PyExprVisitor(ExprVisitor on the C++ side).

Example
-------
.. code-block:: python

    @relax.expr_functor.visitor
    class MyExprVisitor(PyExprVisitor):
        # customize visit function
        def visit_call_(self, op: Call) -> None:
            # just for demo purposes
            ...
    # myvisitor is now a special visitor that visit every Call with
    # user-customized visit_call_
    myvisitor = MyExprVisitor()
    # apply myvisitor to Expr/Binding/BindingBlock/VarDef
    myvisitor.visit_expr(expr)
    myvisitor.visit_binding(binding)
    myvisitor.visit_binding_block(bindingblock)
    myvisitor.visit_var_def(var)
"""

mutator = derived_object
"""
A decorator to wrap user-customized PyExprMutator as TVM object _PyExprMutator.
Note:  Cannot override visit function and post-order rewrite at the same time.

Parameters
----------
mutator_cls : PyExprMutator
    The user-customized PyExprMutator.

Returns
-------
cls : _PyExprMutator
    The decorated TVM object _PyExprMutator(ExprMutator on the C++ side).

Example
-------
.. code-block:: python

    @relax.expr_functor.mutator
    class MyExprMutator(PyExprMutator):
        # customize rewrite function
        def visit_tuple_(self, op: Tuple) -> Expr:
            # just for demo purposes
            ...

    # mymutator is now a special mutator that rewrite every Tuple with
    # user-customized visit_tuple_
    mymutator = MyExprMutator()
    # apply mymutator to Expr/Binding/BindingBlock/VarDef
    mymutator.visit_expr(expr)
    mymutator.visit_binding(binding)
    mymutator.visit_binding_block(bindingblock)
    mymutator.visit_var_def(var)
"""


class ExprFunctor:
    """
    An abstract visitor defined over Expr.
    Defines the default dispatch over expressions, and
    implements memoization.
    """

    def visit_expr(self, expr: Expr) -> Expr:
        """Apply the visitor to an expression."""
        if isinstance(expr, Constant):  # type: ignore
            ret = self.visit_constant_(expr)
        elif isinstance(expr, Tuple):
            ret = self.visit_tuple_(expr)
        elif isinstance(expr, DataflowVar):
            ret = self.visit_dataflow_var_(expr)
        elif isinstance(expr, Var):
            ret = self.visit_var_(expr)
        elif isinstance(expr, ShapeExpr):
            ret = self.visit_shape_expr_(expr)
        elif isinstance(expr, ExternFunc):
            ret = self.visit_extern_func_(expr)
        elif isinstance(expr, GlobalVar):  # type: ignore
            ret = self.visit_global_var_(expr)
        elif isinstance(expr, Function):
            ret = self.visit_function_(expr)
        elif isinstance(expr, Call):  # type: ignore
            ret = self.visit_call_(expr)
        elif isinstance(expr, SeqExpr):
            ret = self.visit_seq_expr_(expr)
        elif isinstance(expr, If):  # type: ignore
            ret = self.visit_if_(expr)
        elif isinstance(expr, Op):
            ret = self.visit_op_(expr)
        elif isinstance(expr, TupleGetItem):
            ret = self.visit_tuple_getitem_(expr)
        elif isinstance(expr, PrimValue):
            ret = self.visit_prim_value_(expr)
        elif isinstance(expr, StringImm):
            ret = self.visit_string_imm_(expr)
        elif isinstance(expr, DataTypeImm):
            ret = self.visit_data_type_imm_(expr)
        else:
            raise TypeError("Invalid type: {0}".format(type(expr)))

        return ret

    def visit_constant_(self, op: Constant):
        raise NotImplementedError()

    def visit_tuple_(self, op: Tuple):
        raise NotImplementedError()

    def visit_dataflow_var_(self, op: DataflowVar):
        raise NotImplementedError()

    def visit_var_(self, op: Var):
        raise NotImplementedError()

    def visit_shape_expr_(self, op: ShapeExpr):
        raise NotImplementedError()

    def visit_extern_func_(self, op: ExternFunc):
        raise NotImplementedError()

    def visit_global_var_(self, op: GlobalVar):
        raise NotImplementedError()

    def visit_function_(self, op: Function):
        raise NotImplementedError()

    def visit_call_(self, op: Call):
        raise NotImplementedError()

    def visit_seq_expr_(self, op: SeqExpr):
        raise NotImplementedError()

    def visit_if_(self, op: If):
        raise NotImplementedError()

    def visit_op_(self, op: Op):
        raise NotImplementedError()

    def visit_tuple_getitem_(self, op: TupleGetItem):
        raise NotImplementedError()

    def visit_prim_value_(self, op: PrimValue):
        raise NotImplementedError()

    def visit_string_imm_(self, op: StringImm):
        raise NotImplementedError()

    def visit_data_type_imm_(self, op: DataTypeImm):
        raise NotImplementedError()

    def visit_var_binding_(self, binding: VarBinding):
        raise NotImplementedError()

    def visit_match_cast_(self, binding: MatchCast):
        raise NotImplementedError()

    def visit_binding_block_(self, block: BindingBlock):
        raise NotImplementedError()

    def visit_dataflow_block_(self, block: DataflowBlock):
        raise NotImplementedError()

    def visit_var_def_(self, var: Var):
        raise NotImplementedError()

    def visit_dataflow_var_def_(self, var: DataflowVar):
        raise NotImplementedError()

    def visit_binding(self, binding: Binding):
        if isinstance(binding, MatchCast):
            self.visit_match_cast_(binding)
        elif isinstance(binding, VarBinding):
            self.visit_var_binding_(binding)
        else:
            raise TypeError("Invalid type: {0}".format(type(binding)))

    def visit_binding_block(self, block: BindingBlock):
        if isinstance(block, DataflowBlock):
            self.visit_dataflow_block_(block)
        elif isinstance(block, BindingBlock):
            self.visit_binding_block_(block)
        else:
            raise TypeError("Invalid type: {0}".format(type(block)))

    def visit_var_def(self, var: Var):
        if isinstance(var, DataflowVar):
            self.visit_dataflow_var_def_(var)
        elif isinstance(var, Var):
            self.visit_var_def_(var)
        else:
            raise TypeError("Invalid type: {0}".format(type(var)))


@tvm._ffi.register_object("expr_functor.PyExprVisitor")
class _PyExprVisitor(Object):
    """
    A TVM object to support customization of ExprVisitor on the python side.
    This is the decorated result returned from visitor decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: visitor, PyExprVisitor
    """

    def __init__(
        self,
        f_visit_expr: Callable = None,
        f_visit_constant_: Callable = None,
        f_visit_tuple_: Callable = None,
        f_visit_var_: Callable = None,
        f_visit_dataflow_var_: Callable = None,
        f_visit_shape_expr_: Callable = None,
        f_visit_extern_func_: Callable = None,
        f_visit_global_var_: Callable = None,
        f_visit_function_: Callable = None,
        f_visit_call_: Callable = None,
        f_visit_seq_expr_: Callable = None,
        f_visit_if_: Callable = None,
        f_visit_op_: Callable = None,
        f_visit_tuple_getitem_: Callable = None,
        f_visit_prim_value_: Callable = None,
        f_visit_string_imm_: Callable = None,
        f_visit_data_type_imm_: Callable = None,
        f_visit_binding: Callable = None,
        f_visit_var_binding_: Callable = None,
        f_visit_match_cast_: Callable = None,
        f_visit_binding_block: Callable = None,
        f_visit_binding_block_: Callable = None,
        f_visit_dataflow_block_: Callable = None,
        f_visit_var_def: Callable = None,
        f_visit_var_def_: Callable = None,
        f_visit_dataflow_var_def_: Callable = None,
        f_visit_span: Callable = None,
    ) -> None:
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprVisitor,  # type: ignore
            f_visit_expr,
            f_visit_constant_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_prim_value_,
            f_visit_string_imm_,
            f_visit_data_type_imm_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_cast_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_span,
        )

    def visit_expr(self, expr: Expr) -> None:
        """Generic dispatcher for Expr.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.
        """
        return _ffi_api.PyExprVisitorVisitExpr(self, expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        return _ffi_api.PyExprVisitorVisitBinding(self, binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> None:
        """Generic dispatcher for BindingBlock.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.
        """
        return _ffi_api.PyExprVisitorVisitBindingBlock(self, block)  # type: ignore

    def visit_var_def(self, var: Var) -> None:
        """Generic dispatcher for visiting the var definition site.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.
        """
        return _ffi_api.PyExprVisitorVisitVarDef(self, var)  # type: ignore


class PyExprVisitor:
    """
    An abstract ExprVisitor with customized methods on the python-side.
    This is the user facing class for method overwriting inheritance.
    _tvm_metadata discribes the class to inherit("cls"), the methods
    that users can overwrite("methods").

    Note: @relax.expr_functor.visitor is required for proper usage of any inherited class.

    See also: visitor, _PyExprVisitor

    Example:

    .. code-block:: python

        @relax.expr_functor.visitor
        def MyExprVisitor(PyExprVisitor):
            ...

    """

    _tvm_metadata = {
        "cls": _PyExprVisitor,
        "methods": [
            "visit_expr",
            "visit_constant_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_prim_value_",
            "visit_string_imm_",
            "visit_data_type_imm_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_cast_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_span",
        ],
    }

    def visit_expr(self, expr: Expr) -> None:
        """Generic dispatcher for Expr.
        Users can customized this function to overwrite VisitExpr(const Expr& expr) on the C++ side.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.
        Users can customized this function to overwrite VisitBinding(const Binding& binding)
        on the C++ side.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> None:
        """Generic dispatcher for BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock(const BindingBlock& block)
        on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def(self, var: Var) -> None:
        """Generic dispatcher for visiting the var definition site.
        Users can customized this function to overwrite VisitVarDef(const Var& var) on the C++ side.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_constant_(self, op: Constant) -> None:
        """Visit Constant.
        Users can customized this function to overwrite VisitExpr_(const ConstantNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Constant
            The Constant to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_(self, op: Tuple) -> None:
        """Visit Tuple.
        Users can customized this function to overwrite VisitExpr_(const TupleNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Tuple
            The Tuple to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> None:
        """Visit Var.
        Users can customized this function to overwrite VisitExpr_(const VarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_dataflow_var_(self, op: DataflowVar) -> None:
        """Visit DataflowVar.
        Users can customized this function to overwrite VisitExpr_(const DataflowVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DataflowVar
            The DataflowVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_shape_expr_(self, op: ShapeExpr) -> None:
        """Visit ShapeExpr.
        Users can customized this function to overwrite VisitExpr_(const ShapeExprNode* op)
        on the C++ side.

        Parameters
        ----------
        op : ShapeExpr
            The ShapeExpr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_extern_func_(self, op: ExternFunc) -> None:
        """Visit ExternFunc.
        Users can customized this function to overwrite VisitExpr_(const ExternFuncNode* op)
        on the C++ side.

        Parameters
        ----------
        op : ExternFunc
            The ExternFunc to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_global_var_(self, op: GlobalVar) -> None:
        """Visit GlobalVar.
        Users can customized this function to overwrite VisitExpr_(const GlobalVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : GlobalVar
            The GlobalVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_function_(self, op: Function) -> None:
        """Visit Function.
        Users can customized this function to overwrite VisitExpr_(const FunctionNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Function
            The Function to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> None:
        """Visit Call.
        Users can customized this function to overwrite VisitExpr_(const CallNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        """Visit SeqExpr.
        Users can customized this function to overwrite VisitExpr_(const SeqExprNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqExpr
            The SeqExpr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_if_(self, op: If) -> None:
        """Visit If.
        Users can customized this function to overwrite VisitExpr_(const IfNode* op)
        on the C++ side.

        Parameters
        ----------
        op : If
            The If to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_op_(self, op: Op) -> None:
        """Visit Op.
        Users can customized this function to overwrite VisitExpr_(const OpNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Op
            The Op to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
        """Visit TupleGetItem.
        Users can customized this function to overwrite VisitExpr_(const TupleGetItemNode* op)
        on the C++ side.

        Parameters
        ----------
        op : TupleGetItem
            The TupleGetItem to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_prim_value_(self, op: PrimValue) -> None:
        """Visit PrimValue.
        Users can customized this function to overwrite VisitExpr_(const PrimValueNode* op)
        on the C++ side.

        Parameters
        ----------
        op : PrimValue
            The PrimValue to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> None:
        """Visit StringImm.
        Users can customized this function to overwrite VisitExpr_(const StringImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_data_type_imm_(self, op: DataTypeImm) -> None:
        """Visit DataTypeImm.
        Users can customized this function to overwrite VisitExpr_(const DataTypeImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DataTypeImm
            The DataTypeImm to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Visit VarBinding.
        Users can customized this function to overwrite VisitBinding_(const VarBindingNode* binding)
        on the C++ side.

        Parameters
        ----------
        binding : VarBinding
            The VarBinding to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_match_cast_(self, binding: MatchCast) -> None:
        """Visit MatchCast.
        Users can customized this function to overwrite VisitBinding_(const MatchCastNode* binding)
        on the C++ side.

        Parameters
        ----------
        binding : MatchCast
            The MatchCast to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block_(self, block: BindingBlock) -> None:
        """Visit BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const BindingBlockNode*
        block) on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The BindingBlock to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        """Visit DataflowBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const DataflowBlockNode*
        block) on the C++ side.

        Parameters
        ----------
        block : DataflowBlock
            The DataflowBlock to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def_(self, var: Var) -> None:
        """Visit the Var definition site.
        Users can customized this function to overwrite VisitVarDef_(const VarNode* var)
        on the C++ side.

        Parameters
        ----------
        var : Var
            The Var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        """Visit the DataflowVar definition site.
        Users can customized this function to overwrite VisitVarDef_(const DataflowVarNode* var)
        on the C++ side.

        Parameters
        ----------
        var : DataflowVar
            The DataflowVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_span(self, span: Span) -> None:
        """Visit Span.
        Users can customized this function to overwrite VisitSpan(const Span& span) on the C++ side.

        Parameters
        ----------
        span : Span
            The Span to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitSpan(self._outer(), span)  # type: ignore


@tvm._ffi.register_object("expr_functor.PyExprMutator")
class _PyExprMutator(Object):
    """
    A TVM object to support customization of ExprMutator on the python side.
    This is the decorated result returned from mutator decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: mutator, PyExprmutator
    """

    def __init__(
        self,
        builder: BlockBuilder = None,
        f_visit_expr: Callable = None,
        f_visit_constant_: Callable = None,
        f_visit_tuple_: Callable = None,
        f_visit_var_: Callable = None,
        f_visit_dataflow_var_: Callable = None,
        f_visit_shape_expr_: Callable = None,
        f_visit_extern_func_: Callable = None,
        f_visit_global_var_: Callable = None,
        f_visit_function_: Callable = None,
        f_visit_call_: Callable = None,
        f_visit_seq_expr_: Callable = None,
        f_visit_if_: Callable = None,
        f_visit_op_: Callable = None,
        f_visit_tuple_getitem_: Callable = None,
        f_visit_prim_value_: Callable = None,
        f_visit_string_imm_: Callable = None,
        f_visit_data_type_imm_: Callable = None,
        f_visit_binding: Callable = None,
        f_visit_var_binding_: Callable = None,
        f_visit_match_cast_: Callable = None,
        f_visit_binding_block: Callable = None,
        f_visit_binding_block_: Callable = None,
        f_visit_dataflow_block_: Callable = None,
        f_visit_var_def: Callable = None,
        f_visit_var_def_: Callable = None,
        f_visit_dataflow_var_def_: Callable = None,
        f_visit_span: Callable = None,
    ) -> None:
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprMutator,  # type: ignore
            builder,
            f_visit_expr,
            f_visit_constant_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_prim_value_,
            f_visit_string_imm_,
            f_visit_data_type_imm_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_cast_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_span,
        )

    def visit_expr(self, expr: Expr) -> Expr:
        """Generic dispatcher for Expr.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation.
        """
        return _ffi_api.PyExprMutatorVisitExpr(self, expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        return _ffi_api.PyExprMutatorVisitBinding(self, binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        """Generic dispatcher for BindingBlock.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation.
        """
        return _ffi_api.PyExprMutatorVisitBindingBlock(self, block)  # type: ignore

    def visit_var_def(self, var: Var) -> Var:
        """Generic dispatcher for visiting the var definition site.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        return _ffi_api.PyExprMutatorVisitVarDef(self, var)  # type: ignore


class PyExprMutator:
    """
    An abstract ExprMutator with customized methods on the python-side.
    This is the user facing class for method overwriting inheritance.
    _tvm_metadata discribes the class to inherit("cls"), the methods that users can
    overwrite("methods"), the constructor's parameters("fields")

    Note: @relax.expr_functor.mutator is required for proper usage of any inherited class.

    See also: visitor, _PyExprVisitor

    Example:

    .. code-block:: python

        @relax.expr_functor.mutator
        def MyExprMutator(PyExprMutator):
            ...

    """

    _tvm_metadata = {
        "cls": _PyExprMutator,
        "fields": ["builder_"],
        "methods": [
            "visit_expr",
            "visit_constant_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_prim_value_",
            "visit_string_imm_",
            "visit_data_type_imm_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_cast_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_span",
        ],
    }

    def __init__(self, mod: Optional[IRModule] = None) -> None:
        """Constructor"""
        self.builder_ = BlockBuilder(mod)

    def visit_expr(self, expr: Expr) -> Expr:
        """Generic dispatcher for Expr.
        Users can customized this function to overwrite VisitExpr(const Expr& expr) on the C++ side.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.
        Users can customized this function to overwrite VisitBinding(const Binding& binding)
        on the C++ side.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        """Generic dispatcher for BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock(const BindingBlock& block)
        on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def(self, var: Var) -> Var:
        """Generic dispatcher for visiting the var definition site.
        Users can customized this function to overwrite VisitVarDef(const Var& var) on the C++ side.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.

        Returns
        -------
        result: Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_constant_(self, op: Constant) -> Expr:
        """Visit Constant.
        Users can customized this function to overwrite VisitExpr_(const ConstantNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Constant
            The Constant to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_(self, op: Tuple) -> Expr:
        """Visit Tuple.
        Users can customized this function to overwrite VisitExpr_(const TupleNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Tuple
            The Tuple to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> Expr:
        """Visit Var.
        Users can customized this function to overwrite VisitExpr_(const VarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
        """Visit DataflowVar.
        Users can customized this function to overwrite VisitExpr_(const DataflowVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DataflowVar
            The DataflowVar to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
        """Visit ShapeExpr.
        Users can customized this function to overwrite VisitExpr_(const ShapeExprNode* op)
        on the C++ side.

        Parameters
        ----------
        op : ShapeExpr
            The ShapeExpr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_extern_func_(self, op: ExternFunc) -> Expr:
        """Visit ExternFunc.
        Users can customized this function to overwrite VisitExpr_(const ExternFuncNode* op)
        on the C++ side.

        Parameters
        ----------
        op : ExternFunc
            The ExternFunc to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_global_var_(self, op: GlobalVar) -> Expr:
        """Visit GlobalVar.
        Users can customized this function to overwrite VisitExpr_(const GlobalVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : GlobalVar
            The GlobalVar to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_function_(self, op: Function) -> Expr:
        """Visit Function.
        Users can customized this function to overwrite VisitExpr_(const FunctionNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Function
            The Function to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> Expr:
        """Visit Call.
        Users can customized this function to overwrite VisitExpr_(const CallNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_seq_expr_(self, op: SeqExpr) -> Expr:
        """Visit SeqExpr.
        Users can customized this function to overwrite VisitExpr_(const SeqExprNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqExpr
            The SeqExpr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_if_(self, op: If) -> Expr:
        """Visit If.
        Users can customized this function to overwrite VisitExpr_(const IfNode* op)
        on the C++ side.

        Parameters
        ----------
        op : If
            The If to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_op_(self, op: Op) -> Expr:
        """Visit Op.
        Users can customized this function to overwrite VisitExpr_(const OpNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Op
            The Op to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
        """Visit TupleGetItem.
        Users can customized this function to overwrite VisitExpr_(const TupleGetItemNode* op)
        on the C++ side.

        Parameters
        ----------
        op : TupleGetItem
            The TupleGetItem to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_prim_value_(self, op: PrimValue) -> Expr:
        """Visit PrimValue.
        Users can customized this function to overwrite VisitExpr_(const PrimValueNode* op)
        on the C++ side.

        Parameters
        ----------
        op : PrimValue
            The PrimValue to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> Expr:
        """Visit StringImm.
        Users can customized this function to overwrite VisitExpr_(const StringImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_data_type_imm_(self, op: DataTypeImm) -> Expr:
        """Visit DataTypeImm.
        Users can customized this function to overwrite VisitExpr_(const DataTypeImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DataTypeImm
            The DataTypeImm to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Visit VarBinding.
        Users can customized this function to overwrite VisitBinding_(const VarBindingNode* binding)
        on the C++ side.

        Parameters
        ----------
        binding : VarBinding
            The VarBinding to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_match_cast_(self, binding: MatchCast) -> None:
        """Visit MatchCast.
        Users can customized this function to overwrite VisitBinding_(const MatchCastNode* binding)
        on the C++ side.

        Parameters
        ----------
        binding : MatchCast
            The MatchCast to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
        """Visit BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const BindingBlockNode*
        block) on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The BindingBlock to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_dataflow_block_(self, block: DataflowBlock) -> BindingBlock:
        """Visit DataflowBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const DataflowBlockNode*
        block) on the C++ side.

        Parameters
        ----------
        block : DataflowBlock
            The DataflowBlock to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def_(self, var: Var) -> Var:
        """Visit the Var definition site.
        Users can customized this function to overwrite VisitVarDef_(const VarNode* var)
        on the C++ side.

        Parameters
        ----------
        var : Var
            The Var to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_dataflow_var_def_(self, var: DataflowVar) -> Var:
        """Visit the DataflowVar definition site.
        Users can customized this function to overwrite VisitVarDef_(const DataflowVarNode* var)
        on the C++ side.

        Parameters
        ----------
        var : DataflowVar
            The DataflowVar to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_span(self, span: Span) -> Span:
        """Visit Span.
        Users can customized this function to overwrite VisitSpan(const Span& span) on the C++ side.

        Parameters
        ----------
        span : Span
            The Span to be visited.

        Returns
        -------
        result : Span
            The span after transformation.
        """
        raise NotImplementedError

    def visit_expr_post_order(self, expr: Expr) -> Expr:
        """Post-order rewrite an Expr and normalize.

        Parameters
        ----------
        expr : Expr
            The Expr to be rewritten.

        Returns
        -------
        result : Expr
            The Expr after post-order rewritten.
        """
        return _ffi_api.PyExprMutatorVisitExprPostOrder(self._outer(), expr)  # type: ignore

    def set_var_remap(self, vid: Id, var: Var) -> None:
        """Remap a var to a new var in use-site.

        Parameters
        ----------
        vid : Id
            The vid of the old var.
        var : Var
            The new var.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorSetVarRemap(self._outer(), vid, var)  # type: ignore

    def get_var_remap(self, vid: Id) -> Var:
        """Remap a var to a new var in use-site.

        Parameters
        ----------
        vid : Id
            The vid of the old var

        Returns
        -------
        var : Var
            The remapped var.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorGetVarRemap(self._outer(), vid)  # type: ignore

    def visit_with_new_scope(self, expr: Expr) -> Expr:
        """Rewrite the expr with a new scope, used in a Function's body and the branches of If.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        var : Var
            The expr after visiting.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitWithNewScope(self._outer(), expr)  # type: ignore

    def lookup_binding(self, var: Var) -> Optional[Expr]:
        """Look up the value bound to a variable.
        Note: For function parameters, this function returns NullOpt.

        Parameters
        ----------
        var : Var
            The var to be looked up.

        Returns
        -------
        var : Var
            The value bound to the input var.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorLookupBinding(self._outer(), var)  # type: ignore

    def with_struct_info(self, var: Var, struct_info: StructInfo) -> Var:
        """Create a new var with specified shape and type if the original var's shape or type does
        not match with the specified ones.

        Parameters
        ----------
        var : Var
            The var to be updated.
        struct_info : StructInfo
            The struct info.

        Returns
        -------
        var : Var
            The var filled with shape and type.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorWithStructInfo(self._outer(), var, struct_info)  # type: ignore
