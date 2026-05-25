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
# ruff: noqa: RUF012
"""The expression and statement functor of TIR."""

from collections.abc import Callable

import tvm_ffi

from tvm.ir import PrimExpr
from tvm.runtime.support import derived_object

from . import _ffi_api
from .expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Broadcast,
    BufferLoad,
    Call,
    CallEffectKind,
    Cast,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    IterVar,
    Let,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    ProducerLoad,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
)
from .stmt import (
    AllocBuffer,
    AssertStmt,
    AttrStmt,
    Bind,
    BufferStore,
    DeclBuffer,
    Evaluate,
    For,
    IfThenElse,
    SBlock,
    SBlockRealize,
    SeqStmt,
    Stmt,
    While,
)

visitor = derived_object
"""
A decorator to wrap user-customized PyStmtExprVisitor as TVM object _PyStmtExprVisitor.

Parameters
----------
visitor_cls : PyStmtExprVisitor
    The user-customized PyStmtExprVisitor.

Returns
-------
cls : _PyStmtExprVisitor
    The decorated TVM object _PyStmtExprVisitor(StmtExprVisitor on the C++ side).

Example
-------
.. code-block:: python

    @tirx.functor.stmt_expr_visitor
    class MyStmtExprVisitor(PyStmtExprVisitor):
        # customize visit function
        def visit_call_(self, op: Call) -> None:
            # just for demo purposes
            ...
    # myvisitor is now a special visitor that visit every Call with
    # user-customized visit_call_
    myvisitor = MyStmtExprVisitor()
    # apply myvisitor to PrimExpr and Stmt
    myvisitor.visit_expr(expr)
    myvisitor.visit_stmt(stmt)
"""

mutator = derived_object
"""
A decorator to wrap user-customized PyStmtExprMutator as TVM object _PyStmtExprMutator.

Parameters
----------
mutator_cls : PyStmtExprMutator
    The user-customized PyStmtExprMutator.

Returns
-------
cls : _PyStmtExprMutator
    The decorated TVM object _PyStmtExprMutator(StmtExprMutator on the C++ side).

Example
-------
.. code-block:: python

    @tirx.functor.stmt_expr_mutator
    class MyStmtExprMutator(PyStmtExprMutator):
        # customize rewrite function
        def visit_add_(self, op: Add) -> PrimExpr:
            # just for demo purposes
            ...

    # mymutator is now a special mutator that rewrite every Add with
    # user-customized visit_add_
    mymutator = MyStmtExprMutator()
    # apply mymutator to PrimExpr and Stmt
    mymutator.visit_expr(expr)
    mymutator.visit_stmt(stmt)
"""


@tvm_ffi.register_object("tirx.PyStmtExprVisitor")
class _PyStmtExprVisitor(tvm_ffi.core.Object):
    """
    An internal wrapper to interface between C++ and Python StmtExprVisitor.
    This is the TVM object that wraps PyStmtExprVisitor.

    Do not use this class directly. Use PyStmtExprVisitor instead.

    See also: PyStmtExprVisitor, stmt_expr_visitor
    """

    def __init__(
        self,
        f_visit_stmt: Callable | None = None,
        f_visit_expr: Callable | None = None,
        # Stmt
        f_visit_bind: Callable | None = None,
        f_visit_attr_stmt: Callable | None = None,
        f_visit_if_then_else: Callable | None = None,
        f_visit_for: Callable | None = None,
        f_visit_while: Callable | None = None,
        f_visit_alloc_buffer: Callable | None = None,
        f_visit_decl_buffer: Callable | None = None,
        f_visit_buffer_store: Callable | None = None,
        f_visit_assert_stmt: Callable | None = None,
        f_visit_seq_stmt: Callable | None = None,
        f_visit_evaluate: Callable | None = None,
        f_visit_block: Callable | None = None,
        f_visit_sblock_realize: Callable | None = None,
        # PrimExpr
        f_visit_var: Callable | None = None,
        f_visit_size_var: Callable | None = None,
        f_visit_buffer_load: Callable | None = None,
        f_visit_producer_load: Callable | None = None,
        f_visit_let: Callable | None = None,
        f_visit_call: Callable | None = None,
        f_visit_add: Callable | None = None,
        f_visit_sub: Callable | None = None,
        f_visit_mul: Callable | None = None,
        f_visit_div: Callable | None = None,
        f_visit_mod: Callable | None = None,
        f_visit_floor_div: Callable | None = None,
        f_visit_floor_mod: Callable | None = None,
        f_visit_min: Callable | None = None,
        f_visit_max: Callable | None = None,
        f_visit_eq: Callable | None = None,
        f_visit_ne: Callable | None = None,
        f_visit_lt: Callable | None = None,
        f_visit_le: Callable | None = None,
        f_visit_gt: Callable | None = None,
        f_visit_ge: Callable | None = None,
        f_visit_and: Callable | None = None,
        f_visit_or: Callable | None = None,
        f_visit_reduce: Callable | None = None,
        f_visit_cast: Callable | None = None,
        f_visit_not: Callable | None = None,
        f_visit_select: Callable | None = None,
        f_visit_ramp: Callable | None = None,
        f_visit_broadcast: Callable | None = None,
        f_visit_shuffle: Callable | None = None,
        f_visit_int_imm: Callable | None = None,
        f_visit_float_imm: Callable | None = None,
        f_visit_string_imm: Callable | None = None,
    ) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.MakePyStmtExprVisitor,  # type: ignore
            f_visit_stmt,
            f_visit_expr,
            # Stmt
            f_visit_bind,
            f_visit_attr_stmt,
            f_visit_if_then_else,
            f_visit_for,
            f_visit_while,
            f_visit_alloc_buffer,
            f_visit_decl_buffer,
            f_visit_buffer_store,
            f_visit_assert_stmt,
            f_visit_seq_stmt,
            f_visit_evaluate,
            f_visit_block,
            f_visit_sblock_realize,
            # PrimExpr
            f_visit_var,
            f_visit_size_var,
            f_visit_buffer_load,
            f_visit_producer_load,
            f_visit_let,
            f_visit_call,
            f_visit_add,
            f_visit_sub,
            f_visit_mul,
            f_visit_div,
            f_visit_mod,
            f_visit_floor_div,
            f_visit_floor_mod,
            f_visit_min,
            f_visit_max,
            f_visit_eq,
            f_visit_ne,
            f_visit_lt,
            f_visit_le,
            f_visit_gt,
            f_visit_ge,
            f_visit_and,
            f_visit_or,
            f_visit_reduce,
            f_visit_cast,
            f_visit_not,
            f_visit_select,
            f_visit_ramp,
            f_visit_broadcast,
            f_visit_shuffle,
            f_visit_int_imm,
            f_visit_float_imm,
            f_visit_string_imm,
        )


class PyStmtExprVisitor:
    """
    A Python StmtExprVisitor to define custom visitor for both Stmt and PrimExpr.

    Users can customize any of the visit function.
    """

    _tvm_metadata = {
        "cls": _PyStmtExprVisitor,
        "methods": [
            "visit_stmt",
            "visit_expr",
            # Stmt
            "visit_bind_",
            "visit_attr_stmt_",
            "visit_if_then_else_",
            "visit_for_",
            "visit_while_",
            "visit_alloc_buffer_",
            "visit_decl_buffer_",
            "visit_buffer_store_",
            "visit_assert_stmt_",
            "visit_seq_stmt_",
            "visit_evaluate_",
            "visit_sblock_",
            "visit_sblock_realize_",
            # PrimExpr
            "visit_var_",
            "visit_size_var_",
            "visit_buffer_load_",
            "visit_producer_load_",
            "visit_let_",
            "visit_call_",
            "visit_add_",
            "visit_sub_",
            "visit_mul_",
            "visit_div_",
            "visit_mod_",
            "visit_floor_div_",
            "visit_floor_mod_",
            "visit_min_",
            "visit_max_",
            "visit_eq_",
            "visit_ne_",
            "visit_lt_",
            "visit_le_",
            "visit_gt_",
            "visit_ge_",
            "visit_and_",
            "visit_or_",
            "visit_reduce_",
            "visit_cast_",
            "visit_not_",
            "visit_select_",
            "visit_ramp_",
            "visit_broadcast_",
            "visit_shuffle_",
            "visit_int_imm_",
            "visit_float_imm_",
            "visit_string_imm_",
        ],
    }

    def visit_stmt(self, stmt: Stmt) -> None:
        """Visit a Stmt.

        Parameters
        ----------
        stmt : Stmt
            The Stmt to be visited.
        """
        _ffi_api.PyStmtExprVisitorVisitStmt(self._outer(), stmt)  # type: ignore

    def visit_expr(self, expr: PrimExpr) -> None:
        """Visit a PrimExpr.

        Parameters
        ----------
        expr : PrimExpr
            The PrimExpr to be visited.
        """
        _ffi_api.PyStmtExprVisitorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_attr_stmt_(self, op: AttrStmt) -> None:
        """Visit AttrStmt.
        Users can customize this function to overwrite VisitStmt_(const AttrStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AttrStmt
            The AttrStmt to be visited.
        """
        print("visit_attr_stmt_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_if_then_else_(self, op: IfThenElse) -> None:
        """Visit IfThenElse.
        Users can customize this function to overwrite VisitStmt_(const IfThenElseNode* op)
        on the C++ side.

        Parameters
        ----------
        op : IfThenElse
            The IfThenElse to be visited.
        """
        print("visit_if_then_else_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_bind_(self, op: Bind) -> None:
        """Visit Bind.
        Users can customize this function to overwrite VisitStmt_(const BindNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Bind
            The Bind node to be visited.
        """
        print("visit_bind_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_for_(self, op: For) -> None:
        """Visit For.
        Users can customize this function to overwrite VisitStmt_(const ForNode* op)
        on the C++ side.

        Parameters
        ----------
        op : For
            The For to be visited.
        """
        print("visit_for_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_while_(self, op: While) -> None:
        """Visit While.
        Users can customize this function to overwrite VisitStmt_(const WhileNode* op)
        on the C++ side.

        Parameters
        ----------
        op : While
            The While to be visited.
        """
        print("visit_while_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_alloc_buffer_(self, op: AllocBuffer) -> None:
        """Visit AllocBuffer.
        Users can customize this function to overwrite VisitStmt_(const AllocBufferNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AllocBuffer
            The AllocBuffer to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_decl_buffer_(self, op: DeclBuffer) -> None:
        """Visit DeclBuffer.
        Users can customize this function to overwrite VisitStmt_(const DeclBufferNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DeclBuffer
            The DeclBuffer to be visited.
        """
        print("visit_decl_buffer_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_buffer_store_(self, op: BufferStore) -> None:
        """Visit BufferStore.
        Users can customize this function to overwrite VisitStmt_(const BufferStoreNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferStore
            The BufferStore to be visited.
        """
        print("visit_buffer_store_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_assert_stmt_(self, op: AssertStmt) -> None:
        """Visit AssertStmt.
        Users can customize this function to overwrite VisitStmt_(const AssertStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AssertStmt
            The AssertStmt to be visited.
        """
        print("visit_assert_stmt_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_seq_stmt_(self, op: SeqStmt) -> None:
        """Visit SeqStmt.
        Users can customize this function to overwrite VisitStmt_(const SeqStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqStmt
            The SeqStmt to be visited.
        """
        print("visit_seq_stmt_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_evaluate_(self, op: Evaluate) -> None:
        """Visit Evaluate.
        Users can customize this function to overwrite VisitStmt_(const EvaluateNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Evaluate
            The Evaluate to be visited.
        """
        print("visit_evaluate_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_sblock_(self, op: SBlock) -> None:
        """Visit SBlock.
        Users can customize this function to overwrite VisitStmt_(const SBlockNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SBlock
            The SBlock to be visited.
        """
        print("visit_sblock_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_sblock_realize_(self, op: SBlockRealize) -> None:
        """Visit BlockRealize.
        Users can customize this function to overwrite VisitStmt_(const SBlockRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SBlockRealize
            The BlockRealize to be visited.
        """
        print("visit_sblock_realize_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> None:
        """Visit Var.

        Users can customize this function to overwrite VisitVar_(const VarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_size_var_(self, op: SizeVar) -> None:
        """Visit SizeVar.

        Users can customize this function to overwrite VisitSizeVar_(const SizeVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SizeVar
            The SizeVar to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        """Visit BufferLoad.

        Users can customize this function to overwrite VisitBufferLoad_(const BufferLoadNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferLoad
            The BufferLoad to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_producer_load_(self, op: ProducerLoad) -> None:
        """Visit ProducerLoad.

        Users can customize this function to overwrite
        VisitProducerLoad_(const ProducerLoadNode* op) on the C++ side.

        Parameters
        ----------
        op : ProducerLoad
            The ProducerLoad to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_let_(self, op: Let) -> None:
        """Visit Let.

        Users can customize this function to overwrite VisitLet_(const LetNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Let
            The Let to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> None:
        """Visit Call.

        Users can customize this function to overwrite VisitCall_(const CallNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_add_(self, op: Add) -> None:
        """Visit Add.

        Users can customize this function to overwrite VisitAdd_(const AddNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Add
            The Add to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_sub_(self, op: Sub) -> None:
        """Visit Sub.

        Users can customize this function to overwrite VisitSub_(const SubNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Sub
            The Sub to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_mul_(self, op: Mul) -> None:
        """Visit Mul.

        Users can customize this function to overwrite VisitMul_(const MulNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Mul
            The Mul to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_div_(self, op: Div) -> None:
        """Visit Div.

        Users can customize this function to overwrite VisitDiv_(const DivNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Div
            The Div to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_mod_(self, op: Mod) -> None:
        """Visit Mod.

        Users can customize this function to overwrite VisitMod_(const ModNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Mod
            The Mod to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_floor_div_(self, op: FloorDiv) -> None:
        """Visit FloorDiv.

        Users can customize this function to overwrite VisitFloorDiv_(const FloorDivNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloorDiv
            The FloorDiv to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_floor_mod_(self, op: FloorMod) -> None:
        """Visit FloorMod.

        Users can customize this function to overwrite VisitFloorMod_(const FloorModNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloorMod
            The FloorMod to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_min_(self, op: Min) -> None:
        """Visit Min.

        Users can customize this function to overwrite VisitMin_(const MinNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Min
            The Min to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_max_(self, op: Max) -> None:
        """Visit Max.

        Users can customize this function to overwrite VisitMax_(const MaxNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Max
            The Max to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_eq_(self, op: EQ) -> None:
        """Visit EQ.

        Users can customize this function to overwrite VisitEQ_(const EQNode* op)
        on the C++ side.

        Parameters
        ----------
        op : EQ
            The EQ to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ne_(self, op: NE) -> None:
        """Visit NE.

        Users can customize this function to overwrite VisitNE_(const NENode* op)
        on the C++ side.

        Parameters
        ----------
        op : NE
            The NE to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_lt_(self, op: LT) -> None:
        """Visit LT.

        Users can customize this function to overwrite VisitLT_(const LTNode* op)
        on the C++ side.

        Parameters
        ----------
        op : LT
            The LT to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_le_(self, op: LE) -> None:
        """Visit LE.

        Users can customize this function to overwrite VisitLE_(const LENode* op)
        on the C++ side.

        Parameters
        ----------
        op : LE
            The LE to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_gt_(self, op: GT) -> None:
        """Visit GT.

        Users can customize this function to overwrite VisitGT_(const GTNode* op)
        on the C++ side.

        Parameters
        ----------
        op : GT
            The GT to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ge_(self, op: GE) -> None:
        """Visit GE.

        Users can customize this function to overwrite VisitGE_(const GENode* op)
        on the C++ side.

        Parameters
        ----------
        op : GE
            The GE to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_and_(self, op: And) -> None:
        """Visit And.

        Users can customize this function to overwrite VisitAnd_(const AndNode* op)
        on the C++ side.

        Parameters
        ----------
        op : And
            The And to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_or_(self, op: Or) -> None:
        """Visit Or.

        Users can customize this function to overwrite VisitOr_(const OrNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Or
            The Or to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_reduce_(self, op: Reduce) -> None:
        """Visit Reduce.

        Users can customize this function to overwrite VisitReduce_(const ReduceNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Reduce
            The Reduce to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_cast_(self, op: Cast) -> None:
        """Visit Cast.

        Users can customize this function to overwrite VisitCast_(const CastNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Cast
            The Cast to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_not_(self, op: Not) -> None:
        """Visit Not.

        Users can customize this function to overwrite VisitNot_(const NotNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Not
            The Not to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_select_(self, op: Select) -> None:
        """Visit Select.

        Users can customize this function to overwrite VisitSelect_(const SelectNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Select
            The Select to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ramp_(self, op: Ramp) -> None:
        """Visit Ramp.

        Users can customize this function to overwrite VisitRamp_(const RampNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Ramp
            The Ramp to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_broadcast_(self, op: Broadcast) -> None:
        """Visit Broadcast.

        Users can customize this function to overwrite VisitBroadcast_(const BroadcastNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Broadcast
            The Broadcast to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_shuffle_(self, op: Shuffle) -> None:
        """Visit Shuffle.

        Users can customize this function to overwrite VisitShuffle_(const ShuffleNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Shuffle
            The Shuffle to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_int_imm_(self, op: IntImm) -> None:
        """Visit IntImm.

        Users can customize this function to overwrite VisitIntImm_(const IntImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : IntImm
            The IntImm to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_float_imm_(self, op: FloatImm) -> None:
        """Visit FloatImm.

        Users can customize this function to overwrite VisitFloatImm_(const FloatImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloatImm
            The FloatImm to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> None:
        """Visit StringImm.

        Users can customize this function to overwrite VisitStringImm_(const StringImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.
        """
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore


class _AnalyzerContextMixin:
    """Shared analyzer context helpers for Python functors."""

    def _init_analyzer(self, analyzer):
        if analyzer is None:
            from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel

            analyzer = Analyzer()
        self.analyzer = analyzer
        self._constraint_scopes = []

    def _real_condition(self, condition):
        if isinstance(condition, Call) and getattr(condition.op, "name", None) == "tirx.likely":
            return condition.args[0]
        return condition

    def _negated_condition(self, condition):
        return self.analyzer.rewrite_simplify(Not(condition))

    def _is_pure(self, expr):
        if isinstance(expr, BufferLoad | ProducerLoad):
            return False
        if isinstance(expr, Call):
            try:
                effect = expr.op.get_attr("TCallEffectKind")
            except AttributeError:
                return False
            if effect is None:
                return False
            effect_value = getattr(effect, "value", effect)
            return effect_value <= CallEffectKind.Pure.value and all(
                self._is_pure(arg) for arg in expr.args
            )
        if isinstance(expr, Let):
            return self._is_pure(expr.value) and self._is_pure(expr.body)
        if isinstance(expr, Reduce):
            return (
                all(self._is_pure(source) for source in expr.source)
                and all(self._is_pure(init) for init in expr.init)
                and self._is_pure(expr.condition)
            )
        if isinstance(expr, Select):
            return (
                self._is_pure(expr.condition)
                and self._is_pure(expr.true_value)
                and self._is_pure(expr.false_value)
            )
        if isinstance(expr, Ramp):
            return (
                self._is_pure(expr.base)
                and self._is_pure(expr.stride)
                and self._is_pure(expr.lanes)
            )
        if isinstance(expr, Broadcast):
            return self._is_pure(expr.value) and self._is_pure(expr.lanes)
        if isinstance(expr, Shuffle):
            return all(self._is_pure(vec) for vec in expr.vectors) and all(
                self._is_pure(index) for index in expr.indices
            )
        if isinstance(expr, Cast):
            return self._is_pure(expr.value)
        if isinstance(expr, Not):
            return self._is_pure(expr.a)
        if isinstance(
            expr,
            Add
            | Sub
            | Mul
            | Div
            | Mod
            | FloorDiv
            | FloorMod
            | Min
            | Max
            | EQ
            | NE
            | LT
            | LE
            | GT
            | GE
            | And
            | Or,
        ):
            return self._is_pure(expr.a) and self._is_pure(expr.b)

        return isinstance(expr, Var | SizeVar | IntImm | FloatImm | StringImm)

    def _push_constraint(self, constraint):
        scope = self.analyzer.constraint_scope(constraint)
        scope.__enter__()
        self._constraint_scopes.append(scope)

    def _pop_constraints(self, depth):
        while len(self._constraint_scopes) > depth:
            self._constraint_scopes.pop().__exit__(None, None, None)


class PyStmtExprVisitorWithAnalyzer(PyStmtExprVisitor, _AnalyzerContextMixin):
    """A Python StmtExprVisitor that maintains an arithmetic analyzer context.

    The analyzer is available as ``self.analyzer`` from user callbacks. The default
    traversal binds loop variables, block iter variables, let variables, and branch
    conditions before visiting nested nodes, so callbacks can query the analyzer
    using the surrounding IR context.
    """

    def __init__(self, analyzer=None):
        super().__init__()
        self._init_analyzer(analyzer)

    def visit_for_(self, op: For) -> None:
        from tvm.ir import Range  # pylint: disable=import-outside-toplevel

        depth = len(self._constraint_scopes)
        self.visit_expr(op.min)
        self.visit_expr(op.extent)
        if op.step is not None:
            self.visit_expr(op.step)
        self.analyzer.bind(op.loop_var, Range.from_min_extent(op.min, op.extent))
        try:
            self.visit_stmt(op.body)
        finally:
            self._pop_constraints(depth)

    def visit_attr_stmt_(self, op: AttrStmt) -> None:
        from tvm.ir import Range  # pylint: disable=import-outside-toplevel

        depth = len(self._constraint_scopes)
        self.visit_expr(op.value)
        if op.attr_key in ("thread_extent", "virtual_thread") and isinstance(op.node, IterVar):
            self.analyzer.bind(
                op.node.var, Range.from_min_extent(IntImm(op.value.dtype, 0), op.value)
            )
        try:
            self.visit_stmt(op.body)
        finally:
            self._pop_constraints(depth)

    def visit_sblock_(self, op: SBlock) -> None:
        depth = len(self._constraint_scopes)
        try:
            for iter_var in op.iter_vars:
                self.analyzer.bind(iter_var.var, iter_var.dom)
            _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore
        finally:
            self._pop_constraints(depth)

    def visit_bind_(self, op: Bind) -> None:
        self.visit_expr(op.value)
        if self._is_pure(op.value):
            self.analyzer.bind(op.var, op.value)

    def visit_seq_stmt_(self, op: SeqStmt) -> None:
        depth = len(self._constraint_scopes)
        try:
            _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore
        finally:
            self._pop_constraints(depth)

    def visit_if_then_else_(self, op: IfThenElse) -> None:
        condition = self._real_condition(op.condition)
        self.visit_expr(op.condition)
        depth = len(self._constraint_scopes)
        with self.analyzer.constraint_scope(condition):
            try:
                self.visit_stmt(op.then_case)
            finally:
                self._pop_constraints(depth)
        if op.else_case is not None:
            with self.analyzer.constraint_scope(self._negated_condition(condition)):
                try:
                    self.visit_stmt(op.else_case)
                finally:
                    self._pop_constraints(depth)

    def visit_assert_stmt_(self, op: AssertStmt) -> None:
        self.visit_expr(op.condition)
        self._push_constraint(op.condition)
        self.visit_expr(op.error_kind)
        for msg in op.message_parts:
            self.visit_expr(msg)

    def visit_let_(self, op: Let) -> None:
        self.visit_expr(op.value)
        if self._is_pure(op.value):
            self.analyzer.bind(op.var, op.value)
        self.visit_expr(op.body)

    def visit_call_(self, op: Call) -> None:
        if getattr(op.op, "name", None) != "tirx.if_then_else":
            _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore
            return

        condition = op.args[0]
        self.visit_expr(condition)
        with self.analyzer.constraint_scope(condition):
            self.visit_expr(op.args[1])
        with self.analyzer.constraint_scope(self._negated_condition(condition)):
            self.visit_expr(op.args[2])

    def visit_select_(self, op: Select) -> None:
        self.visit_expr(op.condition)
        with self.analyzer.constraint_scope(op.condition):
            self.visit_expr(op.true_value)
        with self.analyzer.constraint_scope(self._negated_condition(op.condition)):
            self.visit_expr(op.false_value)

    def visit_reduce_(self, op: Reduce) -> None:
        for iter_var in op.axis:
            self.analyzer.bind(iter_var.var, iter_var.dom)
        _ffi_api.PyStmtExprVisitorDefaultVisitExpr(self._outer(), op)  # type: ignore


@tvm_ffi.register_object("tirx.PyStmtExprMutator")
class _PyStmtExprMutator(tvm_ffi.core.Object):
    """
    A TVM object to support customization of StmtExprMutator on the python side.
    This is the decorated result returned from stmt_expr_mutator decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: stmt_expr_mutator, PyStmtExprMutator
    """

    def __init__(
        self,
        f_visit_stmt: Callable | None = None,
        f_visit_expr: Callable | None = None,
        # Stmt
        f_visit_bind: Callable | None = None,
        f_visit_attr_stmt: Callable | None = None,
        f_visit_if_then_else: Callable | None = None,
        f_visit_for: Callable | None = None,
        f_visit_while: Callable | None = None,
        f_visit_alloc_buffer: Callable | None = None,
        f_visit_decl_buffer: Callable | None = None,
        f_visit_buffer_store: Callable | None = None,
        f_visit_assert_stmt: Callable | None = None,
        f_visit_seq_stmt: Callable | None = None,
        f_visit_evaluate: Callable | None = None,
        f_visit_block: Callable | None = None,
        f_visit_sblock_realize: Callable | None = None,
        # PrimExpr
        f_visit_var: Callable | None = None,
        f_visit_size_var: Callable | None = None,
        f_visit_buffer_load: Callable | None = None,
        f_visit_producer_load: Callable | None = None,
        f_visit_let: Callable | None = None,
        f_visit_call: Callable | None = None,
        f_visit_add: Callable | None = None,
        f_visit_sub: Callable | None = None,
        f_visit_mul: Callable | None = None,
        f_visit_div: Callable | None = None,
        f_visit_mod: Callable | None = None,
        f_visit_floor_div: Callable | None = None,
        f_visit_floor_mod: Callable | None = None,
        f_visit_min: Callable | None = None,
        f_visit_max: Callable | None = None,
        f_visit_eq: Callable | None = None,
        f_visit_ne: Callable | None = None,
        f_visit_lt: Callable | None = None,
        f_visit_le: Callable | None = None,
        f_visit_gt: Callable | None = None,
        f_visit_ge: Callable | None = None,
        f_visit_and: Callable | None = None,
        f_visit_or: Callable | None = None,
        f_visit_reduce: Callable | None = None,
        f_visit_cast: Callable | None = None,
        f_visit_not: Callable | None = None,
        f_visit_select: Callable | None = None,
        f_visit_ramp: Callable | None = None,
        f_visit_broadcast: Callable | None = None,
        f_visit_shuffle: Callable | None = None,
        f_visit_int_imm: Callable | None = None,
        f_visit_float_imm: Callable | None = None,
        f_visit_string_imm: Callable | None = None,
    ) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.MakePyStmtExprMutator,  # type: ignore
            f_visit_stmt,
            f_visit_expr,
            # Stmt
            f_visit_bind,
            f_visit_attr_stmt,
            f_visit_if_then_else,
            f_visit_for,
            f_visit_while,
            f_visit_alloc_buffer,
            f_visit_decl_buffer,
            f_visit_buffer_store,
            f_visit_assert_stmt,
            f_visit_seq_stmt,
            f_visit_evaluate,
            f_visit_block,
            f_visit_sblock_realize,
            # PrimExpr
            f_visit_var,
            f_visit_size_var,
            f_visit_buffer_load,
            f_visit_producer_load,
            f_visit_let,
            f_visit_call,
            f_visit_add,
            f_visit_sub,
            f_visit_mul,
            f_visit_div,
            f_visit_mod,
            f_visit_floor_div,
            f_visit_floor_mod,
            f_visit_min,
            f_visit_max,
            f_visit_eq,
            f_visit_ne,
            f_visit_lt,
            f_visit_le,
            f_visit_gt,
            f_visit_ge,
            f_visit_and,
            f_visit_or,
            f_visit_reduce,
            f_visit_cast,
            f_visit_not,
            f_visit_select,
            f_visit_ramp,
            f_visit_broadcast,
            f_visit_shuffle,
            f_visit_int_imm,
            f_visit_float_imm,
            f_visit_string_imm,
        )


class PyStmtExprMutator:
    """
    A Python StmtExprMutator to define custom mutator for both Stmt and PrimExpr.

    Users can customize any of the visit function.
    """

    _tvm_metadata = {
        "cls": _PyStmtExprMutator,
        "methods": [
            "visit_stmt",
            "visit_expr",
            # Stmt
            "visit_bind_",
            "visit_attr_stmt_",
            "visit_if_then_else_",
            "visit_for_",
            "visit_while_",
            "visit_alloc_buffer_",
            "visit_decl_buffer_",
            "visit_buffer_store_",
            "visit_assert_stmt_",
            "visit_seq_stmt_",
            "visit_evaluate_",
            "visit_sblock_",
            "visit_sblock_realize_",
            # PrimExpr
            "visit_var_",
            "visit_size_var_",
            "visit_buffer_load_",
            "visit_producer_load_",
            "visit_let_",
            "visit_call_",
            "visit_add_",
            "visit_sub_",
            "visit_mul_",
            "visit_div_",
            "visit_mod_",
            "visit_floor_div_",
            "visit_floor_mod_",
            "visit_min_",
            "visit_max_",
            "visit_eq_",
            "visit_ne_",
            "visit_lt_",
            "visit_le_",
            "visit_gt_",
            "visit_ge_",
            "visit_and_",
            "visit_or_",
            "visit_reduce_",
            "visit_cast_",
            "visit_not_",
            "visit_select_",
            "visit_ramp_",
            "visit_broadcast_",
            "visit_shuffle_",
            "visit_int_imm_",
            "visit_float_imm_",
            "visit_string_imm_",
        ],
    }

    def visit_expr(self, expr: PrimExpr) -> PrimExpr:
        """Visit PrimExpr.
        Users can customize this function to overwrite VisitExpr(const PrimExpr& expr)
        on the C++ side.

        Parameters
        ----------
        expr : PrimExpr
            The PrimExpr to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_stmt(self, stmt: Stmt) -> Stmt:
        """Visit Stmt.
        Users can customize this function to overwrite VisitStmt(const Stmt& stmt)
        on the C++ side.

        Parameters
        ----------
        stmt : Stmt
            The Stmt to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorVisitStmt(self._outer(), stmt)  # type: ignore

    def visit_attr_stmt_(self, op: AttrStmt) -> Stmt:
        """Visit AttrStmt.
        Users can customize this function to overwrite VisitStmt_(const AttrStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AttrStmt
            The AttrStmt to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_if_then_else_(self, op: IfThenElse) -> Stmt:
        """Visit IfThenElse.
        Users can customize this function to overwrite VisitStmt_(const IfThenElseNode* op)
        on the C++ side.

        Parameters
        ----------
        op : IfThenElse
            The IfThenElse to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_bind_(self, op: Bind) -> Stmt:
        """Visit Bind.
        Users can customize this function to overwrite VisitStmt_(const BindNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Bind
            The Bind node to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_for_(self, op: For) -> Stmt:
        """Visit For.
        Users can customize this function to overwrite VisitStmt_(const ForNode* op)
        on the C++ side.

        Parameters
        ----------
        op : For
            The For to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_while_(self, op: While) -> Stmt:
        """Visit While.
        Users can customize this function to overwrite VisitStmt_(const WhileNode* op)
        on the C++ side.

        Parameters
        ----------
        op : While
            The While to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_alloc_buffer_(self, op: AllocBuffer) -> Stmt:
        """Visit AllocBuffer.
        Users can customize this function to overwrite VisitStmt_(const AllocBufferNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AllocBuffer
            The AllocBuffer to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_decl_buffer_(self, op: DeclBuffer) -> Stmt:
        """Visit DeclBuffer.
        Users can customize this function to overwrite VisitStmt_(const DeclBufferNode* op)
        on the C++ side.

        Parameters
        ----------
        op : DeclBuffer
            The DeclBuffer to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_buffer_store_(self, op: BufferStore) -> Stmt:
        """Visit BufferStore.
        Users can customize this function to overwrite VisitStmt_(const BufferStoreNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferStore
            The BufferStore to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_assert_stmt_(self, op: AssertStmt) -> Stmt:
        """Visit AssertStmt.
        Users can customize this function to overwrite VisitStmt_(const AssertStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AssertStmt
            The AssertStmt to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_seq_stmt_(self, op: SeqStmt) -> Stmt:
        """Visit SeqStmt.
        Users can customize this function to overwrite VisitStmt_(const SeqStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqStmt
            The SeqStmt to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_evaluate_(self, op: Evaluate) -> Stmt:
        """Visit Evaluate.
        Users can customize this function to overwrite VisitStmt_(const EvaluateNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Evaluate
            The Evaluate to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_sblock_(self, op: SBlock) -> Stmt:
        """Visit SBlock.
        Users can customize this function to overwrite VisitStmt_(const SBlockNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SBlock
            The SBlock to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_sblock_realize_(self, op: SBlockRealize) -> Stmt:
        """Visit BlockRealize.
        Users can customize this function to overwrite VisitStmt_(const SBlockRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SBlockRealize
            The SBlockRealize to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> PrimExpr:
        """Visit Var.

        Users can customize this function to overwrite VisitVar_(const VarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_size_var_(self, op: SizeVar) -> PrimExpr:
        """Visit SizeVar.

        Users can customize this function to overwrite VisitSizeVar_(const SizeVarNode* op)
        on the C++ side.

        Parameters
        ----------
        op : SizeVar
            The SizeVar to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_buffer_load_(self, op: BufferLoad) -> PrimExpr:
        """Visit BufferLoad.

        Users can customize this function to overwrite VisitBufferLoad_(const BufferLoadNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferLoad
            The BufferLoad to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_producer_load_(self, op: ProducerLoad) -> PrimExpr:
        """Visit ProducerLoad.

        Users can customize this function to overwrite
        VisitProducerLoad_(const ProducerLoadNode* op) on the C++ side.

        Parameters
        ----------
        op : ProducerLoad
            The ProducerLoad to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_let_(self, op: Let) -> PrimExpr:
        """Visit Let.

        Users can customize this function to overwrite VisitLet_(const LetNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Let
            The Let to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> PrimExpr:
        """Visit Call.

        Users can customize this function to overwrite VisitCall_(const CallNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_add_(self, op: Add) -> PrimExpr:
        """Visit Add.

        Users can customize this function to overwrite VisitAdd_(const AddNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Add
            The Add to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_sub_(self, op: Sub) -> PrimExpr:
        """Visit Sub.

        Users can customize this function to overwrite VisitSub_(const SubNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Sub
            The Sub to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_mul_(self, op: Mul) -> PrimExpr:
        """Visit Mul.

        Users can customize this function to overwrite VisitMul_(const MulNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Mul
            The Mul to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_div_(self, op: Div) -> PrimExpr:
        """Visit Div.

        Users can customize this function to overwrite VisitDiv_(const DivNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Div
            The Div to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_mod_(self, op: Mod) -> PrimExpr:
        """Visit Mod.

        Users can customize this function to overwrite VisitMod_(const ModNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Mod
            The Mod to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_floor_div_(self, op: FloorDiv) -> PrimExpr:
        """Visit FloorDiv.

        Users can customize this function to overwrite VisitFloorDiv_(const FloorDivNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloorDiv
            The FloorDiv to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_floor_mod_(self, op: FloorMod) -> PrimExpr:
        """Visit FloorMod.

        Users can customize this function to overwrite VisitFloorMod_(const FloorModNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloorMod
            The FloorMod to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_min_(self, op: Min) -> PrimExpr:
        """Visit Min.

        Users can customize this function to overwrite VisitMin_(const MinNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Min
            The Min to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_max_(self, op: Max) -> PrimExpr:
        """Visit Max.

        Users can customize this function to overwrite VisitMax_(const MaxNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Max
            The Max to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_eq_(self, op: EQ) -> PrimExpr:
        """Visit EQ.

        Users can customize this function to overwrite VisitEQ_(const EQNode* op)
        on the C++ side.

        Parameters
        ----------
        op : EQ
            The EQ to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ne_(self, op: NE) -> PrimExpr:
        """Visit NE.

        Users can customize this function to overwrite VisitNE_(const NENode* op)
        on the C++ side.

        Parameters
        ----------
        op : NE
            The NE to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_lt_(self, op: LT) -> PrimExpr:
        """Visit LT.

        Users can customize this function to overwrite VisitLT_(const LTNode* op)
        on the C++ side.

        Parameters
        ----------
        op : LT
            The LT to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_le_(self, op: LE) -> PrimExpr:
        """Visit LE.

        Users can customize this function to overwrite VisitLE_(const LENode* op)
        on the C++ side.

        Parameters
        ----------
        op : LE
            The LE to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_gt_(self, op: GT) -> PrimExpr:
        """Visit GT.

        Users can customize this function to overwrite VisitGT_(const GTNode* op)
        on the C++ side.

        Parameters
        ----------
        op : GT
            The GT to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ge_(self, op: GE) -> PrimExpr:
        """Visit GE.

        Users can customize this function to overwrite VisitGE_(const GENode* op)
        on the C++ side.

        Parameters
        ----------
        op : GE
            The GE to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_and_(self, op: And) -> PrimExpr:
        """Visit And.

        Users can customize this function to overwrite VisitAnd_(const AndNode* op)
        on the C++ side.

        Parameters
        ----------
        op : And
            The And to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_or_(self, op: Or) -> PrimExpr:
        """Visit Or.

        Users can customize this function to overwrite VisitOr_(const OrNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Or
            The Or to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_reduce_(self, op: Reduce) -> PrimExpr:
        """Visit Reduce.

        Users can customize this function to overwrite VisitReduce_(const ReduceNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Reduce
            The Reduce to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_cast_(self, op: Cast) -> PrimExpr:
        """Visit Cast.

        Users can customize this function to overwrite VisitCast_(const CastNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Cast
            The Cast to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_not_(self, op: Not) -> PrimExpr:
        """Visit Not.

        Users can customize this function to overwrite VisitNot_(const NotNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Not
            The Not to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_select_(self, op: Select) -> PrimExpr:
        """Visit Select.

        Users can customize this function to overwrite VisitSelect_(const SelectNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Select
            The Select to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_ramp_(self, op: Ramp) -> PrimExpr:
        """Visit Ramp.

        Users can customize this function to overwrite VisitRamp_(const RampNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Ramp
            The Ramp to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_broadcast_(self, op: Broadcast) -> PrimExpr:
        """Visit Broadcast.

        Users can customize this function to overwrite VisitBroadcast_(const BroadcastNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Broadcast
            The Broadcast to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_shuffle_(self, op: Shuffle) -> PrimExpr:
        """Visit Shuffle.

        Users can customize this function to overwrite VisitShuffle_(const ShuffleNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Shuffle
            The Shuffle to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_int_imm_(self, op: IntImm) -> PrimExpr:
        """Visit IntImm.

        Users can customize this function to overwrite VisitIntImm_(const IntImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : IntImm
            The IntImm to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_float_imm_(self, op: FloatImm) -> PrimExpr:
        """Visit FloatImm.

        Users can customize this function to overwrite VisitFloatImm_(const FloatImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : FloatImm
            The FloatImm to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> PrimExpr:
        """Visit StringImm.

        Users can customize this function to overwrite VisitStringImm_(const StringImmNode* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated PrimExpr.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore


class PyStmtExprMutatorWithAnalyzer(PyStmtExprMutator, _AnalyzerContextMixin):
    """A Python StmtExprMutator that maintains an arithmetic analyzer context.

    The analyzer is available as ``self.analyzer`` from user callbacks. The default
    mutation binds loop variables, block iter variables, let variables, and branch
    conditions before mutating nested nodes, so callbacks can query the analyzer
    using the surrounding IR context.
    """

    def __init__(self, analyzer=None):
        super().__init__()
        self._init_analyzer(analyzer)

    def visit_for_(self, op: For) -> Stmt:
        from tvm.ir import Range  # pylint: disable=import-outside-toplevel

        depth = len(self._constraint_scopes)
        min_value = self.visit_expr(op.min)
        extent = self.visit_expr(op.extent)
        step = self.visit_expr(op.step) if op.step is not None else None
        self.analyzer.bind(op.loop_var, Range.from_min_extent(min_value, extent))
        try:
            body = self.visit_stmt(op.body)
        finally:
            self._pop_constraints(depth)
        if (
            min_value.same_as(op.min)
            and extent.same_as(op.extent)
            and body.same_as(op.body)
            and (
                (step is None and op.step is None)
                or (step is not None and op.step is not None and step.same_as(op.step))
            )
        ):
            return op
        return For(
            op.loop_var,
            min_value,
            extent,
            op.kind,
            body,
            op.thread_binding,
            op.annotations,
            step,
            op.span,
        )

    def visit_attr_stmt_(self, op: AttrStmt) -> Stmt:
        from tvm.ir import Range  # pylint: disable=import-outside-toplevel

        depth = len(self._constraint_scopes)
        value = self.visit_expr(op.value)
        if op.attr_key in ("thread_extent", "virtual_thread") and isinstance(op.node, IterVar):
            self.analyzer.bind(op.node.var, Range.from_min_extent(IntImm(value.dtype, 0), value))
        try:
            body = self.visit_stmt(op.body)
        finally:
            self._pop_constraints(depth)
        if value.same_as(op.value) and body.same_as(op.body):
            return op
        return AttrStmt(op.node, op.attr_key, value, body, op.span)

    def visit_sblock_(self, op: SBlock) -> Stmt:
        depth = len(self._constraint_scopes)
        try:
            for iter_var in op.iter_vars:
                self.analyzer.bind(iter_var.var, iter_var.dom)
            return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore
        finally:
            self._pop_constraints(depth)

    def visit_bind_(self, op: Bind) -> Stmt:
        value = self.visit_expr(op.value)
        if self._is_pure(value):
            self.analyzer.bind(op.var, value)
        if value.same_as(op.value):
            return op
        return Bind(op.var, value, op.span)

    def visit_seq_stmt_(self, op: SeqStmt) -> Stmt:
        depth = len(self._constraint_scopes)
        try:
            seq = [self.visit_stmt(stmt) for stmt in op.seq]
        finally:
            self._pop_constraints(depth)
        if all(new.same_as(old) for new, old in zip(seq, op.seq)):
            return op
        return SeqStmt(seq, op.span)

    def visit_if_then_else_(self, op: IfThenElse) -> Stmt:
        condition = self.visit_expr(op.condition)
        real_condition = self._real_condition(condition)
        depth = len(self._constraint_scopes)
        with self.analyzer.constraint_scope(real_condition):
            try:
                then_case = self.visit_stmt(op.then_case)
            finally:
                self._pop_constraints(depth)
        else_case = None
        if op.else_case is not None:
            with self.analyzer.constraint_scope(self._negated_condition(real_condition)):
                try:
                    else_case = self.visit_stmt(op.else_case)
                finally:
                    self._pop_constraints(depth)
        if (
            condition.same_as(op.condition)
            and then_case.same_as(op.then_case)
            and (
                (else_case is None and op.else_case is None)
                or (
                    else_case is not None
                    and op.else_case is not None
                    and else_case.same_as(op.else_case)
                )
            )
        ):
            return op
        return IfThenElse(condition, then_case, else_case, op.span)

    def visit_assert_stmt_(self, op: AssertStmt) -> Stmt:
        condition = self.visit_expr(op.condition)
        self._push_constraint(condition)
        message_parts = [self.visit_expr(msg) for msg in op.message_parts]
        if condition.same_as(op.condition) and all(
            new.same_as(old) for new, old in zip(message_parts, op.message_parts)
        ):
            return op
        return AssertStmt(condition, op.error_kind, message_parts, op.span)

    def visit_let_(self, op: Let) -> PrimExpr:
        value = self.visit_expr(op.value)
        if self._is_pure(value):
            self.analyzer.bind(op.var, value)
        body = self.visit_expr(op.body)
        if value.same_as(op.value) and body.same_as(op.body):
            return op
        return Let(op.var, value, body, op.span)

    def visit_reduce_(self, op: Reduce) -> PrimExpr:
        for iter_var in op.axis:
            self.analyzer.bind(iter_var.var, iter_var.dom)
        return _ffi_api.PyStmtExprMutatorDefaultVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> PrimExpr:
        if getattr(op.op, "name", None) != "tirx.if_then_else":
            args = [self.visit_expr(arg) for arg in op.args]
            if all(new.same_as(old) for new, old in zip(args, op.args)):
                return op
            return Call(op.dtype, op.op, args, op.annotations, op.span)

        condition = self.visit_expr(op.args[0])
        with self.analyzer.constraint_scope(condition):
            true_value = self.visit_expr(op.args[1])
        with self.analyzer.constraint_scope(self._negated_condition(condition)):
            false_value = self.visit_expr(op.args[2])
        if (
            condition.same_as(op.args[0])
            and true_value.same_as(op.args[1])
            and false_value.same_as(op.args[2])
        ):
            return op
        return Call(op.dtype, op.op, [condition, true_value, false_value], op.annotations, op.span)

    def visit_select_(self, op: Select) -> PrimExpr:
        condition = self.visit_expr(op.condition)
        with self.analyzer.constraint_scope(condition):
            true_value = self.visit_expr(op.true_value)
        with self.analyzer.constraint_scope(self._negated_condition(condition)):
            false_value = self.visit_expr(op.false_value)
        if (
            condition.same_as(op.condition)
            and true_value.same_as(op.true_value)
            and false_value.same_as(op.false_value)
        ):
            return op
        return Select(condition, true_value, false_value, op.span)
