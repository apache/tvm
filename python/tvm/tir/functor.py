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
"""The expression and statement functor of TIR."""
from typing import Callable

import tvm
from tvm.ir import PrimExpr
from tvm.runtime import Object
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
    Cast,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
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
    Allocate,
    AllocateConst,
    AssertStmt,
    AttrStmt,
    Block,
    BlockRealize,
    BufferRealize,
    BufferStore,
    DeclBuffer,
    Evaluate,
    For,
    IfThenElse,
    LetStmt,
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

    @tir.functor.stmt_expr_visitor
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

    @tir.functor.stmt_expr_mutator
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


@tvm.ffi.register_object("tir.PyStmtExprVisitor")
class _PyStmtExprVisitor(Object):
    """
    An internal wrapper to interface between C++ and Python StmtExprVisitor.
    This is the TVM object that wraps PyStmtExprVisitor.

    Do not use this class directly. Use PyStmtExprVisitor instead.

    See also: PyStmtExprVisitor, stmt_expr_visitor
    """

    def __init__(
        self,
        f_visit_stmt: Callable = None,
        f_visit_expr: Callable = None,
        # Stmt
        f_visit_let_stmt: Callable = None,
        f_visit_attr_stmt: Callable = None,
        f_visit_if_then_else: Callable = None,
        f_visit_for: Callable = None,
        f_visit_while: Callable = None,
        f_visit_allocate: Callable = None,
        f_visit_allocate_const: Callable = None,
        f_visit_decl_buffer: Callable = None,
        f_visit_buffer_store: Callable = None,
        f_visit_buffer_realize: Callable = None,
        f_visit_assert_stmt: Callable = None,
        f_visit_seq_stmt: Callable = None,
        f_visit_evaluate: Callable = None,
        f_visit_block: Callable = None,
        f_visit_block_realize: Callable = None,
        # PrimExpr
        f_visit_var: Callable = None,
        f_visit_size_var: Callable = None,
        f_visit_buffer_load: Callable = None,
        f_visit_producer_load: Callable = None,
        f_visit_let: Callable = None,
        f_visit_call: Callable = None,
        f_visit_add: Callable = None,
        f_visit_sub: Callable = None,
        f_visit_mul: Callable = None,
        f_visit_div: Callable = None,
        f_visit_mod: Callable = None,
        f_visit_floor_div: Callable = None,
        f_visit_floor_mod: Callable = None,
        f_visit_min: Callable = None,
        f_visit_max: Callable = None,
        f_visit_eq: Callable = None,
        f_visit_ne: Callable = None,
        f_visit_lt: Callable = None,
        f_visit_le: Callable = None,
        f_visit_gt: Callable = None,
        f_visit_ge: Callable = None,
        f_visit_and: Callable = None,
        f_visit_or: Callable = None,
        f_visit_reduce: Callable = None,
        f_visit_cast: Callable = None,
        f_visit_not: Callable = None,
        f_visit_select: Callable = None,
        f_visit_ramp: Callable = None,
        f_visit_broadcast: Callable = None,
        f_visit_shuffle: Callable = None,
        f_visit_int_imm: Callable = None,
        f_visit_float_imm: Callable = None,
        f_visit_string_imm: Callable = None,
    ) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.MakePyStmtExprVisitor,  # type: ignore
            f_visit_stmt,
            f_visit_expr,
            # Stmt
            f_visit_let_stmt,
            f_visit_attr_stmt,
            f_visit_if_then_else,
            f_visit_for,
            f_visit_while,
            f_visit_allocate,
            f_visit_allocate_const,
            f_visit_decl_buffer,
            f_visit_buffer_store,
            f_visit_buffer_realize,
            f_visit_assert_stmt,
            f_visit_seq_stmt,
            f_visit_evaluate,
            f_visit_block,
            f_visit_block_realize,
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
            "visit_let_stmt_",
            "visit_attr_stmt_",
            "visit_if_then_else_",
            "visit_for_",
            "visit_while_",
            "visit_allocate_",
            "visit_allocate_const_",
            "visit_decl_buffer_",
            "visit_buffer_store_",
            "visit_buffer_realize_",
            "visit_assert_stmt_",
            "visit_seq_stmt_",
            "visit_evaluate_",
            "visit_block_",
            "visit_block_realize_",
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

    def visit_let_stmt_(self, op: LetStmt) -> None:
        """Visit LetStmt.
        Users can customize this function to overwrite VisitStmt_(const LetStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : LetStmt
            The LetStmt to be visited.
        """
        print("visit_let_stmt_", op)
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

    def visit_allocate_(self, op: Allocate) -> None:
        """Visit Allocate.
        Users can customize this function to overwrite VisitStmt_(const AllocateNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Allocate
            The Allocate to be visited.
        """
        print("visit_allocate_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_allocate_const_(self, op: AllocateConst) -> None:
        """Visit AllocateConst.
        Users can customize this function to overwrite VisitStmt_(const AllocateConstNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AllocateConst
            The AllocateConst to be visited.
        """
        print("visit_allocate_const_", op)
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

    def visit_buffer_realize_(self, op: BufferRealize) -> None:
        """Visit BufferRealize.
        Users can customize this function to overwrite VisitStmt_(const BufferRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferRealize
            The BufferRealize to be visited.
        """
        print("visit_buffer_realize_", op)
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

    def visit_block_(self, op: Block) -> None:
        """Visit Block.
        Users can customize this function to overwrite VisitStmt_(const BlockNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Block
            The Block to be visited.
        """
        print("visit_block_", op)
        _ffi_api.PyStmtExprVisitorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_block_realize_(self, op: BlockRealize) -> None:
        """Visit BlockRealize.
        Users can customize this function to overwrite VisitStmt_(const BlockRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BlockRealize
            The BlockRealize to be visited.
        """
        print("visit_block_realize_", op)
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


@tvm.ffi.register_object("tir.PyStmtExprMutator")
class _PyStmtExprMutator(Object):
    """
    A TVM object to support customization of StmtExprMutator on the python side.
    This is the decorated result returned from stmt_expr_mutator decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: stmt_expr_mutator, PyStmtExprMutator
    """

    def __init__(
        self,
        f_visit_stmt: Callable = None,
        f_visit_expr: Callable = None,
        # Stmt
        f_visit_let_stmt: Callable = None,
        f_visit_attr_stmt: Callable = None,
        f_visit_if_then_else: Callable = None,
        f_visit_for: Callable = None,
        f_visit_while: Callable = None,
        f_visit_allocate: Callable = None,
        f_visit_allocate_const: Callable = None,
        f_visit_decl_buffer: Callable = None,
        f_visit_buffer_store: Callable = None,
        f_visit_buffer_realize: Callable = None,
        f_visit_assert_stmt: Callable = None,
        f_visit_seq_stmt: Callable = None,
        f_visit_evaluate: Callable = None,
        f_visit_block: Callable = None,
        f_visit_block_realize: Callable = None,
        # PrimExpr
        f_visit_var: Callable = None,
        f_visit_size_var: Callable = None,
        f_visit_buffer_load: Callable = None,
        f_visit_producer_load: Callable = None,
        f_visit_let: Callable = None,
        f_visit_call: Callable = None,
        f_visit_add: Callable = None,
        f_visit_sub: Callable = None,
        f_visit_mul: Callable = None,
        f_visit_div: Callable = None,
        f_visit_mod: Callable = None,
        f_visit_floor_div: Callable = None,
        f_visit_floor_mod: Callable = None,
        f_visit_min: Callable = None,
        f_visit_max: Callable = None,
        f_visit_eq: Callable = None,
        f_visit_ne: Callable = None,
        f_visit_lt: Callable = None,
        f_visit_le: Callable = None,
        f_visit_gt: Callable = None,
        f_visit_ge: Callable = None,
        f_visit_and: Callable = None,
        f_visit_or: Callable = None,
        f_visit_reduce: Callable = None,
        f_visit_cast: Callable = None,
        f_visit_not: Callable = None,
        f_visit_select: Callable = None,
        f_visit_ramp: Callable = None,
        f_visit_broadcast: Callable = None,
        f_visit_shuffle: Callable = None,
        f_visit_int_imm: Callable = None,
        f_visit_float_imm: Callable = None,
        f_visit_string_imm: Callable = None,
    ) -> None:
        """Constructor."""
        self.__init_handle_by_constructor__(
            _ffi_api.MakePyStmtExprMutator,  # type: ignore
            f_visit_stmt,
            f_visit_expr,
            # Stmt
            f_visit_let_stmt,
            f_visit_attr_stmt,
            f_visit_if_then_else,
            f_visit_for,
            f_visit_while,
            f_visit_allocate,
            f_visit_allocate_const,
            f_visit_decl_buffer,
            f_visit_buffer_store,
            f_visit_buffer_realize,
            f_visit_assert_stmt,
            f_visit_seq_stmt,
            f_visit_evaluate,
            f_visit_block,
            f_visit_block_realize,
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
            "visit_let_stmt_",
            "visit_attr_stmt_",
            "visit_if_then_else_",
            "visit_for_",
            "visit_while_",
            "visit_allocate_",
            "visit_allocate_const_",
            "visit_decl_buffer_",
            "visit_buffer_store_",
            "visit_buffer_realize_",
            "visit_assert_stmt_",
            "visit_seq_stmt_",
            "visit_evaluate_",
            "visit_block_",
            "visit_block_realize_",
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

    def visit_let_stmt_(self, op: LetStmt) -> Stmt:
        """Visit LetStmt.
        Users can customize this function to overwrite VisitStmt_(const LetStmtNode* op)
        on the C++ side.

        Parameters
        ----------
        op : LetStmt
            The LetStmt to be visited.

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

    def visit_allocate_(self, op: Allocate) -> Stmt:
        """Visit Allocate.
        Users can customize this function to overwrite VisitStmt_(const AllocateNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Allocate
            The Allocate to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_allocate_const_(self, op: AllocateConst) -> Stmt:
        """Visit AllocateConst.
        Users can customize this function to overwrite VisitStmt_(const AllocateConstNode* op)
        on the C++ side.

        Parameters
        ----------
        op : AllocateConst
            The AllocateConst to be visited.

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

    def visit_buffer_realize_(self, op: BufferRealize) -> Stmt:
        """Visit BufferRealize.
        Users can customize this function to overwrite VisitStmt_(const BufferRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BufferRealize
            The BufferRealize to be visited.

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

    def visit_block_(self, op: Block) -> Stmt:
        """Visit Block.
        Users can customize this function to overwrite VisitStmt_(const BlockNode* op)
        on the C++ side.

        Parameters
        ----------
        op : Block
            The Block to be visited.

        Returns
        -------
        result : Stmt
            The mutated Stmt.
        """
        return _ffi_api.PyStmtExprMutatorDefaultVisitStmt(self._outer(), op)  # type: ignore

    def visit_block_realize_(self, op: BlockRealize) -> Stmt:
        """Visit BlockRealize.
        Users can customize this function to overwrite VisitStmt_(const BlockRealizeNode* op)
        on the C++ side.

        Parameters
        ----------
        op : BlockRealize
            The BlockRealize to be visited.

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
