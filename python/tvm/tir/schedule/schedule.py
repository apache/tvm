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
# pylint: disable=unused-import
"""The TensorIR schedule class"""
from typing import List, Optional, Union

from tvm._ffi import register_object as _register_object
from tvm.error import TVMError, register_error
from tvm.ir import IRModule, PrimExpr
from tvm.runtime import Object
from tvm.tir import Block, For, IntImm, PrimFunc, Var

from . import _ffi_api_schedule
from .state import ScheduleState, StmtSRef


@register_error
class ScheduleError(TVMError):
    """Error that happens during TensorIR scheduling."""


@_register_object("tir.LoopRV")
class LoopRV(Object):
    """A random variable that refers to a loop"""


@_register_object("tir.BlockRV")
class BlockRV(Object):
    """A random variable that refers to a block"""


ExprRV = PrimExpr  #  A random variable that evaluates to an integer

RAND_VAR_TYPE = Union[ExprRV, BlockRV, LoopRV]  # pylint: disable=invalid-name


@_register_object("tir.Schedule")
class Schedule(Object):
    """The user-facing schedule class

    A schedule is a set of transformations that change the order of computation but
    preserve the semantics of computation. Some example of schedules:
    1) Split a loop into two;
    2) Reorder two loops;
    3) Inline the computation of a specific buffer into its consumer

    The schedule class stores auxiliary information to schedule correctly and efficiently.

    Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
    """

    ERROR_RENDER_LEVEL = {"detail": 0, "fast": 1, "none": 2}

    def __init__(
        self,
        func_or_mod: Union[PrimFunc, IRModule],
        *,
        debug_mode: Union[bool, int] = False,
        error_render_level: str = "detail",
    ):
        """Construct a concrete TensorIR schedule from an IRModule or a PrimFunc

        Parameters
        ----------
        func_or_mod : Union[PrimFunc, IRModule]
            The IRModule or PrimFunc to be scheduled
        debug_mode : Union[bool, int]
            Do extra correctness checking after the class creation and each time
            scheduling primitive
        error_render_level : str = "detail"
            The level of error rendering. Choices: "detail", "fast", "none".
            "detail": Render a detailed error message, with the TIR and error locations printed
            "fast: Show a simple error message without rendering or string manipulation
            "none": Do not show any error message.

        Note
        ----------
        The checks performed includes:
        1) VerifySRefTree
        2) VerifyCachedFlags
        """
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        if not isinstance(debug_mode, int):
            raise TypeError(f"`debug_mode` should be integer or boolean, but gets: {debug_mode}")
        if error_render_level not in Schedule.ERROR_RENDER_LEVEL:
            raise ValueError(
                'error_render_level can be "detail", "fast", or "none", but got: '
                + f"{error_render_level}"
            )
        error_render_level = Schedule.ERROR_RENDER_LEVEL.get(error_render_level)
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.ConcreteSchedule,  # pylint: disable=no-member
            func_or_mod,
            debug_mode,
            error_render_level,
        )

    ########## Utilities ##########

    @property
    def mod(self) -> IRModule:
        """Returns the AST of the module being scheduled"""
        return _ffi_api_schedule.ScheduleModule(self)  # pylint: disable=no-member

    @property
    def state(self) -> ScheduleState:
        """Returns the ScheduleState in the current schedule class"""
        return _ffi_api_schedule.ScheduleGetState(self)  # pylint: disable=no-member

    def copy(self) -> "Schedule":
        """Returns a copy of the schedule, including both the state and the symbol table,
        * guaranteeing that
        * 1) SRef tree is completely reconstructed;
        * 2) The IRModule being scheduled is untouched;
        * 3) All the random variables are valid in the copy, pointing to the correpsonding sref
        * reconstructed
        Returns
        -------
        copy : Schedule
            A new copy of the schedule
        """
        return _ffi_api_schedule.ScheduleCopy(self)  # pylint: disable=no-member

    def seed(self, seed: int) -> None:
        """Seed the randomness
        Parameters
        ----------
        seed : int
            The new random seed, -1 if use device random, otherwise non-negative
        """
        return _ffi_api_schedule.ScheduleSeed(self, seed)  # pylint: disable=no-member

    def show(self, rand_var: RAND_VAR_TYPE) -> str:
        """Returns a string representation of the value that the random variable evaluates to
        Parameters
        ----------
        rand_var : Union[ExprRV, BlockRV, LoopRV]
            The random variable to be evaluated
        Returns
        ----------
        str_repr : str
            The string representation
        """
        return str(self.get(rand_var))

    ########## Lookup ##########

    def get(
        self,
        rand_var_or_sref: Union[RAND_VAR_TYPE, StmtSRef],
    ) -> Optional[Union[int, Block, For]]:
        """Returns:
        - the corresponding Block that a BlockRV evaluates to;
        - the corresponding For that a LoopRV evaluates to;
        - the corresponding integer that a ExprRV evaluates to;
        - the corresponding Block that a block sref points to;
        - the corresponding For that a loop sref points to;
        Parameters
        ----------
        rand_var_or_sref : Union[ExprRV, BlockRV, LoopRV, StmtSRef]
            The random variable / sref to be evaluated
        Returns
        ----------
        result : Optional[Union[int, Block, For]]
            The correpsonding result
        """
        if isinstance(rand_var_or_sref, StmtSRef):
            return rand_var_or_sref.stmt
        result = _ffi_api_schedule.ScheduleGet(self, rand_var_or_sref)  # pylint: disable=no-member
        if isinstance(result, IntImm):
            result = result.value
        return result

    def get_sref(self, rand_var_or_stmt: Union[BlockRV, LoopRV, Block, For]) -> Optional[StmtSRef]:
        """Returns the correpsonding sref to the given
        1) LoopRV
        2) BlockRV
        3) Block
        4) For
        Parameters
        ----------
        rand_var_or_stmt : Union[BlockRV, LoopRV, Block, For]
            The random variable / sref to be evaluated
        Returns
        ----------
        result : Optional[StmtSRef]
            The correpsonding result
        """
        return _ffi_api_schedule.ScheduleGetSRef(  # pylint: disable=no-member
            self, rand_var_or_stmt
        )

    def remove_rv(self, rand_var: RAND_VAR_TYPE) -> None:
        """Remove a random variable from the symbol table
        Parameters
        ----------
        rand_var : Union[BlockRV, LoopRV, ExprRV]
            The random variable to be removed
        """
        return _ffi_api_schedule.ScheduleRemoveRV(self, rand_var)  # pylint: disable=no-member

    ########## Block/Loop relation ##########

    def get_block(
        self,
        name: str,
        func_name: str = "main",
    ) -> BlockRV:
        """Retrieve a block in a specific function with its name
        Parameters
        ----------
        name : str
            The name of the block
        func_name : str = "main"
            The name of the function
        Returns
        ----------
        block : BlockRV
            The block retrieved
            IndexError is raised if 0 or multiple blocks exist with the specific name.
        """
        return _ffi_api_schedule.ScheduleGetBlock(  # pylint: disable=no-member
            self,
            name,
            func_name,
        )

    def get_loops(self, block: BlockRV) -> List[LoopRV]:
        """Get the parent loops of the block in its scope, from outer to inner
        Parameters
        ----------
        block : BlockRV
            The query block
        Returns
        ----------
        loops : List[LoopRV]
            A list of loops above the given block in its scope, from outer to inner
        """
        return _ffi_api_schedule.ScheduleGetLoops(self, block)  # pylint: disable=no-member

    ########## Schedule: loops manipulation ##########
    ########## Schedule: compute location ##########
    def compute_inline(self, block: BlockRV) -> None:
        """Inline a block into its consumer(s). It requires:

        1) The block is a complete non-root block, which only produces one buffer

        2) The block must not be the only leaf in the scope.

        3) The body of the block must be a BufferStore statement in
           the form of, ``A[i, j, k, ...] = ...`` where the indices of
           the LHS are all distinct atomic variables, and no variables
           other than those indexing variables are allowed in the
           statement.

        Parameters
        ----------
        block : BlockRV
            The block to be inlined to its consumer(s)

        Examples
        --------

        Before compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @tvm.script.tir
            def before_inline(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, (128, 128))
                B = tir.alloc_buffer((128, 128))
                C = tir.match_buffer(c, (128, 128))
                with tir.block([128, 128], "B") as [vi, vj]:
                    B[vi, vj] = A[vi, vj] * 2.0
                with tir.block([128, 128], "C") as [vi, vj]:
                    C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do compute-inline:

        .. code-block:: python

            sch = tir.Schedule(before_inline, debug_mode=True)
            sch.compute_inline(sch.get_block("B"))
            print(tvm.script.asscript(sch.mod["main"]))

        After applying compute-inline, the IR becomes:

        .. code-block:: python

            @tvm.script.tir
            def after_inline(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, (128, 128))
                C = tir.match_buffer(c, (128, 128))
                with tir.block([128, 128], "C") as [vi, vj]:
                    C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        _ffi_api_schedule.ScheduleComputeInline(self, block)  # pylint: disable=no-member

    def reverse_compute_inline(self, block: BlockRV) -> None:
        """Inline a block into its only producer. It requires:

        1) The block is a complete non-root block, which only produces and consumes one buffer

        2) The block must not be the only leaf in the scope.

        3) The only producer of the block is a read-after-write producer and a
           complete non-root block

        4) The body of the block must be a BufferStore statement in the form of,
           ``B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)`` where the
           indices of each `BufferLoad` on the RHS are all distinct atomic
           variables, and no variables other than those indexing variables are
           allowed in the statement.

        Parameters
        ----------
        block : BlockRV
            The block to be inlined to its producer

        Examples
        --------

        Before reverse-compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @tvm.script.tir
            def before_inline(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, (128, 128))
                B = tir.alloc_buffer((128, 128))
                C = tir.match_buffer(c, (128, 128))
                with tir.block([128, 128], "B") as [vi, vj]:
                    B[vi, vj] = A[vi, vj] * 2.0
                with tir.block([128, 128], "C") as [vi, vj]:
                    C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do reverse-compute-inline:

        .. code-block:: python

            sch = tir.Schedule(before_inline, debug_mode=True)
            sch.reverse_compute_inline(sch.get_block("C"))
            print(tvm.script.asscript(sch.mod["main"]))

        After applying reverse-compute-inline, the IR becomes:

        .. code-block:: python

            @tvm.script.tir
            def after_inline(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, (128, 128))
                C = tir.match_buffer(c, (128, 128))
                with tir.block([128, 128], "C") as [vi, vj]:
                    C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        _ffi_api_schedule.ScheduleReverseComputeInline(self, block)  # pylint: disable=no-member

    ########## Schedule: loop binding/annotation ##########
    ########## Schedule: cache read/write ##########
    ########## Schedule: reduction ##########
    ########## Schedule: blockize & tensorize ##########


@_register_object("tir.ConcreteSchedule")
class ConcreteSchedule(Schedule):
    """A concrete schedule class of TensorIR. Do not use directly, use tvm.tir.Schedule instead."""
