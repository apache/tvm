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
from tvm.ir import IRModule, PrimExpr
from tvm.runtime import Object
from tvm.tir import Block, For, IntImm, PrimFunc, Var

from . import _ffi_api_schedule
from .state import ScheduleState, StmtSRef


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

    def __init__(
        self,
        func_or_mod: Union[PrimFunc, IRModule],
        debug_mode: Union[bool, int] = False,
    ):
        """Construct a concrete TensorIR schedule from an IRModule or a PrimFunc

        Parameters
        ----------
        func_or_mod : Union[PrimFunc, IRModule]
            The IRModule or PrimFunc to be scheduled
        debug_mode : Union[bool, int]
            Do extra correctness checking after the class creation and each time
            scheduling primitive

        Note
        ----------
        The checks performed includes:
        1) VerifySRefTree
        2) VerifyAffineBinding
        3) VerifyRegionCover
        4) VerifyStagePipeline
        """
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        if not isinstance(debug_mode, int):
            raise TypeError(f"`debug_mode` should be integer or boolean, but gets: {debug_mode}")
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.ConcreteSchedule,  # pylint: disable=no-member
            func_or_mod,
            debug_mode,
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


@_register_object("tir.ConcreteSchedule")
class ConcreteSchedule(Schedule):
    """A concrete schedule class of TensorIR. Do not use directly, use tvm.tir.Schedule instead."""
