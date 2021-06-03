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
"""This file defines ScheduleState, the core data structure of TensorIR scheduling."""
from collections import namedtuple
from enum import IntEnum
from typing import Dict, Optional, Union

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.tir import Block, BlockRealize, For, PrimFunc

from . import _ffi_api_schedule
from .block_scope import BlockScope, StmtSRef

CachedFlags = namedtuple("CachedFlags", ["affine_binding", "region_cover", "stage_pipeline"])


class ScheduleDebugMask(IntEnum):
    """The bitmask of the `debug_mode` flag in the ScheduleState class.

    If the `debug_mode` flag has a certain bit on, then the correpsonding
    verification pass will be conducted. For example, if `(debug_mode & VERIFY_SREF_TREE) != 0`,
    then the correctness of the sref tree will be verified after each schedule instruction.

    Attributes
    ----------
    VERIFY_SREF_TREE : int = 1
        Verify the correctness of the sref tree
    VERIFY_CACHED_FLAGS : int = 2
        Verify the correctness of affine_binding, region_cover and stage_pipeline
    """

    VERIFY_SREF_TREE = 1
    VERIFY_CACHED_FLAGS = 2


@register_object("tir.ScheduleState")
class ScheduleState(Object):
    """The state of scheduling, which exposes a `Replace` method as
    the primary resort for all the scheduling primitives to manipulate the TensorIR.

    The data structure contains the following information
    1) The AST being scheduled (mod)
    2) The sref tree of schedulable statements (indicated by the srefs)
    3) The dependency information of each block scope (block_info)
    4) A reverse mapping from the AST nodes to that in the sref tree (get_sref)
    5) A debug flag, if set, extra checking is enabled (debug_mode)

    Parameters
    ----------
    mod : IRModule
        The AST of the module being scheduled
    debug_mode : int
        Do extra correctness checking after the object construction
        and each time after calling the Replace method.
    """

    mod: IRModule
    debug_mode: int

    def __init__(
        self,
        func_or_mod: Union[PrimFunc, IRModule],
        debug_mode: Union[bool, int] = False,
    ):
        """Construct a schedule state from an IRModule or a PrimFunc

        Parameters
        ----------
        func_or_mod : Union[PrimFunc, IRModule]
            The IRModule or PrimFunc to be scheduled
        debug_mode : Union[bool, int]
            Do extra correctness checking after the class creation and each time
            after calling the Replace method.
            Possible choices of `debug_mode`:
            1) True - Turn on all the checks
            2) False - Turn off all the checks
            3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
        """
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        if not isinstance(debug_mode, int):
            raise TypeError(f"`debug_mode` should be integer or boolean, but gets: {debug_mode}")
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.ScheduleState,  # pylint: disable=no-member
            func_or_mod,
            debug_mode,
        )

    def get_sref(self, stmt: Union[Block, For]) -> Optional[StmtSRef]:
        """Return the corresponding sref that points to the stmt

        Parameters
        ----------
        stmt : Union[Block, For]
            The schedulable statement in the TensorIR to be retrieved for its sref

        Returns
        -------
        sref : StmtSRef
            The corresponding sref
        """
        return _ffi_api_schedule.ScheduleStateGetSRef(self, stmt)  # pylint: disable=no-member

    def get_block_scope(self, block_sref: StmtSRef) -> BlockScope:
        """Get the BlockScope correpsonding to the block sref

        Parameters
        ----------
        block_sref : StmtSRef
            The block sref to be retrieved

        Returns
        -------
        sref : StmtSRef
            The corresponding sref
        """
        return _ffi_api_schedule.ScheduleStateGetBlockScope(  # pylint: disable=no-member
            self, block_sref
        )

    def _get_cached_flags(self, block_sref: StmtSRef) -> CachedFlags:
        """Get the cached flags of the corresponding block

        Parameters
        ----------
        block_sref : StmtSRef
            The block sref to be retrieved

        Returns
        -------
        flags : CachedFlags
            Three flags: affine_binding, region_cover, stage_pipeline

        Note
        -------
        It is an API intended for internal testing use.
        """
        (
            affine_binding,
            region_cover,
            stage_pipeline,
        ) = _ffi_api_schedule.ScheduleStateGetCachedFlags(  # pylint: disable=no-member
            self, block_sref
        )
        return CachedFlags(
            affine_binding=bool(affine_binding.value),
            region_cover=bool(region_cover.value),
            stage_pipeline=bool(stage_pipeline.value),
        )

    def replace(
        self,
        src_sref: StmtSRef,
        tgt_stmt: Union[Block, For, BlockRealize],
        block_sref_reuse: Optional[Dict[Block, Block]] = None,
    ) -> None:
        """
        Replace the part of the AST, as being pointed to by `src_sref`,
        with a specific statement `tgt_stmt`, and maintain the sref tree accordingly.
        Replace will try to perform copy on write as much as possible when the ScheduleState holds
        the only copy to the IRModule and IR nodes.

        Only 3 types of replacements are allowed: from `src_sref->stmt` to `tgt_stmt`.
        1) Block -> Block
        2) Loop -> Loop
        3) Loop -> BlockRealize

        Parameters
        ----------
        src_sref : StmtSRef
            The sref to the statement to be replaced in the TensorIR AST

        tgt_stmt : Union[Block, For, BlockRealize]
            The statement to be replaced to

        block_sref_reuse : Optional[Dict[Block, Block]] = None
            Maps an old block (to be replaced in the subtree under `src_sref->stmt`)
            to a new block (replaced to, in the subtree under `tgt_stmt`), and enforces
            reuse of srefs between them (rather than create new srefs) i.e. after being replaced,
            the sref that points to the old block will point to the new one

        Note
        ----------
        The reuse of loop srefs are detected automatically according to the reuse of loop vars.
        """
        if block_sref_reuse is None:
            block_sref_reuse = {}
        _ffi_api_schedule.ScheduleStateReplace(  # pylint: disable=no-member
            self,
            src_sref,
            tgt_stmt,
            block_sref_reuse,
        )
