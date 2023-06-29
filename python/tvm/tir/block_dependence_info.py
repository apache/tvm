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
"""Define BlockDependenceInfoNode that uses the BlockScope and StmtSRef objects
to store the block level dependences"""

from typing import Union, Optional
from tvm._ffi import register_object
from tvm.ir.module import IRModule
from tvm.runtime import Object
from tvm.tir import Block, PrimFunc

from .block_scope import BlockScope, StmtSRef
from . import _ffi_api


@register_object("tir.BlockDependenceInfo")
class BlockDependenceInfo(Object):
    """
    BlockDependenceInfo
    An object that helps build and query block level dependences using the 2 core objects
    BlockScope and StmtSRef

    The data structures exposed are:
    1) sref2scope: Mapping from the srefs to its corresponding BlockScope
    2) stmt2ref: Mapping from blocks to corresponding StmtSRefs

    Note that this object does not store SRefs to loops as the purpose is only to expose block level
    dependences. This provides the advantage that the scope block (parent block) for a given block
    sref can be directly accessed as sref->parent
    """

    mod: IRModule

    def __init__(self, mod: Union[IRModule, PrimFunc]):
        if isinstance(mod, PrimFunc):
            mod = IRModule({"main": mod})
        if not isinstance(mod, IRModule):
            raise TypeError(f"Expected `mod` to be PrimFunc or IRModule, but gets: {mod}")
        self.__init_handle_by_constructor__(
            _ffi_api.BlockDependenceInfo,  # type: ignore # pylint: disable=no-member
            mod,
        )

    def get_sref(self, block: Block) -> Optional[StmtSRef]:
        """Return the corresponding sref that points to the block

        Parameters
        ----------
        stmt : Block
            The block for which the sref is to be retrived

        Returns
        -------
        sref : StmtSRef
            The corresponding sref
        """
        return _ffi_api.BlockDependenceInfoGetSRef(self, block)  # type: ignore # pylint: disable=no-member

    def get_block_scope(self, block_sref: StmtSRef) -> BlockScope:
        """Get the BlockScope correpsonding to the block sref

        Parameters
        ----------
        block_sref : StmtSRef
            The block sref to be retrieved

        Returns
        -------
        scope : StmtSRef
            The corresponding BlockScope
        """
        return _ffi_api.BlockDependenceInfoGetBlockScope(  # type: ignore # pylint: disable=no-member
            self, block_sref
        )
