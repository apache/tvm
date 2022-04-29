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
"""Transformation on TIR schedule."""
from typing import Optional

from tvm.tir.schedule import Schedule, BlockRV, LoopRV
from . import _ffi_api


def tile_with_tensor_intrin(sch: Schedule, block: BlockRV, intrin_name: str) -> Optional[LoopRV]:
    """Tile a subset of loops in the block according to the given tensor intrinsic.

    Parameters
    ----------
    sch : Schedule
        The schedule to which tiling is applied
    block : BlockRV
        The block whose subset of loops will be tiled
    intrin_name : str
        The name of a tensor intrinsic, must be registerd via TensorIntrin.register(...) beforehand

    Returns
    -------
    tiled_loop_rv : Optional[LoopRV]
        LoopRV corresponding to the outermost loop of a block tiled according to the given intrin
        NullOpt if no valid loop mapping is found
    """
    return _ffi_api.TileWithTensorIntrin(sch, block, intrin_name)  # type: ignore
