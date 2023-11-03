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

from . import _ffi_api
from tvm import tir, ir
from typing import Union, List

from tvm.script import tir as T
from tvm.script.parser.tir import *

def Parallel(*extents: tir.PrimExpr):
    """Tools to construct nested parallel for loop.
       This can be used to create element-wise tensor expression.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ffi_api.Parallel(extents)  # type: ignore[attr-defined] # pylint: disable=no-member

def Pipelined(start: tir.PrimExpr, stop: tir.PrimExpr = None, num_stages: int = 0):
    """Tools to construct pipelined for loop.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.
    stop : PrimExpr
        The maximum value of iteration.
    num_stages : int
        The max number of buffer used between pipeline producers and consumers.
        if num_stages is 0, pipeline will not be enabled.
    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.Pipelined(start, stop, num_stages)  # type: ignore[attr-defined] # pylint: disable=no-member

def launch_program(*grid_size: List[int], num_threads: int):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    grid_size : List[int]
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    num_threads : int
        A integer representing blockDim.x
    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    """
    assert len(grid_size) >= 1 and len(grid_size) <= 3
    tx = T.launch_thread("threadIdx.x", num_threads)
    bx = T.launch_thread("blockIdx.x", grid_size[0])
    if len(grid_size) == 1:
        return bx, tx
    by = T.launch_thread("blockIdx.y", grid_size[1])
    if len(grid_size) == 2:
        return bx, by, tx
    bz = T.launch_thread("blockIdx.z", grid_size[2])
    return bx, by, bz, tx

def region(buffer: tir.Buffer, access_type: str, *args: tir.PrimExpr):
    access_type = {"r" : 1, "w" : 2, "rw": 3}[access_type]
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.region"),
        buffer.data, access_type, *args
    )

def buffer_to_tile_region(buffer: tir.Buffer, access_type: str):
    mins = [0 for _ in buffer.shape]
    extents = [x for x in buffer.shape]
    return region(buffer, access_type, *mins, *extents)

def buffer_load_to_tile_region(load: tir.BufferLoad, access_type: str, extents: List[tir.PrimExpr]):
    mins = [x for x in load.indices]
    return region(load.buffer, access_type, *mins, *extents)

def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str):
    mins = [x.min for x in buffer_region.region]
    extents = [x.extent for x in buffer_region.region]
    return region(buffer_region.buffer, access_type, *mins, *extents)

def copy(src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
         dst: Union[tir.Buffer, tir.BufferLoad],
        ):
    def get_extent(data):
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    # if src_extent and dst_extent:
    #     ir.assert_structural_equal(src_extent, dst_extent)
    if src_extent:
        extent = src_extent
    elif dst_extent:
        extent = dst_extent
    else:
        raise TypeError("Can't deduce copy extents from args")

    def _to_region(data, access_type):
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")

    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.copy"),
        src, dst
    )

class GemmWarpPolicy():
    Square = 0
    FullRow = 1
    FullCol = 2

def gemm(A: tir.Buffer, B: tir.Buffer, C: tir.Buffer,
         transpose_A: bool=False, transpose_B: bool=False, policy: GemmWarpPolicy=GemmWarpPolicy.Square):
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[0] if transpose_A else A.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    assert K == K_B, "gemm K shape check failed"
    Aptr = A.access_ptr("r")
    Bptr = B.access_ptr("r")
    Cptr = C.access_ptr("rw")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.gemm"),
        Aptr, Bptr, Cptr, transpose_A, transpose_B, M, N, K, policy
    )

def fill(buffer: tir.Buffer, value: tir.PrimExpr):
    buffer = buffer.access_ptr("w")
    return tir.call_intrin("handle", tir.op.Op.get("tl.fill"),
                           buffer, value)

def clear(buffer: tir.Buffer):
    return fill(buffer, 0)

def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
    buffer = buffer.access_ptr("r")
    out = out.access_ptr('w')
    return tir.call_intrin("handle", tir.op.Op.get("tl.reduce"),
                           buffer, out, reduce_type, dim, clear)

def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool=True):
    """Perform reduce max on input buffer, store the result to output buffer

    Parameters
    ----------
    buffer : Buffer
        The input buffer.
    out : Buffer
        The output buffer.
    dim : int
        The dimension to perform reduce on
    clear : bool
        If set to False, the output buffer will first be initialized to -inf.
    Returns
    -------
    handle : PrimExpr
    """
    return reduce(buffer, out, "max", dim, clear)

def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool=True):
    return reduce(buffer, out, "sum", dim, clear)
