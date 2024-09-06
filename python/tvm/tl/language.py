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
"""The language interface for tl programs."""

from typing import Union, List, Optional
from tvm import tir
from tvm.script import tir as T
from tvm.script.parser.tir import *
from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from . import _ffi_api
from .layout import Layout, Fragment


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
    # type: ignore[attr-defined] # pylint: disable=no-member
    return _ffi_api.Pipelined(start, stop, num_stages)


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    def __enter__(self) -> Union[Var, List[Var]]:  # type: ignore[override]
        super().__enter__()
        if len(self.frames) == 3:
            return self.frames[0].iter_var.var
        return [frame.iter_var.var for frame in self.frames[0:-2]]


def Kernel(*blocks: List[tir.PrimExpr], threads: int = 128, 
           prelude:Optional[str]=None):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    blocks : List[int]
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        if the value is -1, we skip the threadIdx.x binding.
    prelude : str
        The import c code of the kernel, 
        will be injected before the generated kernel code.
    layout_annotation: Optional[Map[tir.Buffer, tir.IndexMap]]
        The layout annotation map, used to annotate the layout of the buffers.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    """
    attrs:dict = {}
    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, threads, attrs)


def use_swizzle(panel_size: int, order: str = "row"):
    device_func = (
        "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    )
    return T.attr(
        None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>"
    )


def alloc_shared(shape, dtype, scope="shared.dyn"):
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(shape, dtype, scope="local.fragment"):
    return T.alloc_buffer(shape, dtype, scope=scope)


def annotate_layout(layout_map):
    layout_map = {buffer.data: layout for buffer, layout in layout_map.items()}
    return T.block_attr({"layout_map": layout_map})


def import_source(source:str):
    return T.block_attr({"pragma_import_c": source})


def region(buffer: tir.BufferLoad, access_type: str, *args: tir.PrimExpr):
    access_type = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tir.call_intrin("handle", tir.op.Op.get("tl.region"), buffer, access_type, *args)


def buffer_to_tile_region(buffer: tir.Buffer, access_type: str):
    mins = [0 for _ in buffer.shape]
    extents = [x for x in buffer.shape]
    return region(T.BufferLoad(buffer, mins), access_type, *extents)


def buffer_load_to_tile_region(load: tir.BufferLoad, access_type: str, extents: List[tir.PrimExpr]):
    return region(load, access_type, *extents)


def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str):
    mins = [x.min for x in buffer_region.region]
    extents = [x.extent for x in buffer_region.region]
    return region(T.BufferLoad(buffer_region.buffer, mins), access_type, *extents)


def copy(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
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

    return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst)


def c2d_im2col(
    img: tir.Buffer,
    col: tir.Buffer,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
):
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.c2d_im2col"),
        img.access_ptr("r"),
        col.access_ptr("w"),
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
    )


class GemmWarpPolicy:
    Square = 0
    FullRow = 1
    FullCol = 2


def gemm(
    A: tir.Buffer,
    B: tir.Buffer,
    C: tir.Buffer,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
):
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[0] if transpose_A else A.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    assert K == K_B, "gemm K shape check failed"
    Aptr = A.access_ptr("r")
    Bptr = B.access_ptr("r")
    Cptr = C.access_ptr("rw")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.gemm"),
        Aptr,
        Bptr,
        Cptr,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
    )


def fill(buffer: tir.Buffer, value: tir.PrimExpr):
    buffer = buffer.access_ptr("w")
    return tir.call_intrin("handle", tir.op.Op.get("tl.fill"), buffer, value)


def clear(buffer: tir.Buffer):
    return fill(buffer, 0)


def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
    buffer = buffer.access_ptr("r")
    out = out.access_ptr("w")
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.reduce"), buffer, out, reduce_type, dim, clear
    )


def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True):
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


def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True):
    return reduce(buffer, out, "min", dim, clear)


def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int):
    return reduce(buffer, out, "sum", dim, True)


def atomic_add(dst, value):
    return T.call_extern("handle", "atomicAdd", T.address_of(dst), value)
