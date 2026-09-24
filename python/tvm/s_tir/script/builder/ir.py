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
"""S-TIR block construction over shared primitive operations."""

import contextlib
import threading
from numbers import Integral
from typing import Any

from tvm import ir
from tvm import tirx as tir
from tvm.ir import TensorLoad, TensorRegion, is_prim_expr
from tvm.tirx import Buffer, Expr, IntImm, Var
from tvm.tirx.layout import Layout
from tvm.tirx.script.builder.ir import _get_layout, _record_meta_resource

from . import _ffi_api
from .frame import BlockInitFrame, SBlockFrame

_block_name_suffix = threading.local()


def _get_sblock_name_suffix() -> str:
    """Get the current block name suffix for macro expansion."""
    return getattr(_block_name_suffix, "value", "")


@contextlib.contextmanager
def block_name_suffix_context(block_suffix: str):
    """Context manager to set block name suffix during macro expansion.

    Parameters
    ----------
    block_suffix : str
        The suffix to append to block names (e.g., "_1", "_2").

    Yields
    ------
    None
    """
    old_suffix = getattr(_block_name_suffix, "value", "")
    _block_name_suffix.value = block_suffix
    try:
        yield
    finally:
        _block_name_suffix.value = old_suffix


def sblock(name: str = "", no_realize: bool = False, exec_scope: str = "") -> SBlockFrame:
    """The sblock declaration statement.

    Parameters
    ----------
    name : str
        The name of the sblock.

    no_realize : bool
        The flag whether to construct SBlockRealize or SBlock.

    exec_scope : str
        The execution scope of the block.

    Returns
    -------
    res : SBlockFrame
        The SBlockFrame.
    """
    block_suffix = _get_sblock_name_suffix()
    if block_suffix and name:
        name = name + block_suffix
    return _ffi_api.Block(name, no_realize, exec_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def init() -> BlockInitFrame:
    """The block initialization statement.

    Returns
    -------
    res : BlockInitFrame
        The BlockInitFrame.
    """
    return _ffi_api.Init()  # type: ignore[attr-defined] # pylint: disable=no-member


def where(predicate: Expr | int) -> None:
    """The block predicate statement.

    Parameters
    ----------
    predicate : Union[Expr, Literal[0, 1]]
        The predicate condition.
    """
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    if isinstance(predicate, int):
        if predicate in [0, 1]:
            predicate = IntImm("bool", predicate)
        else:
            raise ValueError(f"Invalid value for predicate: {predicate}")
    _ffi_api.Where(predicate)  # type: ignore[attr-defined] # pylint: disable=no-member


def reads(*buffer_slices: list[TensorRegion | TensorLoad]) -> None:
    """The block buffer region reading statement.

    Parameters
    ----------
    buffer_slices : List[Union[TensorRegion, TensorLoad]]
        The array of buffer regions to read.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore[assignment]
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore[assignment]
    _ffi_api.Reads(buffer_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def writes(*buffer_slices: list[TensorRegion | TensorLoad]) -> None:
    """The block buffer region writing statement.

    Parameters
    ----------
    buffer_slices : List[Union[TensorRegion, TensorLoad]]
        The array of buffer regions to write.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore[assignment]
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore[assignment]
    _ffi_api.Writes(buffer_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def sblock_attr(attrs: dict[str, Any]) -> None:
    """The block annotation statement (for non-tirx SBlock usage).

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotation of the block.
    """
    return _ffi_api.BlockAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def sblock_alloc_buffer(
    shape: list[Expr] | tuple[Expr] | Expr | Integral,
    dtype: str = "float32",
    data: Var = None,
    strides: list[Expr] | None = None,
    elem_offset: Expr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
) -> Buffer:
    """SBlock-level buffer allocation function.

    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The type of the buffer prior to flattening.
    dtype : str
        The data type in the content of the buffer.
    data : Var
        The pointer to the head of the data.
    strides : List[Expr]
        The strides of each dimension.
    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).
    scope : str
        The optional storage scope of buffer data pointer.
    align : int
        The alignment requirement of data pointer in bytes.
    offset_factor : int
        The factor of elem_offset field.
    layout: Optional[Union[str, Layout]]
        The layout of the buffer.

    allocated_addr: Optional[Union[int, Tuple[int]]]
        The address of the allocated buffer. Might be multi-dimensional.
        There can be pooled storage scopes on some devices. For example,
        the Trainium device has a pooled storage scope for the SRAN buffers. ("trn.sbuf")
        CUDA has a pooled storage scope for the shared memory ("shared.dyn")

    Returns
    -------
    res : Buffer
        The allocated buffer.
    """
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    if strides is not None:
        strides = [Var(s, "int64") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    buf = _ffi_api.SBlockAllocBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    _record_meta_resource(buf, skip_frames=2)
    return buf


def _as_range(dom: ir.Range | list[Expr]) -> ir.Range:
    """The range constructor.

    Parameters
    ----------
    dom : Union[Range, List[Expr]]
        The domain.

    Returns
    -------
    res : Range
        The Range.
    """
    if isinstance(dom, ir.Range):
        return dom
    if isinstance(dom, list | tuple):
        from tvm.sym import Analyzer  # pylint: disable=import-outside-toplevel

        extent = Analyzer().simplify(dom[1] - dom[0])
        if isinstance(extent, tir.IntImm):
            return ir.Range.from_min_extent(dom[0], extent)
        return ir.Range(dom[0], dom[1])
    if is_prim_expr(dom):
        return ir.Range(IntImm(dom.ty, 0), dom)
    return ir.Range(0, dom)


class axis:  # pylint: disable=invalid-name
    """The axis class"""

    @staticmethod
    def spatial(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr,
        dtype: str = "int32",
    ) -> Var:
        """The spatial block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisSpatial(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr,
        dtype: str = "int32",
    ) -> Var:
        """The reduced block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisReduce(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr,
        dtype: str = "int32",
    ) -> Var:
        """The scanning block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisScan(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr,
        dtype: str = "int32",
    ) -> Var:
        """The opaque block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisOpaque(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds: str, bindings: list[Expr], dtype: str = "int32") -> list[Var] | Var:
        """The block axis remapping function.

        Parameters
        ----------
        kinds : str
            The types of the iteration variables.

        bindings : List[Expr]
            The binding values of the iteration variables.

        dtype : str
            The data types of the iteration variables.

        Returns
        -------
        res : Var
            The iteration variables.
        """
        iter_vars = _ffi_api.AxisRemap(  # type: ignore[attr-defined] # pylint: disable=no-member
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name
