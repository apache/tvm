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
# pylint: disable=redefined-builtin
"""TVM Script nodes."""

from typing import Optional, Union, List, Callable
import synr
from tvm.arith import Analyzer
from tvm.runtime import ObjectGeneric, convert
from tvm.tir import PrimExpr, Buffer, BufferLoad, IntImm, Ramp, BufferRegion
from tvm.ir import Span, Range


class Slice:
    """A helper class to present slice information for BufferSlice

    Parameters
    ----------
    start : Union[PrimExpr, int]
        The start index.

    stop : Optional[Union[PrimExpr, int]]
        The stop index, None means the Slice is an element-wise index

    step : int
        The slice step

    span : Optional[Span]
        The location of the slice in the source.
    """

    start: Union[PrimExpr, int]
    stop: Optional[Union[PrimExpr, int]]
    step: int
    span: Optional[Span]

    def __init__(
        self,
        start: Union[PrimExpr, int],
        stop: Optional[Union[PrimExpr, int]] = None,
        step: int = 1,
        span: Optional[Span] = None,
    ):
        self.start = start
        self.stop = stop
        self.step = step
        self.span = span

    def as_index_expr(self, report_error: Callable[[str, Union[Span, synr.ast.Span]], None]):
        """Helper to create index PrimExpr from slice object
        Parameters
        ----------
        report_error: Callable[[str, Union[Span, synr.ast.Span]], None]
            The error report func
        """
        if self.stop is None:
            # scalar index
            return self.start
        if self.step < 1:
            report_error("Slice's step should be positive integer", self.span)
        lanes = Analyzer().simplify((self.stop - self.start + self.step - 1) // self.step)
        if not isinstance(lanes, (int, IntImm)):
            report_error("Slice's lanes should be constant for buffer indices", self.span)
        if lanes == 1:
            return self.start
        return Ramp(self.start, self.step, int(lanes), self.span)


class BufferSlice(ObjectGeneric):
    """A generic object for representing general buffer access. Following cases are supported:
        - element wise access buffer[i, j], which can be converted to BufferLoad if necessary
        - slice access buffer[i: i + 1, j : j + 2]
        - union of element and slice buffer[i, j: j + 2]

        This node is used in TVMScript to parse BufferLoad, BufferRegion and Realize

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    indices : List[Union[Slice, PrimExpr, int]]
        The access indexes can be slice, PrimExpr or int.

    report_error: Callable[[str, Union[Span, synr.ast.Span]], None]
        The error report func

    span : Optional[Span]
        The location of the buffer access in the source.
    """

    buffer: Buffer
    slices: List[Slice]
    report_error: Callable[[str, Union[Span, synr.ast.Span]], None]
    span: Optional[Span]

    def __init__(
        self,
        buffer: Buffer,
        indices: List[Union[Slice, PrimExpr, int]],
        report_error: Callable[[str, Union[Span, synr.ast.Span]], None],
        span: Optional[Span] = None,
    ):
        def check_index(index: Union[int, PrimExpr]):
            """Check input index is non-negative integer or PrimExpr"""
            if isinstance(index, int):
                if index < 0:
                    report_error("Negative index is not allowed during buffer access", span)
            elif isinstance(index, PrimExpr):
                element_dtype = index.dtype.split("x", maxsplit=1)[0]
                if element_dtype[:3] != "int":
                    report_error(
                        "index expected an integer type PrimExpr but got " + str(index.dtype),
                        index.span,
                    )
            else:
                report_error(
                    "Unsupported index type, expected int or tvm.tir.PrimExpr, but got "
                    + str(type(index)),
                    span,
                )

        slices: List[Union[Slice, BufferSlice]] = []
        for index in indices:
            if isinstance(index, Slice):
                index.start, index.stop = [convert(_) for _ in [index.start, index.stop]]
                check_index(index.start)
                check_index(index.stop)
                slices.append(index)
            elif isinstance(index, (PrimExpr, int)):
                check_index(index)
                slices.append(Slice(index))
            elif isinstance(index, BufferSlice):
                buffer_load = index.asobject()
                check_index(buffer_load)
                slices.append(Slice(buffer_load))
            else:
                report_error(
                    "Unsupported index type for BufferSlice, "
                    + "expected int, tvm.tir.PrimExpr, tvm.tir.Slice, but got "
                    + str(type(index)),
                    span,
                )

        self.buffer = buffer
        self.slices = slices
        self.report_error = report_error
        self.span = span

    def __str__(self):
        regions: List[str] = []
        for s in self.slices:
            if s.stop is None:
                regions.append(str(s.start))
            else:
                regions.append(str(s.start) + ": " + str(s.stop))

        return self.buffer.name + "[" + ", ".join(regions) + "]"

    def asobject(self) -> BufferLoad:
        """Convert object."""
        indices = [s.as_index_expr(self.report_error) for s in self.slices]
        return BufferLoad(self.buffer, indices, span=self.span)

    def as_buffer_region(self, analyzer: Optional[Analyzer] = None) -> BufferRegion:
        """Construct BufferRegion from BufferSlice

        Parameters
        ----------
        analyzer : Optional[tvm.arith.Analyzer]
            The analyzer for simplifying. If not provided, the method will construct a new one

        Returns
        -------
        buffer_region : BufferRegion
            The constructed BufferRegion.
        """
        region: List[Range] = []
        for s in self.slices:
            start = s.start if isinstance(s.start, PrimExpr) else IntImm("int32", s.start)
            extent = IntImm(start.dtype, 1) if s.stop is None else s.stop - s.start
            if not analyzer:
                analyzer = Analyzer()
            if isinstance(extent, PrimExpr):
                extent = analyzer.simplify(extent)
            if s.step != 1:
                self.report_error("BufferRegion do not support non-trivial stride", s.span)
            region.append(Range.from_min_extent(start, extent, span=s.span))
        return BufferRegion(self.buffer, region)

    def astype(self, dtype: str, span: Optional[Span] = None) -> PrimExpr:
        return self.asobject().astype(dtype, span)

    @property
    def dtype(self) -> str:
        """Return the dtype referenced by the slice.

        Implemented as a property so that ``slice.dtype`` has the same
        calling convention as ``primexpr.dtype``.  This allows a
        BufferSlice object can be assigned to a variable without
        requiring a type annotation on the variable, similar to other
        expressions.
        """
        return self.asobject().dtype
