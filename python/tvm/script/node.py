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

from tvm.runtime import ObjectGeneric
from tvm.tir import PrimExpr, Buffer, BufferLoad
from tvm.ir import Span


class Slice:
    """A helper class to present slice information for BufferSlice

    Parameters
    ----------
    start : Union[PrimExpr, int]
        The start index.

    stop : Optional[Union[PrimExpr, int]]
        The stop index, None means the Slice is an element-wise index

    span : Optional[Span]
        The location of the slice in the source.
    """

    start: Union[PrimExpr, int]
    stop: Optional[Union[PrimExpr, int]]
    span: Optional[Span]

    def __init__(
        self,
        start: Union[PrimExpr, int],
        stop: Optional[Union[PrimExpr, int]] = None,
        span: Optional[Span] = None,
    ):
        self.start = start
        self.stop = stop
        self.span = span


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
                if index.dtype != "int32":
                    report_error(
                        "index expected an int32 type PrimExpr but got " + str(index.dtype),
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
        for s in self.slices:
            if s.stop is not None:
                self.report_error("BufferLoad only accepts elementwise access", self.span)

        indices = [s.start for s in self.slices]
        return BufferLoad(self.buffer, indices, span=self.span)
