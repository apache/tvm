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
# pylint: disable=invalid-name
"""Various type definitions to help instantiate CUTLASS kernels."""
import re
import enum
from enum import auto as enum_auto


class GeneratorTarget(enum.Enum):
    Library = enum_auto()


class DataType(enum.Enum):
    f16 = enum_auto()
    f32 = enum_auto()


ShortDataTypeNames = {
    DataType.f16: "h",
    DataType.f32: "s",
}


DataTypeNames = {
    DataType.f16: "f16",
    DataType.f32: "f32",
}

DataTypeTag = {
    DataType.f16: "cutlass::half_t",
    DataType.f32: "float",
}

DataTypeSize = {
    DataType.f16: 16,
    DataType.f32: 32,
}


class MathOperation(enum.Enum):
    multiply_add = enum_auto()


MathOperationTag = {
    MathOperation.multiply_add: "cutlass::arch::OpMultiplyAdd",
}


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()


LayoutTag = {
    LayoutType.ColumnMajor: "cutlass::layout::ColumnMajor",
    LayoutType.RowMajor: "cutlass::layout::RowMajor",
}


TransposedLayout = {
    LayoutType.ColumnMajor: LayoutType.RowMajor,
    LayoutType.RowMajor: LayoutType.ColumnMajor,
}


ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "n",
    LayoutType.RowMajor: "t",
}


class OpcodeClass(enum.Enum):
    Simt = enum_auto()
    TensorOp = enum_auto()
    WmmaTensorOp = enum_auto()


OpcodeClassNames = {
    OpcodeClass.Simt: "simt",
    OpcodeClass.TensorOp: "tensorop",
    OpcodeClass.WmmaTensorOp: "wmma_tensorop",
}

OpcodeClassTag = {
    OpcodeClass.Simt: "cutlass::arch::OpClassSimt",
    OpcodeClass.TensorOp: "cutlass::arch::OpClassTensorOp",
    OpcodeClass.WmmaTensorOp: "cutlass::arch::OpClassWmmaTensorOp",
}


class OperationKind(enum.Enum):
    Gemm = enum_auto()


OperationKindNames = {
    OperationKind.Gemm: "gemm",
}


class Target(enum.Enum):
    library = enum_auto()


def substitute_template(template, values):
    """Instantiate a kernel template using `values`."""
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class GemmKind(enum.Enum):
    Gemm = enum_auto()


GemmKindNames = {
    GemmKind.Gemm: "gemm",
}


class EpilogueFunctor(enum.Enum):
    LinearCombination = enum_auto()
    LinearCombinationRelu = enum_auto()
    LinearCombinationBias = enum_auto()
    LinearCombinationGelu = enum_auto()


EpilogueFunctorTag = {
    EpilogueFunctor.LinearCombination: "cutlass::epilogue::thread::LinearCombination",
    EpilogueFunctor.LinearCombinationRelu: "cutlass::epilogue::thread::LinearCombinationRelu",
    EpilogueFunctor.LinearCombinationBias: "cutlass::epilogue::thread::LinearCombination",
    EpilogueFunctor.LinearCombinationGelu: "cutlass::epilogue::thread::LinearCombinationGELU",
}


class SwizzlingFunctor(enum.Enum):
    Identity1 = enum_auto()
    Identity2 = enum_auto()
    Identity4 = enum_auto()
    Identity8 = enum_auto()
    Batched = enum_auto()


SwizzlingFunctorTag = {
    SwizzlingFunctor.Identity1: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>",
    SwizzlingFunctor.Identity2: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>",
    SwizzlingFunctor.Identity4: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>",
    SwizzlingFunctor.Identity8: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>",
    SwizzlingFunctor.Batched: "cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle",
}


class MathInstruction:
    """Describe characteristics of a math instruction."""

    def __init__(
        self,
        instruction_shape,
        element_a,
        element_b,
        element_accumulator,
        opcode_class,
        math_operation=MathOperation.multiply_add,
    ):
        self.instruction_shape = instruction_shape
        self.element_a = element_a
        self.element_b = element_b
        self.element_accumulator = element_accumulator
        self.opcode_class = opcode_class
        self.math_operation = math_operation


class TileDescription:
    """Describe characteristics of a GEMM tile."""

    def __init__(
        self, threadblock_shape, stages, warp_count, math_instruction, min_compute, max_compute
    ):
        self.threadblock_shape = threadblock_shape
        self.stages = stages
        self.warp_count = warp_count
        self.math_instruction = math_instruction
        self.minimum_compute_capability = min_compute
        self.maximum_compute_capability = max_compute

    def procedural_name(self):
        return "%dx%d_%dx%d" % (
            self.threadblock_shape[0],
            self.threadblock_shape[1],
            self.threadblock_shape[2],
            self.stages,
        )


class TensorDescription:
    def __init__(self, element, layout, alignment=1):
        self.element = element
        self.layout = layout
        self.alignment = alignment
