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
#
# \file library.py
#
# \brief Generates the CUTLASS Library's instances
#

import re

###################################################################################################

import enum

# The following block implements enum.auto() for Python 3.5 variants that don't include it such
# as the default 3.5.2 on Ubuntu 16.04.
#
# https://codereview.stackexchange.com/questions/177309/reimplementing-pythons-enum-auto-for-compatibility

try:
    from enum import auto as enum_auto
except ImportError:
    __cutlass_library_auto_enum = 0

    def enum_auto() -> int:
        global __cutlass_library_auto_enum
        i = __cutlass_library_auto_enum
        __cutlass_library_auto_enum += 1
        return i


###################################################################################################

#
class GeneratorTarget(enum.Enum):
    Library = enum_auto()


#
GeneratorTargetNames = {GeneratorTarget.Library: "library"}
#

###################################################################################################

#
class DataType(enum.Enum):
    f16 = enum_auto()
    f32 = enum_auto()
    invalid = enum_auto()


#
ShortDataTypeNames = {
    DataType.f16: "h",
    DataType.f32: "s",
}

#
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


###################################################################################################

#
class MathOperation(enum.Enum):
    multiply_add = enum_auto()


MathOperationTag = {
    MathOperation.multiply_add: "cutlass::arch::OpMultiplyAdd",
}

###################################################################################################

#
class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()
    ColumnMajorInterleaved2 = enum_auto()
    RowMajorInterleaved2 = enum_auto()
    ColumnMajorInterleaved32 = enum_auto()
    RowMajorInterleaved32 = enum_auto()
    ColumnMajorInterleaved64 = enum_auto()
    RowMajorInterleaved64 = enum_auto()


#
LayoutTag = {
    LayoutType.ColumnMajor: "cutlass::layout::ColumnMajor",
    LayoutType.RowMajor: "cutlass::layout::RowMajor",
    LayoutType.ColumnMajorInterleaved2: "cutlass::layout::ColumnMajorInterleaved<2>",
    LayoutType.RowMajorInterleaved2: "cutlass::layout::RowMajorInterleaved<2>",
    LayoutType.ColumnMajorInterleaved32: "cutlass::layout::ColumnMajorInterleaved<32>",
    LayoutType.RowMajorInterleaved32: "cutlass::layout::RowMajorInterleaved<32>",
    LayoutType.ColumnMajorInterleaved64: "cutlass::layout::ColumnMajorInterleaved<64>",
    LayoutType.RowMajorInterleaved64: "cutlass::layout::RowMajorInterleaved<64>",
}

#
TransposedLayout = {
    LayoutType.ColumnMajor: LayoutType.RowMajor,
    LayoutType.RowMajor: LayoutType.ColumnMajor,
    LayoutType.ColumnMajorInterleaved2: LayoutType.RowMajorInterleaved2,
    LayoutType.RowMajorInterleaved2: LayoutType.ColumnMajorInterleaved2,
    LayoutType.ColumnMajorInterleaved32: LayoutType.RowMajorInterleaved32,
    LayoutType.RowMajorInterleaved32: LayoutType.ColumnMajorInterleaved32,
    LayoutType.ColumnMajorInterleaved64: LayoutType.RowMajorInterleaved64,
    LayoutType.RowMajorInterleaved64: LayoutType.ColumnMajorInterleaved64,
}

#
ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "n",
    LayoutType.ColumnMajorInterleaved2: "n2",
    LayoutType.ColumnMajorInterleaved32: "n32",
    LayoutType.ColumnMajorInterleaved64: "n64",
    LayoutType.RowMajor: "t",
    LayoutType.RowMajorInterleaved2: "t2",
    LayoutType.RowMajorInterleaved32: "t32",
    LayoutType.RowMajorInterleaved64: "t64",
}

###################################################################################################
#
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

###################################################################################################

#
class OperationKind(enum.Enum):
    Gemm = enum_auto()


#
OperationKindNames = {
    OperationKind.Gemm: "gemm",
}

#
class Target(enum.Enum):
    library = enum_auto()


ArchitectureNames = {
    50: "maxwell",
    60: "pascal",
    61: "pascal",
    70: "volta",
    75: "turing",
    80: "ampere",
}

###################################################################################################

#
def SubstituteTemplate(template, values):
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


###################################################################################################
#
class GemmKind(enum.Enum):
    Gemm = enum_auto()
    Universal = enum_auto()


#
GemmKindNames = {
    GemmKind.Gemm: "gemm",
    GemmKind.Universal: "gemm",
}

#
class EpilogueFunctor(enum.Enum):
    LinearCombination = enum_auto()
    LinearCombinationClamp = enum_auto()
    LinearCombinationRelu = enum_auto()
    LinearCombinationBiasRelu = enum_auto()
    LinearCombinationGelu = enum_auto()
    LinearCombinationBias = enum_auto()


#
EpilogueFunctorTag = {
    EpilogueFunctor.LinearCombination: "cutlass::epilogue::thread::LinearCombination",
    EpilogueFunctor.LinearCombinationClamp: "cutlass::epilogue::thread::LinearCombinationClamp",
    EpilogueFunctor.LinearCombinationRelu: "cutlass::epilogue::thread::LinearCombinationRelu",
    EpilogueFunctor.LinearCombinationGelu: "cutlass::epilogue::thread::LinearCombinationGELU",
    EpilogueFunctor.LinearCombinationBias: "cutlass::epilogue::thread::LinearCombinationBias",
}

#
class SwizzlingFunctor(enum.Enum):
    Identity1 = enum_auto()
    Identity2 = enum_auto()
    Identity4 = enum_auto()
    Identity8 = enum_auto()


#
SwizzlingFunctorTag = {
    SwizzlingFunctor.Identity1: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>",
    SwizzlingFunctor.Identity2: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>",
    SwizzlingFunctor.Identity4: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>",
    SwizzlingFunctor.Identity8: "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>",
}

###################################################################################################

#
class MathInstruction:
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


#
class TileDescription:
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


#
class TensorDescription:
    def __init__(self, element, layout, alignment=1):
        self.element = element
        self.layout = layout
        self.alignment = alignment


###################################################################################################
