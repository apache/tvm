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
# pylint: disable=invalid-name,line-too-long
"""Various type definitions to help instantiate ComposableKernel kernels."""
import re
import enum
from enum import auto as enum_auto
from dataclasses import dataclass

from tvm.tir.expr import IntImm, FloatImm


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()


LayoutTag = {
    LayoutType.ColumnMajor: "ck::tensor_layout::gemm::ColumnMajor",
    LayoutType.RowMajor: "ck::tensor_layout::gemm::RowMajor",
}


ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "N",
    LayoutType.RowMajor: "T",
}


class DataType(enum.Enum):
    f16 = enum_auto()
    bf16 = enum_auto()
    f32 = enum_auto()
    f64 = enum_auto()


DataTypeNames = {
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.f64: "f64",
}


DataTypeTag = {
    DataType.f16: "ck::half_t",
    DataType.bf16: "ck::bhalf_t",
    DataType.f32: "float",
    DataType.f64: "double",
}


DataTypeSize = {
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.f64: 64,
}


ShortDataTypeNames = {
    DataType.f16: "h",
    DataType.f32: "s",
    DataType.f64: "d",
}


@dataclass
class TensorDesc:
    element: DataType
    layout: LayoutType


class OperationKind(enum.Enum):
    Gemm = enum_auto()
    Conv1d = enum_auto()
    Conv2d = enum_auto()
    Conv3d = enum_auto()
    Softmax = enum_auto()
    LayerNorm = enum_auto()
    GroupNorm = enum_auto()


OperationKindNames = {
    OperationKind.Gemm: "gemm",
    OperationKind.Conv1d: "conv1d",
    OperationKind.Conv2d: "conv2d",
    OperationKind.Conv3d: "conv3d",
    OperationKind.Softmax: "softmax",
    OperationKind.LayerNorm: "layernorm",
    OperationKind.GroupNorm: "groupnorm",
}


class TensorOperation(enum.Enum):
    PassThrough = enum_auto()
    Add = enum_auto()
    AddAdd = enum_auto()
    AddMul = enum_auto()
    AddMulTanh = enum_auto()
    AlphaBetaAdd = enum_auto()
    AddRelu = enum_auto()
    AddFastGelu = enum_auto()
    AddTanh = enum_auto()
    AddHardswish = enum_auto()
    AddSwish = enum_auto()
    AddSigmoid = enum_auto()
    AddReluAdd = enum_auto()
    AddAddRelu = enum_auto()
    AddSigmoidMul = enum_auto()
    AddSigmoidMulTanh = enum_auto()
    AddHardswishAdd = enum_auto()
    UnaryIdentic = enum_auto()
    UnarySquare = enum_auto()
    UnaryAbs = enum_auto()
    UnarySqrt = enum_auto()
    AddMulAdd = enum_auto()
    AddAddAdd = enum_auto()
    AddAddAddRelu = enum_auto()
    Bilinear = enum_auto()
    CausalMask = enum_auto()


TensorOperationTag = {
    TensorOperation.PassThrough: "ck::tensor_operation::element_wise::PassThrough",
    TensorOperation.Add: "ck::tensor_operation::element_wise::Add",
    TensorOperation.AddAdd: "ck::tensor_operation::element_wise::AddAdd",
    TensorOperation.AddMul: "ck::tensor_operation::element_wise::AddMul",
    TensorOperation.AddMulTanh: "ck::tensor_operation::element_wise::AddMulTanh",
    TensorOperation.AlphaBetaAdd: "ck::tensor_operation::element_wise::AlphaBetaAdd",
    TensorOperation.AddRelu: "ck::tensor_operation::element_wise::AddRelu",
    TensorOperation.AddFastGelu: "ck::tensor_operation::element_wise::AddFastGelu",
    TensorOperation.AddTanh: "ck::tensor_operation::element_wise::AddTanh",
    TensorOperation.AddSigmoid: "ck::tensor_operation::element_wise::AddSigmoid",
    TensorOperation.AddHardswish: "ck::tensor_operation::element_wise::AddHardswish",
    TensorOperation.AddSwish: "ck::tensor_operation::element_wise::AddSwish",
    TensorOperation.AddReluAdd: "ck::tensor_operation::element_wise::AddReluAdd",
    TensorOperation.AddAddRelu: "ck::tensor_operation::element_wise::AddAddRelu",
    TensorOperation.AddHardswishAdd: "ck::tensor_operation::element_wise::AddHardswishAdd",
    TensorOperation.AddMulAdd: "ck::tensor_operation::element_wise::AddMulAdd",
    TensorOperation.AddAddAdd: "ck::tensor_operation::element_wise::AddAddAdd",
    TensorOperation.AddAddAddRelu: "ck::tensor_operation::element_wise::AddAddAddRelu",
    TensorOperation.AddSigmoidMul: "ck::tensor_operation::element_wise::AddSigmoidMul",
    TensorOperation.AddSigmoidMulTanh: "ck::tensor_operation::element_wise::AddSigmoidMulTanh",
    TensorOperation.UnaryIdentic: "ck::tensor_operation::element_wise::UnaryIdentic",
    TensorOperation.UnarySquare: "ck::tensor_operation::element_wise::UnarySquare",
    TensorOperation.UnaryAbs: "ck::tensor_operation::element_wise::UnaryAbs",
    TensorOperation.UnarySqrt: "ck::tensor_operation::element_wise::UnarySqrt",
    TensorOperation.Bilinear: "ck::tensor_operation::element_wise::Bilinear",
    TensorOperation.CausalMask: "True",
}


ShortTensorOperationNames = {
    TensorOperation.PassThrough: "PT",
    TensorOperation.Add: "A",
    TensorOperation.AddAdd: "AA",
    TensorOperation.AddMul: "AM",
    TensorOperation.AddMulTanh: "AMT",
    TensorOperation.AlphaBetaAdd: "ABA",
    TensorOperation.AddRelu: "ARu",
    TensorOperation.AddFastGelu: "AFG",
    TensorOperation.AddTanh: "AT",
    TensorOperation.AddSigmoid: "AS",
    TensorOperation.AddHardswish: "AH",
    TensorOperation.AddSwish: "ASW",
    TensorOperation.AddReluAdd: "ARA",
    TensorOperation.AddAddRelu: "AAR",
    TensorOperation.AddHardswishAdd: "AHA",
    TensorOperation.AddMulAdd: "AMA",
    TensorOperation.AddAddAdd: "AAA",
    TensorOperation.AddAddAddRelu: "AAAR",
    TensorOperation.AddSigmoidMul: "ASM",
    TensorOperation.AddSigmoidMulTanh: "ASMT",
    TensorOperation.UnaryIdentic: "UI",
    TensorOperation.UnarySquare: "USR",
    TensorOperation.UnaryAbs: "UA",
    TensorOperation.UnarySqrt: "USQ",
    TensorOperation.Bilinear: "B",
    TensorOperation.CausalMask: "CM",
}
