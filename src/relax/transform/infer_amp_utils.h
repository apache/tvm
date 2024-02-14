/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file infer_amp_utils.h
 * \brief Utility functions to be used in to_mixed_precision pass.
 */

#ifndef TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_
#define TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_

#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

using runtime::DLDataType2String;
using runtime::String;
using runtime::String2DLDataType;

enum MixedPrecisionPolicyKind : int { kAlways = 0, kFollow = 1, kNever = 2 };

/*! \brief the operator pattern */
using TMixedPrecisionPolicy = int;

// NType is the message we want to track for vars with nested tensorstructinfo
// which represents the realization decision of the var.
// The string is the name of the dtype decision.
using NType = NestedMsg<String>;

struct NTypeEqual {
  bool operator()(const NType& a, const NType& b) const {
    auto dtype_equal = [](const String& a, const String& b) { return a == b; };
    return Equal(a, b, dtype_equal);
  }
};

// Construct a NType from an StructInfo
NType NTypeFrom(const StructInfo& sinfo, DataType dtype = DataType::Void());

// Construct a NType from an Expr
NType NTypeFrom(const Expr& expr, DataType dtype = DataType::Void());

// Merge two messages, we keep the higher precision type for each leaf tensor
NType NTypeMerge(const NType& a, const NType& b);

// The map that notes the NType message of each var
using VarDTypeMap = std::unordered_map<Var, NType, ObjectPtrHash, ObjectPtrEqual>;

// Call is a call node, out_dtype is the expected output_dtype
using FInferMixedPrecision =
    runtime::TypedPackedFunc<Call(const Call& call_node, const DataType& out_dtype)>;

Array<ObjectRef> InferMixedPrecisionFollow(const Call& call, const DataType& out_dtype);

Array<ObjectRef> InferMixedPrecisionNever(const Call& call, const DataType& out_dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_
