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

#include "infer_amp_utils.h"

namespace tvm {
namespace relax {

NType NTypeFrom(const StructInfo& sinfo, DataType dtype) {
  auto fmapleaf = [&](const StructInfo& sinfo) -> NType {
    const auto* tensor = sinfo.as<TensorStructInfoNode>();
    ICHECK(tensor) << "Expected TensorStructInfo, but got " << sinfo;
    if (dtype == DataType::Void())
      return NType(DLDataType2String(tensor->dtype));
    else
      return NType(DLDataType2String(dtype));
  };
  return MapToNestedMsg<String>(sinfo, fmapleaf);
}

NType NTypeFrom(const Expr& expr, DataType dtype) { return NTypeFrom(GetStructInfo(expr), dtype); }

NType NTypeMerge(const NType& a, const NType& b) {
  auto fcombine = [&](const String& a_str, const String& b_str) -> String {
    if (a_str == "") {
      return b_str;
    } else if (b_str == "") {
      return a_str;
    }

    DataType a = DataType(String2DLDataType(a_str));
    DataType b = DataType(String2DLDataType(b_str));
    ICHECK_EQ(a.code(), b.code());
    ICHECK_EQ(a.lanes(), b.lanes());
    return a.bits() > b.bits() ? a_str : b_str;
  };
  return CombineNestedMsg<String>(a, b, fcombine);
}

Array<ObjectRef> InferMixedPrecisionFollow(const Call& call, const DataType& out_dtype) {
  return {Integer(MixedPrecisionPolicyKind::kFollow), call};
}

Array<ObjectRef> InferMixedPrecisionNever(const Call& call, const DataType& out_dtype) {
  return {Integer(MixedPrecisionPolicyKind::kNever), call};
}

}  // namespace relax
}  // namespace tvm
