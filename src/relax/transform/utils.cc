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

#include "utils.h"

namespace tvm {
namespace relax {

bool IsScalarTensor(const StructInfo& sinfo) {
  if (!sinfo->IsInstance<TensorStructInfoNode>()) {
    return false;
  }
  TensorStructInfo tensor_sinfo = Downcast<TensorStructInfo>(sinfo);
  if (!tensor_sinfo->shape.defined() || !tensor_sinfo->shape->IsInstance<ShapeExprNode>()) {
    return false;
  }
  return tensor_sinfo->shape.as<ShapeExprNode>()->values.size() == 0;
}

bool IsScalarTensor(const Expr& expr) { return IsScalarTensor(GetStructInfo(expr)); }

bool IsNestedTensor(const StructInfo& sinfo) {
  return IsNestedTensorConditioned(sinfo, [](const TensorStructInfo& sinfo) { return true; });
}

bool IsNestedTensor(const Expr& expr) { return IsNestedTensor(GetStructInfo(expr)); }

}  // namespace relax
}  // namespace tvm
