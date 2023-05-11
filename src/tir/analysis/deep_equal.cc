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
 * \file tir/analysis/deep_equal.cc
 * \brief Deep equality checking.
 */
#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>

namespace tvm {
namespace tir {

class DeepCmpSEqualHandler : public SEqualReducer::Handler {
 public:
  // use direct recursion.
  bool SEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars,
                    const Optional<ObjectPathPair>&) final {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    return vtable_->SEqualReduce(lhs.get(), rhs.get(), SEqualReducer(this, nullptr, false)) &&
           !fail_;
  }

  void DeferFail(const ObjectPathPair&) final { fail_ = true; }
  bool IsFailDeferralEnabled() final { return false; }

  ObjectRef MapLhsToRhs(const ObjectRef& lhs) final { return ObjectRef(nullptr); }
  void MarkGraphNode() final {}

 private:
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
  bool fail_ = false;
};

bool ExprDeepEqual::operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
  // quick path
  if (lhs.same_as(rhs)) return true;
  if (!lhs.defined() && rhs.defined()) return false;
  if (!rhs.defined() && lhs.defined()) return false;
  if (lhs->type_index() != rhs->type_index()) return false;
  if (auto* plhs = lhs.as<IntImmNode>()) {
    auto* prhs = rhs.as<IntImmNode>();
    return plhs->dtype == prhs->dtype && plhs->value == prhs->value;
  }
  if (lhs.as<AnyNode>()) {
    return false;
  }
  return DeepCmpSEqualHandler().SEqualReduce(lhs, rhs, false, NullOpt);
}

TVM_REGISTER_GLOBAL("tir.analysis.expr_deep_equal")
    .set_body_typed([](const PrimExpr& lhs, const PrimExpr& rhs) {
      return ExprDeepEqual()(lhs, rhs);
    });

}  // namespace tir
}  // namespace tvm
