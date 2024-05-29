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
 * \file is_pure_function.cc
 * \brief PrimFunc purity analysis
 */
#include <tvm/ir/op.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tir {

namespace {
class PurityChecker : TIRVisitorWithPath {
 public:
  static bool Check(const PrimFunc& func, bool assert_on_error) {
    PurityChecker visitor(assert_on_error);
    visitor(func);
    return visitor.is_pure_;
  }

 private:
  explicit PurityChecker(bool assert_on_error) : assert_on_error_(assert_on_error) {}

  void VisitStmt_(const AllocateNode* op, ObjectPath path) override {
    internal_allocations_.insert(op->buffer_var);
    TIRVisitorWithPath::VisitStmt_(op, path);
  }

  void VisitStmt_(const BufferStoreNode* op, ObjectPath path) override {
    TIRVisitorWithPath::VisitStmt_(op, path);

    if (!internal_allocations_.count(op->buffer->data)) {
      is_pure_ = false;
      LOG_IF(FATAL, assert_on_error_) << "AssertionError: "
                                      << "Pure functions must not write to buffers, "
                                      << ", but function contains store to " << op->buffer
                                      << op->indices << " of value " << op->value;
    }
  }

  void VisitExpr_(const CallNode* call, ObjectPath path) override {
    TIRVisitorWithPath::VisitExpr_(call, path);

    static auto op_call_effect = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    CallEffectKind effect = [&]() {
      if (auto opt = call->op.as<Op>()) {
        return static_cast<CallEffectKind>(op_call_effect[opt.value()]->value);
      } else {
        return CallEffectKind::kOpaque;
      }
    }();

    if (effect == CallEffectKind::kUpdateState || effect == CallEffectKind::kOpaque) {
      is_pure_ = false;
      LOG_IF(FATAL, assert_on_error_)
          << "AssertionError: "
          << "Pure functions must not contain calls to impure operators, "
          << "but " << GetRef<PrimExpr>(call) << " calls operator " << call->op
          << ", which has side effect " << effect;
    }
  }

  bool assert_on_error_{false};
  bool is_pure_{true};
  std::unordered_set<Var> internal_allocations_;
};
}  // namespace

bool IsPureFunction(const PrimFunc& func, bool assert_on_error) {
  return PurityChecker::Check(func, assert_on_error);
}

TVM_REGISTER_GLOBAL("tir.analysis.is_pure_function").set_body_typed(IsPureFunction);

}  // namespace tir
}  // namespace tvm
