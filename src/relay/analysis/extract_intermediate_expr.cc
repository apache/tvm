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
 * \file extract_intermediate_expr.cc
 * \brief Used for extracting Relay Expr
    by the expression ID of the main function
    that we can see in `print(mod["main"])`.
 */
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class ExtractIntermediateExprWrapper : private MixedModeVisitor {
 public:
  explicit ExtractIntermediateExprWrapper(const IRModule& mod, const int expr_id)
      : mod_(mod), target_expr_id_(expr_id), counter_(0) {}

  IRModule Extract() {
    VisitExpr(this->mod_->Lookup("main"));

    // ensure the target expr_id we want to extract is valid.
    ICHECK(target_expr_id_ >= 0 && target_expr_id_ < counter_);

    return IRModule::FromExpr(target_op_, {});
  }

 private:
  using MixedModeVisitor::VisitExpr_;

  const IRModule mod_;
  /*! \brief the expr id that we want to extract. */
  const int target_expr_id_;
  int counter_;
  Expr target_op_;

  void VisitExpr_(const CallNode* n) final {
    CheckCounterAndIncrease(GetRef<Expr>(n));
    MixedModeVisitor::VisitExpr_(n);
  }

  void VisitExpr_(const TupleNode* n) final {
    CheckCounterAndIncrease(GetRef<Expr>(n));
    MixedModeVisitor::VisitExpr_(n);
  }

  void VisitExpr_(const TupleGetItemNode* n) final {
    CheckCounterAndIncrease(GetRef<Expr>(n));
    MixedModeVisitor::VisitExpr_(n);
  }

  void CheckCounterAndIncrease(const Expr& expr) {
    if (target_expr_id_ == counter_) {
      target_op_ = expr;
    }
    ++counter_;
  }
};

IRModule ExtractIntermediateExprPacked(const IRModule& mod, const int expr_id) {
  return ExtractIntermediateExprWrapper(mod, expr_id).Extract();
}

TVM_REGISTER_GLOBAL("relay.analysis.ExtractIntermediateExpr")
    .set_body_typed(ExtractIntermediateExprPacked);

}  // namespace relay
}  // namespace tvm
