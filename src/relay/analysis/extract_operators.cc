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
 * \file extract_operators.cc
 * \brief Extract unique operators from an IRModule
 */
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class OperatorExtractorWrapper : private MixedModeVisitor {
 public:
  explicit OperatorExtractorWrapper(const IRModule& mod) : mod_(mod) {}

  Map<String, tvm::Integer> Extract() {
    VisitExpr(this->mod_->Lookup("main"));

    return operator_freqs_;
  }

 private:
  using MixedModeVisitor::VisitExpr_;

  const IRModule mod_;
  /*! \brief Map of operator to frequency. */
  Map<String, tvm::Integer> operator_freqs_;

  void VisitExpr_(const CallNode* n) final {
    VisitExpr(n->op);

    auto op = n->op.as<OpNode>();
    if (op) {
      auto it = operator_freqs_.find(op->name);
      ICHECK(it != operator_freqs_.end())
          << "Call's OpNode must be visited and registered before access";
      operator_freqs_.Set(op->name, 1 + operator_freqs_.at(op->name).IntValue());
    }

    MixedModeVisitor::VisitExpr_(n);
  }

  void VisitExpr_(const OpNode* n) final {
    // NOTE: OpNode is visited only once for every operator kind
    // regardless of how many times that op appears in the graph.
    operator_freqs_.Set(n->name, 0U);
  }
};

Map<String, tvm::Integer> ExtractOperatorsPacked(const IRModule& mod) {
  return OperatorExtractorWrapper(mod).Extract();
}

TVM_REGISTER_GLOBAL("relay.analysis.ExtractOperators").set_body_typed(ExtractOperatorsPacked);

}  // namespace relay
}  // namespace tvm
