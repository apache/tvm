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
 * \brief Extract operator frequencies from an IRModule
 */
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

using PackedOperatorFrequencyMap = Map<String, Integer>;

class OperatorExtractorWrapper : private ExprVisitor {
 public:
  explicit OperatorExtractorWrapper(const IRModule& mod) : mod_(mod) {}

  Map<String, Integer> Extract() {
    VisitExpr(this->mod_->Lookup("main"));

    return this->operator_freqs;
  }

 private:
  const IRModule mod_;
  // Map of operator name to the number of times they appear in the module.
  Map<String, tvm::Integer> operator_freqs;

  void VisitExpr_(const OpNode* n) final {

    auto it = this->operator_freqs.find(n->name);
	if (it == this->operator_freqs.end()) {
        this->operator_freqs.Set(n->name, 0U);
	}
    std::cout << n->name << std::endl;

	this->operator_freqs.Set(n->name, 1 + this->operator_freqs.at(n->name));

    ExprVisitor::VisitExpr_(n);
  }
};

PackedOperatorFrequencyMap ExtractOperatorsPacked(const IRModule& mod) {
    return OperatorExtractorWrapper(mod).Extract();
}

TVM_REGISTER_GLOBAL("relay.analysis.ExtractOperators").set_body_typed(ExtractOperatorsPacked);

}  // namespace relay
}  // namespace tvm
