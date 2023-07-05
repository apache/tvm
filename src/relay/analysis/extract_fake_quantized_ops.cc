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
 * \file extract_fake_quantized_ops.cc
 * \brief Extract fake quantized operators from an IRModule
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../transforms/fake_quantization_to_integer.h"

namespace tvm {
namespace relay {

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;

class ExtractFakeQuantizedOpsWrapper : private MixedModeVisitor {
 public:
  Map<String, tvm::Integer> Extract(const IRModule& m) {
    IRModule mod(m);
    mod = transform::InferType()(mod);
    VisitExpr(mod->Lookup("main"));

    return fake_quantized_op_freqs_;
  }

 private:
  using MixedModeVisitor::VisitExpr_;

  void VisitExpr_(const CallNode* call_node) override {
    if (call_node->op == quantize_op_) {
      SubgraphExtractor extractor;
      ExprSet subgraph = extractor.GetSubgraph(GetRef<Expr>(call_node));

      for (auto expr : subgraph) {
        const Op op = Downcast<Op>(expr.as<CallNode>()->op);
        if (op != dequantize_op_) {
          if (fake_quantized_op_freqs_.find(op->name) != fake_quantized_op_freqs_.end()) {
            fake_quantized_op_freqs_.Set(op->name,
                                         fake_quantized_op_freqs_.at(op->name).IntValue() + 1);
          } else {
            fake_quantized_op_freqs_.Set(op->name, 1);
          }
        }
      }
    }
  }

  Map<String, tvm::Integer> fake_quantized_op_freqs_;
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
};

Map<String, tvm::Integer> ExtractFakeQuantizedOpsPacked(const IRModule& mod) {
  return ExtractFakeQuantizedOpsWrapper().Extract(mod);
}

TVM_REGISTER_GLOBAL("relay.analysis.ExtractFakeQuantizedOps")
    .set_body_typed(ExtractFakeQuantizedOpsPacked);

}  // namespace relay
}  // namespace tvm
