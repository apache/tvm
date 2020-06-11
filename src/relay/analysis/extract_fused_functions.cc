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
 * \file extract_fused_functions.cc
 * \brief Apply fusion and extract fused primitive functions from an IRModule
 */
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class FusedFunctionExtractorWrapper : private ExprVisitor {
 public:
  explicit FusedFunctionExtractorWrapper(const IRModule& mod) : mod_(mod) {}

  IRModule Extract() {
    VisitExpr(this->mod_->Lookup("main"));

    auto functions = Map<GlobalVar, BaseFunc>();
    for (auto pair : this->functions) {
      functions.Set(GlobalVar(pair.first), pair.second);
    }

    this->mod_->functions = functions;
    return this->mod_;
  }

 private:
  const IRModule mod_;
  // This is not simply Map<GlobalVar, Function> because GlobalVar doesn't
  // have the desired equals property
  Map<String, Function> functions;

  void VisitExpr_(const FunctionNode* n) final {
    if (n->HasNonzeroAttr(attr::kPrimitive)) {
      // Add function to functions, keyed by function hash string
      Function func = Function(n->params, n->body, n->ret_type, n->type_params, n->attrs);
      size_t hash_ = tvm::StructuralHash()(func);
      this->functions.Set(std::to_string(hash_), func);
    }

    ExprVisitor::VisitExpr_(n);
  }
};

namespace transform {

Pass ExtractFusedFunctions() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return FusedFunctionExtractorWrapper(m).Extract(); };
  auto fused_function_extractor_pass = CreateModulePass(pass_func, 1, "ExtractFusedFunctions", {});

  return Sequential({SimplifyInference(), FuseOps(3), fused_function_extractor_pass},
                    "ExtractFusedFunctions");
}

TVM_REGISTER_GLOBAL("relay.analysis.ExtractFusedFunctions").set_body_typed(ExtractFusedFunctions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
