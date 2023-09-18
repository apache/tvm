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
 * \file src/relax/transform/canonicalize_qnn_ops.cc
 * \brief Pass that lower QNN operations into equivalent chain of ops.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class QnnCanonicalizer : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call_node) final {
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));
    static const auto& lower_map = Op::GetAttrMap<FQnnLower>("FQnnLower");
    const Op& op = Downcast<Op>(call->op);
    if (lower_map.count(op)) {
      return lower_map[op](call);
    }
    return call;
  }
};

Expr CanonicalizeQnnOps(const Expr& e) { return QnnCanonicalizer().VisitExpr(e); }

namespace transform {

Pass QnnCanonicalize() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeQnnOps(f));
      };
  return CreateFunctionPass(pass_func, 1, "QnnCanonicalize", {});
}

TVM_REGISTER_GLOBAL("relax.transform.QnnCanonicalize").set_body_typed(QnnCanonicalize);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
