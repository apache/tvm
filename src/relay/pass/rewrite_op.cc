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
 * Copyright (c) 2019 by Contributors
 * \file rewrite_op.cc
 * \brief Rewrites an expr with another expr. This pass can be used to transform
 * an op based on its shape, dtype or layout to another op or a sequence of ops.
 */

#include <tvm/operation.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

namespace rewrite_op {

// Call registered FTVMRewriteOp of an op
// Returns the altered expression
Expr OpRewriter(const Call& ref_call, const Array<Expr>& new_args, const NodeRef& ctx) {
  static auto fop_rewrite = Op::GetAttr<FTVMRewriteOp>("FTVMRewriteOp");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  bool modified = false;
  if (fop_rewrite.count(op)) {
    tvm::Array<tvm::relay::Type> arg_types;
    for (auto& expr : ref_call->args) {
      arg_types.push_back(expr->checked_type());
    }
    Expr altered_value = fop_rewrite[op](ref_call->attrs, new_args, arg_types);
    if (altered_value.defined()) {
      new_e = altered_value;
      modified = true;
    }
  }
  if (!modified) {
    new_e = CallNode::make(ref_call->op, new_args, ref_call->attrs);
  }

  const CallNode* new_call = new_e.as<CallNode>();
  CHECK(new_call) << "Can only replace the original operator with another call node";
  return GetRef<Call>(new_call);
}

Expr RewriteOp(const Expr& expr) { return ForwardRewrite(expr, OpRewriter, nullptr); }

}  // namespace rewrite_op

namespace transform {

Pass RewriteOp() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(relay::rewrite_op::RewriteOp(f));
      };
  return CreateFunctionPass(pass_func, 3, "RewriteOp", {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.RewriteOp").set_body_typed(RewriteOp);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
