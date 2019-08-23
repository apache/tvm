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
 * \file legalize.cc
 * \brief Converts an expr to another expr. This pass can be used to transform an op based on its
 * shape, dtype or layout to another op or a sequence of ops.
 */

#include <tvm/operation.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

namespace legalize {

// Call registered FTVMLegalize of an op
// Returns the legalized expression
Expr Legalizer(const Call& ref_call, const Array<Expr>& new_args, const NodeRef& ctx) {
  static auto fop_legalize = Op::GetAttr<FTVMLegalize>("FTVMLegalize");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  bool modified = false;
  if (fop_legalize.count(op)) {
    // Collect input and output dtypes to pass on to Legalize API.
    tvm::Array<tvm::relay::Type> types;
    for (auto& expr : ref_call->args) {
      types.push_back(expr->checked_type());
    }
    types.push_back(ref_call->checked_type());

    // Transform the op by calling the registered legalize function.
    Expr legalized_value = fop_legalize[op](ref_call->attrs, new_args, types);

    // Check if the transformation succeeded. If not, revert back to the original ref_call->op.
    if (legalized_value.defined()) {
      new_e = legalized_value;
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

Expr Legalize(const Expr& expr) { return ForwardRewrite(expr, Legalizer, nullptr); }

}  // namespace legalize

namespace transform {

Pass Legalize() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(relay::legalize::Legalize(f));
      };
  return CreateFunctionPass(pass_func, 3, "Legalize", {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.Legalize").set_body_typed(Legalize);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
