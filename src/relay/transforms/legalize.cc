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
 * \file legalize.cc
 * \brief Converts an expr to another expr. This pass can be used to transform an op based on its
 * shape, dtype or layout to another op or a sequence of ops.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace relay {

namespace legalize {

// Call registered FTVMLegalize of an op
// Returns the legalized expression
class Legalizer : public ExprRewriter {
 public:
  explicit Legalizer(const std::string& legalize_map_attr_name)
      : legalize_map_attr_name_{legalize_map_attr_name} {}

  Expr Rewrite_(const CallNode* call_node, const Expr& post) override {
    // Get the new_call node without any changes to current call node.
    Call new_call = Downcast<Call>(post);

    // Check if the string is registered.
    if (!Op::HasAttrMap(legalize_map_attr_name_)) {
      return post;
    }

    // Collect the registered legalize function.
    auto fop_legalize = Op::GetAttrMap<FTVMLegalize>(legalize_map_attr_name_);
    auto call_op = call_node->op;
    if (call_op.as<OpNode>()) {
      Op op = Downcast<Op>(call_node->op);

      if (fop_legalize.count(op)) {
        // Collect the new_args.
        tvm::Array<Expr> call_args = new_call->args;

        // Collect input and output dtypes to pass on to Legalize API.
        tvm::Array<tvm::relay::Type> types;
        for (auto arg : call_node->args) {
          types.push_back(arg->checked_type());
        }
        types.push_back(call_node->checked_type());

        // Transform the op by calling the registered legalize function.
        Expr legalized_value = fop_legalize[op](call_node->attrs, call_args, types);

        // Return the new expr if the transformation succeeded.
        if (legalized_value.defined()) {
          // Check that the returned Expr from legalize is CallNode.
          const CallNode* legalized_call_node = legalized_value.as<CallNode>();
          ICHECK(legalized_call_node)
              << "Can only replace the original operator with another call node";
          return legalized_value;
        }
      }
    }

    return post;
  }

 private:
  std::string legalize_map_attr_name_;
};

Expr Legalize(const Expr& expr, const std::string& legalize_map_attr_name) {
  auto rewriter = Legalizer(legalize_map_attr_name);
  return PostOrderRewrite(expr, &rewriter);
}

}  // namespace legalize

namespace transform {

Pass Legalize(const String& legalize_map_attr_name) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::legalize::Legalize(f, legalize_map_attr_name));
      };
  return CreateFunctionPass(pass_func, 1, "Legalize", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.Legalize").set_body_typed(Legalize);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
