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
 * \file src/relay/pass/extern_op.cc
 * \brief Wraps a call with subgraph_begin and subgraph_end to indicate that
 * the op of this call node will use external compiler.
 */

#include <tvm/operation.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace extern_op {

class ExternOpWrapper : public ExprMutator {
 public:
  explicit ExternOpWrapper(const std::string& compiler) : compiler_(compiler) {}

  Expr VisitExpr_(const CallNode* cn) {
    auto new_e = ExprMutator::VisitExpr_(cn);

    Call call = Downcast<Call>(new_e);
    static auto fextern = Op::GetAttr<FTVMExternOp>("FTVMExternOp");
    Op op = Downcast<Op>(call->op);
    CHECK(op.operator->());

    if (fextern.count(op)) {
      bool external = fextern[op](call->attrs, call->args, compiler_);
      if (external) {
        tvm::Array<tvm::relay::Expr> subgraph_begins;
        for (const auto& it : call->args) {
          const auto* begin_op =
            runtime::Registry::Get("relay.op.annotation._make.subgraph_begin");
          CHECK(begin_op);
          Expr begin = (*begin_op)(it, compiler_);
          subgraph_begins.push_back(begin);
        }
        Expr begin_call = CallNode::make(call->op, subgraph_begins, call->attrs);
        const auto* end_op =
          runtime::Registry::Get("relay.op.annotation._make.subgraph_end");
        CHECK(end_op);
        Expr end = (*end_op)(begin_call, compiler_);
        return end;
      }
    }
    return new_e;
  }

 private:
  std::string compiler_;
};

/*!
 * \brief Eleminates the back-to-back subgraph_begin and end annotations if they
 * are using the same external compiler. For example, the following graph
 *
 *  subgraph_begin
 *       |
 *      op1
 *       |
 *  subgraph_end
 *       |
 *  subgraph_begin
 *       |
 *      op2
 *       |
 *  subgraph_end
 *
 * will be updated to if op1 and op2 require codegen from the same external
 * compiler.
 *
 *  subgraph_begin
 *       |
 *      op1
 *       |
 *      op2
 *       |
 *  subgraph_end
 */
struct EliminateAnnotation : public ExprMutator {
  Expr VisitExpr_(const CallNode* cn) {
    Expr new_e = ExprMutator::VisitExpr_(cn);
    const auto* op_node = cn->op.as<OpNode>();
    if (op_node && GetRef<Op>(op_node) == Op::Get("annotation.subgraph_begin")) {
      Expr input = cn->args[0];
      if (input.as<CallNode>() == nullptr) return new_e;
      Call input_call = Downcast<Call>(input);
      if (input_call.defined()) {
        const auto* call_op = input_call->op.as<OpNode>();
        if (call_op &&
            GetRef<Op>(call_op) == Op::Get("annotation.subgraph_end")) {
          auto end_attrs = cn->attrs.as<SubgraphAttrs>();
          auto begin_attrs = input_call->attrs.as<SubgraphAttrs>();
          if (end_attrs && begin_attrs &&
              end_attrs->compiler == begin_attrs->compiler) {
            // Eliminate end and begin
            return input_call->args[0];
          }
        }
      }
    }
    return new_e;
  }
};

Expr ExternOp(const Expr& expr, const std::string& compiler) {
  Expr annotated = ExternOpWrapper(compiler).Mutate(expr);
  return EliminateAnnotation().Mutate(annotated);
}

}  // namespace extern_op

namespace transform {

Pass ExternOp(const std::string& compiler) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(relay::extern_op::ExternOp(f, compiler));
      };
  auto func_pass = CreateFunctionPass(pass_func, 1, "ExternOpFunc",
                                      {ir::StringImm::make("InferType")});
  return transform::Sequential({func_pass, InferType()}, "ExternOp");
}

TVM_REGISTER_API("relay._transform.ExternOp")
.set_body_typed(ExternOp);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
