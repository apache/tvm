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
 * \file src/relay/pass/annotate_compiler.cc
 * \brief Wraps a call with compiler_begin and compiler_end to indicate that
 * the op of this call node will use external compiler.
 */

#include <tvm/operation.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace annotate_compiler {

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateCompilerWrapper : public ExprMutator {
 public:
  explicit AnnotateCompilerWrapper(const std::string& compiler) : compiler_(compiler) {}

  Expr VisitExpr_(const CallNode* cn) {
    auto new_e = ExprMutator::VisitExpr_(cn);

    Call call = Downcast<Call>(new_e);
    static auto fannotate = Op::GetAttr<FTVMAnnotateCompiler>("FTVMAnnotateCompiler");
    Op op = Downcast<Op>(call->op);
    CHECK(op.operator->());

    if (fannotate.count(op)) {
      bool external = fannotate[op](call->attrs, call->args, compiler_);
      if (external) {
        tvm::Array<tvm::relay::Expr> compiler_begins;
        for (const auto& it : call->args) {
          const auto* begin_op =
            runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
          CHECK(begin_op);
          Expr begin = (*begin_op)(it, compiler_);
          compiler_begins.push_back(begin);
        }
        Expr update_call = CallNode::make(call->op, compiler_begins, call->attrs);
        const auto* end_op =
          runtime::Registry::Get("relay.op.annotation._make.compiler_end");
        CHECK(end_op);
        Expr end = (*end_op)(update_call, compiler_);
        return end;
      }
    } else {
      LOG(WARNING) << op.operator->()->name << " in " << compiler_ << " is not registered";
    }
    return new_e;
  }

 private:
  std::string compiler_;
};

Expr AnnotateCompiler(const Expr& expr, const std::string& compiler) {
  return AnnotateCompilerWrapper(compiler).Mutate(expr);
}

}  // namespace annotate_compiler

namespace transform {

Pass AnnotateCompiler(const std::string& compiler) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(relay::annotate_compiler::AnnotateCompiler(f, compiler));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateCompilerFunc",
                                      {ir::StringImm::make("InferType")});
  return transform::Sequential({func_pass, InferType()}, "AnnotateCompiler");
}

TVM_REGISTER_API("relay._transform.AnnotateCompiler")
.set_body_typed(AnnotateCompiler);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
