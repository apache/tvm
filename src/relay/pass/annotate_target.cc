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
 * \file src/relay/pass/annotate_target.cc
 * \brief Wraps a call with compiler_begin and compiler_end to indicate that
 * the op of this call node will use external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace annotate_target {

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateTargetWrapper : public ExprMutator {
 public:
  explicit AnnotateTargetWrapper(const std::string& target) : target_(target) {}

  Expr VisitExpr_(const CallNode* cn) {
    // TODO(@zhiics, @comaniac) Handle composite functions.
    auto new_e = ExprMutator::VisitExpr_(cn);

    Call call = Downcast<Call>(new_e);
    static auto fannotate = Op::GetAttr<FTVMAnnotateTarget>("target." + target_);
    Op op = Downcast<Op>(call->op);
    CHECK(op.defined());

    if (fannotate.count(op)) {
      bool external = fannotate[op](call->attrs, call->args);
      if (external) {
        tvm::Array<tvm::relay::Expr> compiler_begins;
        for (const auto& it : call->args) {
          const auto* begin_op =
            runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
          CHECK(begin_op);
          Expr begin = (*begin_op)(it, target_);
          compiler_begins.push_back(begin);
        }
        Expr update_call = CallNode::make(call->op, compiler_begins, call->attrs);
        const auto* end_op =
          runtime::Registry::Get("relay.op.annotation._make.compiler_end");
        CHECK(end_op);
        Expr end = (*end_op)(update_call, target_);
        return end;
      }
    } else {
      LOG(WARNING) << op->name << " in " << target_
                   << " is not registered. It will be executed on CPU.";
    }
    return new_e;
  }

 private:
  std::string target_;
};

Expr AnnotateTarget(const Expr& expr, const std::string& target) {
  return AnnotateTargetWrapper(target).Mutate(expr);
}

}  // namespace annotate_target

namespace transform {

Pass AnnotateTarget(const std::string& target) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::annotate_target::AnnotateTarget(f, target));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetFunc",
                                      {tir::StringImmNode::make("InferType")});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTarget");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTarget")
.set_body_typed(AnnotateTarget);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
