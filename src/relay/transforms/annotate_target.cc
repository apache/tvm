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
 * \file src/relay/transforms/annotate_target.cc
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

// Cache compiler_begin op for equivalence check.
static const Op& compiler_begin_op = Op::Get("annotation.compiler_begin");

// A helper class to insert annotation boundaries for a program region that will
// be handled by a specific compiler.
class AnnotateTargetWrapper : public ExprMutator {
 public:
  explicit AnnotateTargetWrapper(const std::string& target) : target_(target) {}

  Expr Annotate(const Expr& expr) {
    return InsertEnd(Mutate(expr));
  }

  bool IsSupported(const Expr& expr) {
    if (expr->IsInstance<CallNode>()) {
      Call call = Downcast<Call>(expr);
      auto fannotate = Op::GetAttr<FTVMAnnotateTarget>("target." + target_);
      if (call->op->IsInstance<OpNode>()) {
        Op op = Downcast<Op>(call->op);
        CHECK(op.defined());
        if (fannotate.count(op)) {
          return fannotate[op](call->attrs, call->args);
        }
      } else if (call->op->IsInstance<FunctionNode>()) {
        // handle composite functions
        Function func = Downcast<Function>(call->op);
        CHECK(func.defined());
        auto comp_name = func->GetAttr<String>(attr::kComposite);
        if (comp_name.defined()) {
          std::string comp_name_str = comp_name;
          size_t i = comp_name_str.find('.');
          if (i != std::string::npos) {
            std::string target = comp_name_str.substr(0, i);
            if (target == target_) return true;
          }
        }
      }
    }
    if (expr->IsInstance<TupleGetItemNode>()) {
      TupleGetItem get = Downcast<TupleGetItem>(expr);
      if (get->tuple->IsInstance<CallNode>() &&
          get->tuple.as<CallNode>()->op == compiler_begin_op) {
        return true;
      }
    }
    return false;
  }

  Expr InsertEnd(const Expr& arg) {
    if (IsSupported(arg)) {
      const auto *end_op =
        runtime::Registry::Get("relay.op.annotation._make.compiler_end");
      CHECK(end_op);
      Expr end = (*end_op)(arg, target_);
      return end;
    }
    return arg;
  }

  Expr VisitExpr_(const CallNode* cn) {
    auto new_e = ExprMutator::VisitExpr_(cn);

    Call call = Downcast<Call>(new_e);

    // add end annotations if the args are supported
    Array<Expr> compiler_ends;
    for (const auto& it : call->args) {
      compiler_ends.push_back(InsertEnd(it));
    }
    call = Call(call->op, compiler_ends, call->attrs);

    // add begin annotations if the call node is supported
    if (IsSupported(call)) {
      tvm::Array<tvm::relay::Expr> compiler_begins;
      const auto* begin_op =
        runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
      for (const auto& it : call->args) {
        CHECK(begin_op);
        Expr begin = (*begin_op)(it, target_);
        compiler_begins.push_back(begin);
      }
      call = Call(call->op, compiler_begins, call->attrs);
    }

    return std::move(call);
  }

  Expr VisitExpr_(const TupleNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto tup = Downcast<Tuple>(new_e);
    Array<Expr> new_fields;
    for (auto field : tup->fields) {
      new_fields.push_back(InsertEnd(field));
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const TupleGetItemNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto get = Downcast<TupleGetItem>(new_e);
    if (IsSupported(get->tuple)) {
      const auto* begin_op =
        runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
      CHECK(begin_op);
      return TupleGetItem((*begin_op)(InsertEnd(get->tuple), target_), get->index);
    } else {
      return TupleGetItem(InsertEnd(get->tuple), get->index);
    }
  }

  Expr VisitExpr_(const FunctionNode* fn) {
    Function func;
    Expr new_body;
    // don't step into composite functions
    if (fn->GetAttr<String>(attr::kComposite).defined()) {
      func = GetRef<Function>(fn);
      new_body = func->body;
    } else {
      auto new_e = ExprMutator::VisitExpr_(fn);
      func = Downcast<Function>(new_e);
      new_body = InsertEnd(func->body);
    }

    return Function(
      func->params,
      new_body,
      func->ret_type,
      func->type_params,
      func->attrs);
  }

  Expr VisitExpr_(const LetNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto let = Downcast<Let>(new_e);
    return Let(
      let->var,
      InsertEnd(let->value),
      InsertEnd(let->body));
  }

  Expr VisitExpr_(const IfNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto iff = Downcast<If>(new_e);
    return If(
      InsertEnd(iff->cond),
      InsertEnd(iff->true_branch),
      InsertEnd(iff->false_branch));
  }

  Expr VisitExpr_(const RefCreateNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto create = Downcast<RefCreate>(new_e);
    return RefCreate(InsertEnd(create->value));
  }

  Expr VisitExpr_(const RefReadNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto read = Downcast<RefRead>(new_e);
    return RefRead(InsertEnd(read->ref));
  }

  Expr VisitExpr_(const RefWriteNode* op) {
    auto new_e = ExprMutator::VisitExpr_(op);

    auto write = Downcast<RefWrite>(new_e);
    return RefWrite(
      InsertEnd(write->ref),
      InsertEnd(write->value));
  }

 private:
  std::string target_;
};

Expr AnnotateTarget(const Expr& expr, const std::string& target) {
  return AnnotateTargetWrapper(target).Annotate(expr);
}

}  // namespace annotate_target

namespace transform {

Pass AnnotateTarget(const std::string& target) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::annotate_target::AnnotateTarget(f, target));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetFunc",
                                      {"InferType"});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTarget");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTarget")
.set_body_typed(AnnotateTarget);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
