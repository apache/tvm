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
 * \brief Wraps an expr with compiler_begin and compiler_end to indicate that
 * this expr should be handled by the external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "pass_utils.h"

namespace tvm {
namespace relay {
namespace annotate_target {

static const PackedFunc* make_begin_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
static const PackedFunc* make_end_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_end");
static const char default_target[] = "default";
// A helper class to insert annotation boundaries for all the ops of a program
// region that will be handled by a specific compiler.
class AnnotateTargetRewriter : public ExprRewriter {
 public:
  explicit AnnotateTargetRewriter(Array<runtime::String> targets) : targets_(std::move(targets)) {}

 protected:
  /*! \brief The target backends for annotation. */
  Array<runtime::String> targets_;
  /*! \brief Maintain the decision of the target for each op expr. */
  std::unordered_map<Expr, std::string, ObjectPtrHash, ObjectPtrEqual> op_expr_to_target_;

  /*!
   * \brief This function annotates a compiler end and a compiler begin to all arguments.
   *
   *  The compiler end is based on the arg target while the compiler begin is based on the given
   *  target. If target is not given and all arguments are going to the same target, then we will
   *  use that target; otherwise we use default for this op. Note that all arg exprs must be
   *  available in op_expr_to_target before calling this function.
   *
   * \param args An array of arguments of the given node.
   * \param target The target of the current node.
   * \return A pair of target and annotated argument expressions.
   */
  std::pair<std::string, Array<Expr>> AnnotateArgs(const Array<Expr>& args,
                                                   const std::string& target = "") {
    std::string ref_target = "";
    Array<Expr> compiler_begins;
    Array<Expr> compiler_ends;
    for (auto arg : args) {
      std::string arg_target = default_target;
      const CallNode* call = arg.as<CallNode>();

      if (call && call->op == CompilerBeginOp()) {
        // Argument is already compiler begin node meaning that this is not the first time
        // running this pass, so we simply remove it and will add a new one later.
        ICHECK_EQ(call->args.size(), 1U);
        // Do not alter existing annotation if not default
        if (default_target != call->attrs.as<CompilerAttrs>()->compiler) {
          compiler_begins.push_back(arg);
        } else {
          // Remove default
          compiler_ends.push_back(call->args[0]);
        }
        const CallNode* end = call->args[0].as<CallNode>();
        if (end && end->op == CompilerEndOp()) {
          arg_target = end->attrs.as<CompilerAttrs>()->compiler;
        }
      } else if (op_expr_to_target_.find(arg) != op_expr_to_target_.end()) {
        arg_target = op_expr_to_target_[arg];
        // If an argument is a call node and has no argument, then it should be tensor ops such as
        // zeros, so we treat it as input vars.
        if (call && call->args.size() == 0) {
          compiler_ends.push_back(arg);
        } else {
          compiler_ends.push_back(InsertAnnotation(arg, arg_target, make_end_op));
        }
      } else {
        // Input vars.
        compiler_ends.push_back(arg);
      }

      // Maintain reference target in case the target of the current node is unassigned.
      if (ref_target == "") {
        ref_target = arg_target;
      } else if (ref_target != arg_target) {
        ref_target = default_target;
      }
    }

    // Determine compiler begin target.
    std::string op_target = (target == "") ? ref_target : target;

    if (ref_target != "") {
      for (const auto& end : compiler_ends) {
        compiler_begins.push_back(InsertAnnotation(end, op_target, make_begin_op));
      }
    } else {
      return {op_target, args};
    }
    return {op_target, compiler_begins};
  }

  Expr InsertAnnotation(const Expr& expr, const std::string& target, const PackedFunc* ann_op) {
    Expr new_op = (*ann_op)(expr, target);
    new_op->checked_type_ = expr->checked_type_;
    return new_op;
  }

  Expr InsertCompilerEndAndPropogateTarget(const Expr& expr) {
    /*!
     * \brief This function inserts compiler end to expr and maps the corresponding target to the
     * new expression.
     *
     *  This function checks for expr existence within the map and inserts the annotation.
     *  If the expression has a free variable (e.g: relay.zeros, relay.ones) we do not insert
     *  compiler end, since there are no compiler begins for it.
     *  Further, it propagates the target to the new expression and returns it
     *
     * \param expr A relay expression
     * \return An annotated and target-propagated relay expression.
     */
    Expr new_expr = expr;
    const CallNode* call = expr.as<CallNode>();
    const TupleNode* tup = expr.as<TupleNode>();
    if (op_expr_to_target_.find(expr) != op_expr_to_target_.end()) {
      // Check whether expr has args, if not - do not insert compiler_end.
      if (expr->IsInstance<RefWriteNode>() || expr->IsInstance<RefCreateNode>() ||
          expr->IsInstance<RefReadNode>() || expr->IsInstance<TupleGetItemNode>() ||
          (call && !call->args.empty()) || (tup && !tup->fields.empty())) {
        std::string target = op_expr_to_target_[new_expr];
        new_expr = InsertAnnotation(new_expr, target, make_end_op);
        op_expr_to_target_[new_expr] = target;
      }
    } else if (call && call->op == CompilerEndOp()) {
      if (default_target == call->attrs.as<CompilerAttrs>()->compiler) {
        ICHECK_EQ(call->args.size(), 1U);
        new_expr = call->args[0];
        std::string target = op_expr_to_target_[new_expr];
        new_expr = InsertAnnotation(new_expr, target, make_end_op);
        op_expr_to_target_[new_expr] = target;
      }
    }

    return std::move(new_expr);
  }

 public:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    // Supported targets for this node. The order implies the priority.
    std::vector<std::string> supported_targets;

    auto op_node = pre->op.as<OpNode>();

    // This graph has annotations, meaning that this is not the first time running this pass.
    if (op_node && pre->op == CompilerBeginOp()) {
      // Bypass compiler begin due to lack of target information. It will be processed
      // when the following op handling arguments.
      ICHECK_EQ(pre->args.size(), 1U);
      // Preserve annotations
      return post;
    } else if (op_node && pre->op == CompilerEndOp()) {
      // Override compiler end with the new target.
      ICHECK_EQ(pre->args.size(), 1U);
      auto input_expr = post.as<CallNode>()->args[0];
      // Already annotated. Recover target
      if (op_expr_to_target_.find(input_expr) == op_expr_to_target_.end()) {
        op_expr_to_target_[input_expr] = post.as<CallNode>()->attrs.as<CompilerAttrs>()->compiler;
      }
      ICHECK(op_expr_to_target_.find(input_expr) != op_expr_to_target_.end());
      // Preserve annotated nodes
      return post;
    }
    // Check prior to peeking first argument
    if (pre->args.size()) {
      // Peek the first argument. If it is compiler begin then this node had annotated by
      // another target before, so we also consider that target as a supported target.
      const CallNode* first_arg_call = pre->args[0].as<CallNode>();
      if (first_arg_call && first_arg_call->op == CompilerBeginOp()) {
        std::string arg_target = first_arg_call->attrs.as<CompilerAttrs>()->compiler;
        if (arg_target != default_target) {
          // annotated already
          return post;
        }
      }
    }

    // Check which targets this op can be offloaded.
    if (op_node) {
      // TVM operators: Check target specific op checking function and add to supported_targets
      // if it is supported.
      Op op = Downcast<Op>(pre->op);
      ICHECK(op.defined());
      for (const auto& target : this->targets_) {
        if (!Op::HasAttrMap("target." + std::string(target))) {
          continue;
        }
        auto fannotate = Op::GetAttrMap<FTVMAnnotateTarget>("target." + std::string(target));
        const Expr& ex = GetRef<Expr>(pre);
        if (fannotate.count(op) && fannotate[op](ex)) {
          supported_targets.push_back(target);
        }
      }
    } else if (pre->op->IsInstance<FunctionNode>()) {
      // Composite function: Add the target of a composite function to supported_targets
      // if it is in the target list.
      Function func = Downcast<Function>(pre->op);
      ICHECK(func.defined());
      if (auto comp_name = func->GetAttr<String>(attr::kComposite)) {
        std::string comp_name_str = comp_name.value();
        size_t i = comp_name_str.find('.');
        if (i != std::string::npos) {
          std::string comp_target = comp_name_str.substr(0, i);
          for (const auto& target : this->targets_) {
            if (std::string(target) == comp_target) {
              supported_targets.push_back(comp_target);
              break;
            }
          }
        }
      }
    }
    supported_targets.push_back(default_target);  // Make default as the last option.
    // Visit and mutate arguments after the target of this op has been determined.
    Call post_call = Downcast<Call>(post);
    if (pre->op->IsInstance<VarNode>()) {
      auto new_call = RewriteVarCall(post_call);
      if (nullptr != new_call) return GetRef<Expr>(new_call->get());
    }
    // TODO(@comaniac, @zhiics): Now we simply assign this node to the target with
    // the highest priority, but we should preserve all supported targets so that
    // we can make a better decision.
    std::string target = supported_targets[0];

    // Add annotations to each arg.
    auto target_n_args = AnnotateArgs(post_call->args, target);
    Array<Expr> compiler_begins = std::get<1>(target_n_args);
    Call new_call = Call(post_call->op, compiler_begins, post_call->attrs);
    new_call->checked_type_ = pre->checked_type_;
    new_call->span = pre->span;

    // Update the target map.
    op_expr_to_target_[new_call] = target;
    return std::move(new_call);
  }

  virtual std::unique_ptr<Call> RewriteVarCall(const Call& post_call) { return nullptr; }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) override {
    auto tuple = Downcast<Tuple>(post);

    auto target_n_args = AnnotateArgs(tuple->fields);
    auto new_expr = WithFields(tuple, std::get<1>(target_n_args));
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) override {
    auto expr = Downcast<TupleGetItem>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->tuple}));
    auto new_expr = TupleGetItem(std::get<1>(target_n_args)[0], expr->index);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const FunctionNode* fn, const Expr& post) override {
    Function func;
    Expr new_body;
    // don't step into composite functions
    if (fn->GetAttr<String>(attr::kComposite).defined()) {
      func = GetRef<Function>(fn);
      new_body = func->body;
    } else {
      func = Downcast<Function>(post);
      new_body = InsertCompilerEndAndPropogateTarget(func->body);
    }
    return WithFields(func, func->params, new_body);
  }

  Expr Rewrite_(const LetNode* op, const Expr& post) override {
    auto let = Downcast<Let>(post);

    Expr new_expr;
    std::pair<std::string, Array<Expr>> target_n_args;
    Expr new_body = InsertCompilerEndAndPropogateTarget(let->body);
    // Do not annotate function literal with let binding.
    if (let->value->IsInstance<FunctionNode>()) {
      new_expr = Let(let->var, let->value, new_body);
    } else {
      target_n_args = AnnotateArgs({let->value});
      new_expr = Let(let->var, std::get<1>(target_n_args)[0], new_body);
    }

    return std::move(new_expr);
  }

  Expr Rewrite_(const IfNode* op, const Expr& post) override {
    auto expr = Downcast<If>(post);
    Expr new_cond = InsertCompilerEndAndPropogateTarget(expr->cond);
    Expr new_true_branch = InsertCompilerEndAndPropogateTarget(expr->true_branch);
    Expr new_false_branch = InsertCompilerEndAndPropogateTarget(expr->false_branch);

    auto new_expr = If(new_cond, new_true_branch, new_false_branch);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefCreateNode* op, const Expr& post) override {
    auto expr = Downcast<RefCreate>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->value}));
    auto new_expr = RefCreate(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefReadNode* op, const Expr& post) override {
    auto expr = Downcast<RefRead>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref}));
    auto new_expr = RefRead(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefWriteNode* op, const Expr& post) override {
    auto expr = Downcast<RefWrite>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref, expr->value}));
    auto new_expr = RefWrite(std::get<1>(target_n_args)[0], std::get<1>(target_n_args)[1]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }
};

// A helper class to insert annotation boundaries for call ops and function nodes
// in a program region that will be handled by a specific compiler.
class CallOpsTargetRewriter : public AnnotateTargetRewriter {
 public:
  explicit CallOpsTargetRewriter(Array<runtime::String> targets)
      : AnnotateTargetRewriter(std::move(targets)) {}

  std::unique_ptr<Call> RewriteVarCall(const Call& post_call) override {
    Array<Expr> ends;
    for (auto arg : post_call->args) {
      ends.push_back(InsertCompilerEndAndPropogateTarget(arg));
    }
    auto new_call = std::make_unique<Call>(post_call->op, ends, post_call->attrs);
    (*new_call)->checked_type_ = post_call->checked_type_;
    return new_call;
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) override {
    auto tuple = Downcast<Tuple>(post);
    Array<Expr> new_fields;
    new_fields.reserve(tuple->fields.size());

    for (auto f : tuple->fields) {
      new_fields.push_back(InsertCompilerEndAndPropogateTarget(f));
    }
    return WithFields(tuple, new_fields);
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) override {
    auto expr = Downcast<TupleGetItem>(post);
    return std::move(TupleGetItem(InsertCompilerEndAndPropogateTarget(expr->tuple), expr->index));
  }

  Expr Rewrite_(const IfNode* op, const Expr& post) override {
    auto expr = Downcast<If>(post);
    Expr new_cond = InsertCompilerEndAndPropogateTarget(expr->cond);
    Expr new_true_branch = InsertCompilerEndAndPropogateTarget(expr->true_branch);
    Expr new_false_branch = InsertCompilerEndAndPropogateTarget(expr->false_branch);

    auto new_expr = If(new_cond, new_true_branch, new_false_branch);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefCreateNode* op, const Expr& post) override {
    auto expr = Downcast<RefCreate>(post);
    auto new_expr = RefCreate(InsertCompilerEndAndPropogateTarget(expr->value));
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefReadNode* op, const Expr& post) override {
    auto expr = Downcast<RefRead>(post);
    auto new_expr = RefRead(InsertCompilerEndAndPropogateTarget(expr->ref));
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefWriteNode* op, const Expr& post) override {
    auto expr = Downcast<RefWrite>(post);
    auto new_expr = RefWrite(InsertCompilerEndAndPropogateTarget(expr->ref),
                             InsertCompilerEndAndPropogateTarget(expr->value));
    return std::move(new_expr);
  }
};

Expr AnnotateTarget(const Expr& expr, const Array<runtime::String>& targets,
                    bool include_non_call_ops) {
  auto r = include_non_call_ops ? std::make_unique<AnnotateTargetRewriter>(targets)
                                : std::make_unique<CallOpsTargetRewriter>(targets);
  return PostOrderRewrite(expr, r.get());
}

}  // namespace annotate_target

namespace transform {

Pass AnnotateTarget(const Array<runtime::String>& targets, bool include_non_call_ops) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::annotate_target::AnnotateTarget(f, targets, include_non_call_ops));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetFunc", {"InferType"});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTarget");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTarget").set_body_typed(AnnotateTarget);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
