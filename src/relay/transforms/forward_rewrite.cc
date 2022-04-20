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
 *
 * \file forward_rewrite.cc
 * \brief Apply rewriting rules in a forward fashion.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "pass_utils.h"

namespace tvm {
namespace relay {

// Realizer class that realizes the expression
// Note that we can take benefit of its internal memo
// so that calling realize repeatively won't hurt perf.
class TempRealizer : private MixedModeMutator {
 public:
  Expr Realize(Expr expr) { return Mutate(expr); }

 private:
  Expr DispatchVisitExpr(const Expr& expr) final {
    Expr res;
    if (const auto* temp = expr.as<TempExprNode>()) {
      res = temp->Realize();
    } else {
      res = MixedModeMutator::DispatchVisitExpr(expr);
    }
    return res;
  }
};

class ForwardRewriter : private MixedModeMutator {
 public:
  ForwardRewriter(const OpAttrMap<FForwardRewrite>* rewrite_map,
                  std::function<ObjectRef(const Call&)> fcontext,
                  std::function<Expr(const Expr&)> fmulti_ref_trigger)
      : rewrite_map_(rewrite_map), fcontext_(fcontext), fmulti_ref_trigger_(fmulti_ref_trigger) {}

  ForwardRewriter(const FForwardRewrite* rewrite_func,
                  std::function<ObjectRef(const Call&)> fcontext,
                  std::function<Expr(const Expr&)> fmulti_ref_trigger)
      : rewrite_func_(rewrite_func), fcontext_(fcontext), fmulti_ref_trigger_(fmulti_ref_trigger) {}

  // Transform expression.
  Expr Rewrite(const Expr& expr) {
    if (fmulti_ref_trigger_ != nullptr) {
      ref_counter_ = GetExprRefCount(expr);
    }
    return realizer_.Realize(this->VisitExpr(expr));
  }

 private:
  // The rewrite rule.
  const OpAttrMap<FForwardRewrite>* rewrite_map_{nullptr};
  const FForwardRewrite* rewrite_func_{nullptr};
  // The context.const
  std::function<ObjectRef(const Call&)> fcontext_{nullptr};
  // The multiple reference trigger
  std::function<Expr(const Expr&)> fmulti_ref_trigger_{nullptr};
  // Internal ref counter
  std::unordered_map<const Object*, size_t> ref_counter_;
  // internal realizer
  TempRealizer realizer_;

  // Visit and allow non-realized version.
  Expr GetTempExpr(const Expr& expr, const Expr& post) {
    if (fmulti_ref_trigger_ != nullptr) {
      Expr ret = post;
      auto it = ref_counter_.find(expr.get());
      ICHECK(it != ref_counter_.end());
      if (it->second > 1) {
        ret = fmulti_ref_trigger_(ret);
      }
      return ret;
    } else {
      return post;
    }
  }

  // Automatic fold TupleGetItem.
  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) final {
    Expr tuple = this->GetTempExpr(op->tuple, post.as<TupleGetItemNode>()->tuple);
    if (const auto* ptuple = tuple.as<TupleNode>()) {
      return ptuple->fields[op->index];
    } else {
      if (tuple.same_as(op->tuple)) {
        return GetRef<Expr>(op);
      } else {
        return TupleGetItem(tuple, op->index);
      }
    }
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) final {
    tvm::Array<Expr> fields;
    fields.reserve(tuple_node->fields.size());

    const auto* post_tuple_node = post.as<TupleNode>();
    for (size_t i = 0; i < tuple_node->fields.size(); ++i) {
      fields.push_back(this->GetTempExpr(tuple_node->fields[i], post_tuple_node->fields[i]));
    }

    return WithFields(GetRef<Tuple>(tuple_node), fields);
  }

  Expr Rewrite_(const CallNode* call_node, const Expr& post) final {
    const Call& ref_call = GetRef<Call>(call_node);
    PackedFunc frewrite;
    if (rewrite_func_) {
      frewrite = *rewrite_func_;
    } else {
      ICHECK(rewrite_map_);
      frewrite = rewrite_map_->get(call_node->op, nullptr);
    }
    const auto* post_node = post.as<CallNode>();
    auto new_op = post_node->op;
    if (new_op->IsInstance<FunctionNode>()) {
      new_op = realizer_.Realize(new_op);
    }
    bool unchanged = call_node->op.same_as(new_op);

    Array<Expr> call_args;
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      Expr new_arg = this->GetTempExpr(call_node->args[i], post_node->args[i]);
      if (frewrite == nullptr) {
        new_arg = realizer_.Realize(new_arg);
      }
      unchanged &= new_arg.same_as(call_node->args[i]);
      call_args.push_back(new_arg);
    }
    // try to rewrite.
    if (frewrite != nullptr) {
      Expr res = frewrite(ref_call, call_args,
                          fcontext_ != nullptr ? fcontext_(ref_call) : ObjectRef(nullptr));
      if (res.defined()) return res;
      // abort, use old rule
      for (size_t i = 0; i < call_args.size(); ++i) {
        Expr arg = call_args[i];
        Expr new_arg = realizer_.Realize(arg);
        if (!arg.same_as(new_arg)) {
          call_args.Set(i, new_arg);
          unchanged = false;
        }
      }
    }
    if (unchanged) return ref_call;
    return Call(new_op, call_args, call_node->attrs, call_node->type_args, call_node->span);
  }
};

Expr ForwardRewrite(const Expr& expr, const String& rewrite_map_name,
                    std::function<ObjectRef(const Call&)> fcontext,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger) {
  auto rewrite_map = Op::GetAttrMap<FForwardRewrite>(rewrite_map_name);
  return ForwardRewriter(&rewrite_map, fcontext, fmulti_ref_trigger).Rewrite(expr);
}

Expr ForwardRewrite(const Expr& expr, const FForwardRewrite& rewrite_func,
                    std::function<ObjectRef(const Call&)> fcontext,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger) {
  return ForwardRewriter(&rewrite_func, fcontext, fmulti_ref_trigger).Rewrite(expr);
}

}  // namespace relay
}  // namespace tvm
