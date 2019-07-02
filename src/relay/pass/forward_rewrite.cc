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
 * Copyright (c) 2018 by Contributors
 *
 * \file forward_rewrite.cc
 * \brief Apply rewriting rules in a forward fashion.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include "pass_util.h"

namespace tvm {
namespace relay {

// Realizer class that realizes the expression
// Note that we can take benefit of its internal memo
// so that calling realize repeatively won't hurt perf.
class TempRealizer : private ExprMutator {
 public:
  Expr Realize(Expr expr) {
    return VisitExpr(expr);
  }

 private:
  Expr VisitExpr(const Expr& expr) final {
    auto it = memo_.find(expr);
    if (it != memo_.end()) {
      return it->second;
    } else {
      Expr res;
      if (const auto* temp = expr.as_derived<TempExprNode>()) {
        res = temp->Realize();

      } else {
        res = ExprFunctor::VisitExpr(expr);
      }
      memo_[res] = res;
      return res;
    }
  }
};

class ForwardRewriter : private ExprMutator {
 public:
  ForwardRewriter(const OpMap<FForwardRewrite>* rewrite_map,
                  std::function<NodeRef(const Call&)> fcontext,
                  std::function<Expr(const Expr&)> fmulti_ref_trigger)
      : rewrite_map_(rewrite_map),
        fcontext_(fcontext),
        fmulti_ref_trigger_(fmulti_ref_trigger) {}

  ForwardRewriter(const FForwardRewrite* rewrite_func,
                  std::function<NodeRef(const Call&)> fcontext,
                  std::function<Expr(const Expr&)> fmulti_ref_trigger)
      : rewrite_func_(rewrite_func),
        fcontext_(fcontext),
        fmulti_ref_trigger_(fmulti_ref_trigger) {}


  // Transform expression.
  Expr Rewrite(Expr expr) {
    if (fmulti_ref_trigger_ != nullptr) {
      ref_counter_ = GetExprRefCount(expr);
    }
    return this->VisitExpr(expr);
  }

 private:
  // The rewrite rule.
  const OpMap<FForwardRewrite>* rewrite_map_{nullptr};
  const FForwardRewrite* rewrite_func_{nullptr};
  // The context.const
  std::function<NodeRef(const Call&)> fcontext_{nullptr};
  // The multiple reference trigger
  std::function<Expr(const Expr&)> fmulti_ref_trigger_{nullptr};
  // Internal ref counter
  std::unordered_map<const Node*, size_t> ref_counter_;
  // internal realizer
  TempRealizer realizer_;

  Expr VisitExpr(const Expr& expr) final {
    // by default always realize.
    return realizer_.Realize(ExprMutator::VisitExpr(expr));
  }

  // Visit and allow non-realized version.
  Expr GetTempExpr(const Expr& expr)  {
    if (fmulti_ref_trigger_ != nullptr) {
      Expr ret = ExprMutator::VisitExpr(expr);
      auto it = ref_counter_.find(expr.get());
      CHECK(it != ref_counter_.end());
      if (it->second > 1) {
        ret = fmulti_ref_trigger_(ret);
      }
      return ret;
    } else {
      return ExprMutator::VisitExpr(expr);
    }
  }

  // Automatic fold TupleGetItem.
  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr tuple = this->GetTempExpr(op->tuple);
    if (const auto* ptuple = tuple.as<TupleNode>()) {
      return ptuple->fields[op->index];
    } else {
      if (tuple.same_as(op->tuple)) {
        return GetRef<Expr>(op);
      } else {
        return TupleGetItemNode::make(tuple, op->index);
      }
    }
  }

  Expr VisitExpr_(const TupleNode* op) final {
    tvm::Array<Expr> fields;
    bool all_fields_unchanged = true;
    for (auto field : op->fields) {
      auto new_field = this->GetTempExpr(field);
      fields.push_back(new_field);
      all_fields_unchanged &= new_field.same_as(field);
    }

    if (all_fields_unchanged) {
      return GetRef<Expr>(op);
    } else {
      return TupleNode::make(fields);
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    const Call& ref_call = GetRef<Call>(call_node);
    PackedFunc frewrite;
    if (rewrite_func_) {
      frewrite = *rewrite_func_;
    } else {
      CHECK(rewrite_map_);
      frewrite = rewrite_map_->get(call_node->op, nullptr);
    }

    auto new_op = this->Mutate(call_node->op);
    bool unchanged = call_node->op.same_as(new_op);

    Array<Expr> call_args;
    for (auto arg : call_node->args) {
      Expr new_arg = this->GetTempExpr(arg);
      if (frewrite == nullptr) {
        new_arg = realizer_.Realize(new_arg);
      }
      unchanged &= new_arg.same_as(arg);
      call_args.push_back(new_arg);
    }
    // try to rewrite.
    if (frewrite != nullptr) {
      Expr res = frewrite(
          ref_call, call_args,
          fcontext_ != nullptr ? fcontext_(ref_call) : NodeRef(nullptr));
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
    return CallNode::make(
        new_op, call_args, call_node->attrs, call_node->type_args);
  }
};

Expr ForwardRewrite(const Expr& expr,
                    const std::string& rewrite_map_name,
                    std::function<NodeRef(const Call&)> fcontext,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger) {
  auto rewrite_map = Op::GetAttr<FForwardRewrite>(rewrite_map_name);
  return ForwardRewriter(&rewrite_map, fcontext, fmulti_ref_trigger).Rewrite(expr);
}

Expr ForwardRewrite(const Expr& expr,
                    const FForwardRewrite& rewrite_func,
                    std::function<NodeRef(const Call&)> fcontext,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger) {
  return ForwardRewriter(&rewrite_func, fcontext, fmulti_ref_trigger).Rewrite(expr);
}

}  // namespace relay
}  // namespace tvm
