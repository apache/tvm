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
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

// Pattern Matcher

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  bool Match(const DFPattern& pattern, const Expr& expr);

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  std::unordered_map<DFPattern, Expr, ObjectHash, ObjectEqual> memo_;
};

bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  return VisitDFPattern(pattern, expr);
}

bool DFPatternMatcher::VisitDFPattern(const DFPattern& pattern, const Expr& expr) {
  if (memo_.count(pattern)) {
    return expr.same_as(memo_[pattern]);
  } else {
    auto out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern] = expr;
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const AltPatternNode* op, const Expr& expr) {
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}
bool DFPatternMatcher::VisitDFPattern_(const AttrPatternNode* attr_pattern, const Expr& expr) {
  bool matches = false;
  if (const auto* op_node = expr.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    auto attributes = attr_pattern->attrs.as<DictAttrsNode>()->dict;
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      auto op_map = Op::GetAttr<TVMRetValue>(attr_name);
      if (op_map.count(op)) {
        switch (op_map[op].type_code()) {
          case kDLInt:
            if (auto* val = kv.second.as<IntImmNode>()) {
              matches = val->value == op_map[op].operator int64_t();
            }
            break;
          case kDLFloat:
            if (auto* val = kv.second.as<FloatImmNode>()) {
              matches = val->value == op_map[op].operator double();
            }
            break;
          case kTVMStr:
            if (auto* val = kv.second.as<tir::StringImmNode>()) {
              matches = val->value == op_map[op].operator std::string();
            }
            break;
          default:
            throw "Unsupported type";
        }
      }
    }
  }
  return matches;
}
bool DFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* call_node = expr.as<CallNode>()) {
    if (op->args.size() == call_node->args.size()) {
      matches = VisitDFPattern(op->op, call_node->op);
      size_t i = 0;
      while (matches && i < op->args.size()) {
        matches &= VisitDFPattern(op->args[i], call_node->args[i]);
        ++i;
      }
    }
  }
  return matches;
}
bool DFPatternMatcher::VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) {
  return op->expr == expr;
}
bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    matches = (op->index == tuple_get_item_node->index) &&
              VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return matches;
}
bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleNode>()) {
    if (op->fields.size() == tuple_node->fields.size()) {
      matches = true;
      size_t i = 0;
      while (matches && i < op->fields.size()) {
        matches &= VisitDFPattern(op->fields[i], tuple_node->fields[i]);
        ++i;
      }
    }
  }
  return matches;
}
Expr InferType(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::InferType()(mod);
  if (expr.as<FunctionNode>()) {
    return mod->Lookup("main");
  } else {
    return mod->Lookup("main").as<FunctionNode>()->body;
  }
}
bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}
bool DFPatternMatcher::VisitDFPattern_(const VarPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* var_node = expr.as<VarNode>()) {
    matches = true;
    if (op->name_hint() != "") {
      matches &= op->name_hint() == var_node->name_hint();
    }
  }
  return matches;
}
bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) {
  return true;
}

// DFPatternMutator

DFPattern DFPatternMutator::Mutate(const DFPattern& pattern) { return VisitDFPattern(pattern); }

DFPattern DFPatternMutator::VisitDFPattern(const DFPattern& pattern) {
  auto it = this->memo_.find(pattern);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    auto new_pattern = DFPatternFunctor::VisitDFPattern(pattern);
    memo_[pattern] = new_pattern;
    return new_pattern;
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const AltPatternNode* op) {
  auto new_left = Mutate(op->left);
  auto new_right = Mutate(op->right);

  if (new_left.same_as(op->left) && new_right.same_as(op->right)) {
    return GetRef<DFPattern>(op);
  } else {
    return AltPatternNode::make(new_left, new_right);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const AttrPatternNode* op) {
  auto new_pattern = Mutate(op->pattern);
  if (new_pattern.same_as(op->pattern)) {
    return GetRef<DFPattern>(op);
  } else {
    return AttrPatternNode::make(new_pattern, op->attrs);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const CallPatternNode* op) {
  auto new_op = Mutate(op->op);
  bool unchanged = op->op.same_as(new_op);
  tvm::Array<DFPattern> call_args;
  for (auto arg : op->args) {
    auto new_arg = Mutate(arg);
    call_args.push_back(new_arg);
    unchanged &= arg.same_as(new_arg);
  }
  if (unchanged) {
    return GetRef<DFPattern>(op);
  } else {
    return CallPatternNode::make(new_op, call_args, op->attrs, op->type_args);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const ExprPatternNode* op) {
  return GetRef<DFPattern>(op);
}

DFPattern DFPatternMutator::VisitDFPattern_(const TupleGetItemPatternNode* op) {
  auto new_tuple = Mutate(op->tuple);
  if (new_tuple.same_as(op->tuple)) {
    return GetRef<DFPattern>(op);
  } else {
    return TupleGetItemPatternNode::make(op->tuple, op->index);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const TuplePatternNode* op) {
  bool unchanged = true;
  tvm::Array<DFPattern> fields;
  for (auto field : op->fields) {
    auto new_field = Mutate(field);
    fields.push_back(new_field);
    unchanged &= field.same_as(new_field);
  }
  if (unchanged) {
    return GetRef<DFPattern>(op);
  } else {
    return TuplePatternNode::make(fields);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const TypePatternNode* op) {
  auto new_pattern = Mutate(op->pattern);
  if (new_pattern.same_as(op->pattern)) {
    return GetRef<DFPattern>(op);
  } else {
    return TypePatternNode::make(new_pattern, op->type);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const VarPatternNode* op) {
  return GetRef<DFPattern>(op);
}

DFPattern DFPatternMutator::VisitDFPattern_(const WildcardPatternNode* op) {
  return GetRef<DFPattern>(op);
}

// Prepare

class DFPatternPrepare : protected DFPatternMutator {
 public:
  DFPattern Prepare(const DFPattern& pattern) { return Mutate(pattern); }
  DFPattern VisitDFPattern_(const CallPatternNode* op) {
    auto post = DFPatternMutator::VisitDFPattern_(op);
    auto* post_node = post.as<CallPatternNode>();
    if (auto* expr_pattern = post_node->op.as<ExprPatternNode>()) {
      if (auto* op_node = expr_pattern->expr.as<OpNode>()) {
        if ((op_node->name == "add") || (op_node->name == "multiply")) {
          tvm::Array<DFPattern> call_args;
          for (auto it = post_node->args.rbegin(); it != post_node->args.rend(); ++it) {
            call_args.push_back(*it);
          }
          return AltPatternNode::make(
              post, CallPatternNode::make(post_node->op, call_args, post_node->attrs,
                                          post_node->type_args));
        }
      }
    }
    return post;
  }
};

TVM_REGISTER_GLOBAL("relay.df_pattern.match").set_body_typed([](DFPattern pattern, Expr expr) {
  return DFPatternMatcher().Match(DFPatternPrepare().Prepare(pattern), expr);
});

// Rewrite

DFPatternCallback DFPatternCallbackNode::make(DFPattern pattern, PackedFunc function) {
  ObjectPtr<DFPatternCallbackNode> n = make_object<DFPatternCallbackNode>();
  n->pattern_ = std::move(pattern);
  n->function_ = std::move(function);
  return DFPatternCallback(n);
}

TVM_REGISTER_NODE_TYPE(DFPatternCallbackNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.DFPatternCallback")
.set_body_typed(DFPatternCallbackNode::make);

class PatternRewriter : public ExprMutator {
 public:
  PatternRewriter(const Array<DFPatternCallback>& callbacks) : callbacks_(callbacks) {}
  Expr Rewrite(const Expr& pre) {
    return this->VisitExpr(pre);
  }

 protected:
  Expr VisitExpr(const Expr& pre) override {
    auto post = ExprMutator::VisitExpr(pre);
    Expr out = post;
    for (auto& callback : callbacks_) {
      if (auto* callback_node = callback.as<DFPatternCallbackNode>()) {
        if (matcher_.Match(callback_node->pattern_, out)) {
          out = callback_node->function_(pre, out);
        }
      }
    }
    return out;
  }
  DFPatternMatcher matcher_;
  Array<DFPatternCallback> callbacks_;
};

TVM_REGISTER_GLOBAL("relay.df_pattern.rewrite")
.set_body_typed([](Array<DFPatternCallback> callbacks, Expr expr) {
  return PatternRewriter(callbacks).Rewrite(expr);
});

}  // namespace relay
}  // namespace tvm
