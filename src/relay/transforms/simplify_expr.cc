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
 * \file src/relay/transforms/simplify_expr.cc
 * \brief A pass for simplifying the Relay expression.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/support/logging.h>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

class SimplifyPattern {
 public:
  virtual Expr callback(const Expr& pre, const Expr& post,
                        const Map<DFPattern, Array<Expr>>& node_map) const = 0;

  DFPattern pattern() const { return pattern_; }

 protected:
  /*! \brief Pattern for rewriting */
  DFPattern pattern_;
};

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op.
 */
class SimplifyReshape : public SimplifyPattern {
 public:
  SimplifyReshape() {
    x_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto reshape1 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    auto reshape2 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    pattern_ = reshape1({reshape2({x_})});
  }

  Expr callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];
    bool const_shape = true;
    Array<Integer> newshape;
    for (auto dim : Downcast<TensorType>(pre->checked_type())->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_shape = false;
        break;
      }
      newshape.push_back(Downcast<Integer>(dim));
    }
    if (const_shape) {
      return MakeReshape(x, newshape);
    }
    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

/*!
 * \brief FullArgwhere finds full followed by argwhere and turns it into an Arange op
 */
class FullArgwhere : public SimplifyPattern {
 public:
  FullArgwhere() {
    x_ = ConstantPattern(make_object<ConstantPatternNode>());
    full_ = IsOp("full")({x_}) ||
            IsOp("dyn.full")({x_, WildcardPattern(make_object<WildcardPatternNode>())});
    pattern_ = IsOp("argwhere")({full_});
  }

  Expr callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];
    auto dtype = pre->checked_type_.as<TensorTypeNode>()->dtype;
    auto shape = pre->checked_type_.as<TensorTypeNode>()->shape;
    if (IsConstScalar(x) && shape.size() == 2) {
      auto x_val = ToScalar(x.as<ConstantNode>()->data);
      if (x_val > 0) {
        Expr start = MakeConstantScalar(dtype, 0);
        Expr end = MakeTake(MakeShapeOf(node_map[full_][0], dtype), start, Integer(0), "clip");
        Expr step = MakeConstantScalar(dtype, 1);
        return MakeReshape(MakeArange(start, end, step, dtype), {-1, 1});
      }
    }
    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Full op */
  DFPattern full_;
};

/*!
 * \brief ExprSimplifier simplifies the Relay expression.
 */
class ExprSimplifier {
 public:
  explicit ExprSimplifier(IRModule mod) : mod_(mod) {
    CreateCallback(SimplifyReshape());
    CreateCallback(FullArgwhere());
  }
  template <typename T>
  void CreateCallback(const T& pattern) {
    auto func = [pattern](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = pattern.callback(pre, post, node_map);
    };
    callbacks_.push_back(DFPatternCallback(pattern.pattern(), PackedFunc(func), true));
  }

  Expr Simplify(const Expr& expr) { return RewritePatterns(callbacks_, expr, mod_); }

 private:
  IRModule mod_;
  /*! \brief Callbacks for expr simplification */
  Array<DFPatternCallback> callbacks_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  return ExprSimplifier(mod).Simplify(expr);
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SimplifyExpr(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
