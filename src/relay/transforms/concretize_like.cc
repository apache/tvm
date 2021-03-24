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
 * \file concretize_like.cc
 * \brief Converts `*_like` operators to their explicit shape equivalent (e.g. `zeros_like(x, y)` to
 * `zeros(x, y.shape)`), when the target shape is concrete. This removes unnecessary dependencies
 * and can enable more opportunities for operator fusion.
 */

#include <tvm/relay/transform.h>

#include "pattern_utils.h"
#include "simplify_expr.h"

namespace tvm {
namespace relay {

class ConcretizeLikeRewrite : public DFPatternRewrite {
 public:
  ConcretizeLikeRewrite(const Op& op) {
    ICHECK(op->num_inputs == 1 || op->num_inputs == 2)
        << "ConcretizeLike does not handle operators that aren't unary or binary, got: " << op;
    like_pat_ = IsWildcard();
    data_pat_ = IsWildcard();
    if (op->num_inputs == 1) {
      pattern_ = IsExpr(op)({like_pat_});
    } else {
      pattern_ = IsExpr(op)({data_pat_, like_pat_});
    }
    require_type_ = true;
  }

  virtual bool Check(const Expr& pre, const Expr& post,
                     const Map<DFPattern, Array<Expr>>& node_map) const {
    const CallNode* call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (!call_node->checked_type_.defined()) {
      // TODO(@altanh): maybe because of the input being rewritten?
      return false;
    }

    const TensorTypeNode* like_ty = call_node->checked_type().as<TensorTypeNode>();
    ICHECK(like_ty) << "got non-Tensor *_like call type " << PrettyPrint(call_node->checked_type());

    return true;
  }

  virtual Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                          DataType dtype) const = 0;

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    if (!Check(pre, post, node_map)) {
      return post;
    }

    const TensorTypeNode* like_ty = pre->checked_type().as<TensorTypeNode>();
    Array<Integer> cshape;
    for (const auto& dim : like_ty->shape) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        cshape.push_back(Integer(GetRef<IntImm>(imm)));
      } else {
        return post;
      }
    }

    return Concretize(node_map, cshape, like_ty->dtype);
  }

 protected:
  DFPattern data_pat_;
  DFPattern like_pat_;
};

class ConcretizeZerosLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeZerosLikeRewrite() : ConcretizeLikeRewrite(Op::Get("zeros_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeZeros(shape, dtype);
  }

  TVM_DF_PATTERN_REWRITE_GETTER(ConcretizeZerosLikeRewrite);
};

class ConcretizeOnesLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeOnesLikeRewrite() : ConcretizeLikeRewrite(Op::Get("ones_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeOnes(shape, dtype);
  }

  TVM_DF_PATTERN_REWRITE_GETTER(ConcretizeOnesLikeRewrite);
};

class ConcretizeReshapeLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeReshapeLikeRewrite() : ConcretizeLikeRewrite(Op::Get("reshape_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeReshape(node_map[data_pat_][0], shape);
  }

  TVM_DF_PATTERN_REWRITE_GETTER(ConcretizeReshapeLikeRewrite);
};

class ConcretizeCollapseSumLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeCollapseSumLikeRewrite() : ConcretizeLikeRewrite(Op::Get("collapse_sum_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    ICHECK_LE(shape.size(), std::numeric_limits<int64_t>::max());
    static const Op& op = Op::Get("collapse_sum_to");
    auto attrs = make_object<InitOpAttrs>();
    attrs->shape = shape;
    auto cshape =
        MakeConstantTensor(DataType::Int(32), {static_cast<int64_t>(shape.size())}, shape);
    return Call(op, {node_map[data_pat_][0], cshape}, Attrs(attrs));
  }

  TVM_DF_PATTERN_REWRITE_GETTER(ConcretizeCollapseSumLikeRewrite);
};

class ConcretizeBroadcastToLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeBroadcastToLikeRewrite() : ConcretizeLikeRewrite(Op::Get("broadcast_to_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeBroadCastTo(node_map[data_pat_][0], shape);
  }

  TVM_DF_PATTERN_REWRITE_GETTER(ConcretizeBroadcastToLikeRewrite);
};

Expr ConcretizeLike(const Expr& expr, const IRModule& mod) {
  static Array<DFPatternCallback> callbacks = {
      ConcretizeZerosLikeRewrite::GetCallback(), ConcretizeOnesLikeRewrite::GetCallback(),
      ConcretizeReshapeLikeRewrite::GetCallback(), ConcretizeCollapseSumLikeRewrite::GetCallback(),
      ConcretizeBroadcastToLikeRewrite::GetCallback()};
  return RewritePatterns(callbacks, expr, mod);
}

namespace transform {

Pass ConcretizeLike() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ConcretizeLike(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "ConcretizeLike", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ConcretizeLike").set_body_typed(ConcretizeLike);

}  // namespace transform

}  // namespace relay
}  // namespace tvm