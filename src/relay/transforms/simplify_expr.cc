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

#include "simplify_expr.h"

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op.
 */
class SimplifyReshape : public DFPatternRewrite {
 public:
  SimplifyReshape() {
    x_ = IsWildcard();
    auto reshape1 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    auto reshape2 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    pattern_ = reshape1({reshape2({x_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
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
 * \brief SimplifyCast matches the pattern of cast data to the same dtype.
 */
class SimplifyCast : public DFPatternRewrite {
 public:
  SimplifyCast() {
    data_pat_ = IsWildcard();
    like_pat_ = IsWildcard();
    pattern_ = IsOp("cast_like")({data_pat_, like_pat_}) || IsOp("cast")({data_pat_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const TensorTypeNode* data_ty = call->args[0]->checked_type().as<TensorTypeNode>();
    const TensorTypeNode* like_ty = pre->checked_type().as<TensorTypeNode>();
    if (like_ty->dtype == data_ty->dtype) {
      return node_map[data_pat_][0];
    }
    return post;
  }

 protected:
  DFPattern data_pat_;
  DFPattern like_pat_;
};

/*!
 * \brief SimplifyTranspose matches the pattern of consecutive transpose op,
 *   and merges or cancels them.
 */
class SimplifyTranspose : public DFPatternRewrite {
 public:
  SimplifyTranspose() {
    x_ = IsWildcard();
    auto trans1 = IsOp("transpose") || IsOp("layout_transform");
    auto trans2 = IsOp("transpose") || IsOp("layout_transform");
    pattern_ = trans1({trans2({x_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto x = node_map[x_][0];

    Call trans_call = Downcast<Call>(post);

    // Try to fuse any rank changing layout transformations
    if (auto layout_trans = FoldRankChangingLayoutTrans(x, trans_call)) {
      if (auto attr = layout_trans.value()->attrs.as<LayoutTransformAttrs>()) {
        // Prune any trivial layout transformation
        if (attr->src_layout == attr->dst_layout) {
          return x;
        }
      }
      return layout_trans.value();
    }

    // Initialize axes
    int ndim = Downcast<TensorType>(pre->checked_type())->shape.size();
    Array<Integer> axes;
    for (int i = 0; i < ndim; ++i) {
      axes.push_back(i);
    }

    // Collect axes changes from the matched pattern, including two consecutive transposes.
    std::vector<std::vector<int>> interm_axes;
    interm_axes.push_back(GetTransposeAxisOrder(trans_call, ndim));
    trans_call = Downcast<Call>(trans_call->args[0]);
    interm_axes.push_back(GetTransposeAxisOrder(trans_call, ndim));

    // Calculate the final axes in reverse order (from root to output)
    auto it = interm_axes.rbegin();
    while (it != interm_axes.rend()) {
      auto interm = *it;

      Array<Integer> new_axes;
      for (int i = 0; i < ndim; ++i) {
        new_axes.push_back(axes[interm[i]]);
      }
      axes = new_axes;
      it++;
    }

    // Check if the transpose is still required
    bool need_transpose = false;
    for (int i = 0; i < ndim; ++i) {
      if (axes[i] != i) {
        need_transpose = true;
        break;
      }
    }

    if (need_transpose) {
      return MakeTranspose(x, axes);
    }
    return x;
  }

  String PermuteLayout(const String& layout, std::vector<int> axes_order) const {
    std::string new_layout{};
    std::string old_layout{layout};
    ICHECK_EQ(axes_order.size(), layout.size())
        << "Number of axes must match the number of named axes in the layout to permute: length("
        << old_layout << ") != " << axes_order.size();
    std::stringstream order;
    for (auto axis : axes_order) {
      new_layout += old_layout[axis];
      order << axis << ", ";
    }
    DLOG(INFO) << "Using transpose axes order {" << order.str()
               << "} to permute layout: " << old_layout << " to " << new_layout;
    return new_layout;
  }

  struct RankChangingLayoutDescriptor {
    Layout src_layout;
    Layout dst_layout;
    // Either a rank changing layout transform or a transpose
    Call other_transform;
  };

  std::unique_ptr<RankChangingLayoutDescriptor> GetRankChangeDescriptor(const Call& call) const {
    std::unique_ptr<RankChangingLayoutDescriptor> desc{nullptr};
    if (auto attr = call->attrs.as<LayoutTransformAttrs>()) {
      if (attr->src_layout.length() != attr->dst_layout.length()) {
        desc = std::make_unique<RankChangingLayoutDescriptor>();
        desc->src_layout = Layout(attr->src_layout);
        desc->dst_layout = Layout(attr->dst_layout);
        desc->other_transform = Downcast<Call>(call->args[0]);
      }
    }
    if (auto attr = Downcast<Call>(call->args[0])->attrs.as<LayoutTransformAttrs>()) {
      if (attr->src_layout.length() != attr->dst_layout.length()) {
        if (!desc) {
          desc = std::make_unique<RankChangingLayoutDescriptor>();
          desc->src_layout = Layout(attr->src_layout);
          desc->dst_layout = Layout(attr->dst_layout);
          desc->other_transform = call;
        } else {
          ICHECK(desc->src_layout->name == attr->dst_layout)
              << "Back-to-back layout transforms must have the same intermediate layout: "
              << desc->src_layout->name << " != " << attr->dst_layout;
          desc->src_layout = Layout(attr->src_layout);
        }
      }
    }
    return desc;
  }

  /*
   * \brief Fuse call and it's argument into a single layout_transform operator
   * when either call or it's argument is a rang changing layout_transform, e.g.,
   *
   *  Simplify
   *
   *  [N, H, W, C] -> Transpose -> [N, C, H, W] -> LayoutTrans -> [N, C, H, W, 4c]
   *
   *  to,
   *
   *  [N, H, W, C] -> LayoutTrans -> [N, C, H, W, 4c].
   *
   * \param The input expression to the matched pattern
   * \param The pattern root; the second of two consecutive Transpose/LayoutTransform ops
   */
  Optional<Call> FoldRankChangingLayoutTrans(const Expr& data, const Call& call) const {
    // Check to see if either the first or second call in matched pattern
    // is a rank changing layout transform. If so, return a descriptor containing
    // the layouts and any additional transpose or layout transform op.
    auto desc = GetRankChangeDescriptor(call);
    if (desc == nullptr) {
      // No rank changing layout transform
      return Optional<Call>{nullptr};
    }

    Optional<Expr> output_layout_trans;
    // Fuse a rank increasing layout transform and a preceeding transpose
    if (desc->src_layout->axes.size() < desc->dst_layout->axes.size()) {
      auto axes = GetTransposeAxisOrder(desc->other_transform, desc->src_layout->axes.size());
      // Calculate the reverse axis order and apply to the source layout
      std::vector<int> inverse(axes.size());
      for (size_t i = 0; i < axes.size(); i++) {
        inverse[axes[i]] = i;
      }
      String new_layout = PermuteLayout(desc->src_layout->name, inverse);
      output_layout_trans = MakeLayoutTransform(data, new_layout, desc->dst_layout->name);
      // Fuse a rank descreasing layout transform followed by a transpose
    } else if (desc->src_layout->axes.size() > desc->dst_layout->axes.size()) {
      auto axes = GetTransposeAxisOrder(desc->other_transform, desc->dst_layout->axes.size());
      String new_layout = PermuteLayout(desc->dst_layout->name, axes);
      output_layout_trans = MakeLayoutTransform(data, desc->src_layout->name, new_layout);
      // Fuse two back-to-back layout transformations which change rank
    } else if (desc->other_transform->attrs.as<LayoutTransformAttrs>()) {
      output_layout_trans =
          MakeLayoutTransform(data, desc->src_layout->name, desc->dst_layout->name);
    }
    return Downcast<Call>(output_layout_trans);
  }

  std::vector<int> GetTransposeAxisOrder(const Call& call, int ndim) const {
    std::vector<int> attr_axes;
    if (auto attr = call->attrs.as<TransposeAttrs>()) {
      if (attr->axes.defined()) {
        for (int i = 0; i < ndim; ++i) {
          int64_t axis = attr->axes[i];
          axis += (axis < 0) ? ndim : 0;
          attr_axes.push_back(axis);
        }
      } else {
        // Empty axes means reverse
        for (int i = ndim - 1; i >= 0; --i) {
          attr_axes.push_back(i);
        }
      }
    } else if (auto attr = call->attrs.as<LayoutTransformAttrs>()) {
      Layout src_layout(attr->src_layout);
      Layout dst_layout(attr->dst_layout);
      for (int i = 0; i < ndim; ++i) {
        attr_axes.push_back(src_layout.IndexOf(dst_layout[i]));
      }
    } else {
      CHECK(false) << "Expected transpose or layout_transform, but got "
                   << Downcast<Op>(call->op)->name;
    }
    return std::move(attr_axes);
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

/*!
 * \brief FullElementwise finds full like ops followed by broadcasting ops, and eliminates
 * the full op by directly passing the fill value into the broadcasting op.
 */
class FullElementwise : public DFPatternRewrite {
 public:
  FullElementwise() {
    x_ = IsWildcard();
    data_ = IsWildcard();
    value_ = IsConstant();

    full_ = IsOp("full")({value_}) || IsOp("full_like")({data_, value_});
    ones_ = IsOp("ones")({}) || IsOp("ones_like")({data_});
    zeros_ = IsOp("zeros")({}) || IsOp("zeros_like")({data_});

    Map<String, ObjectRef> attrs;
    attrs.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    DFPattern op = IsWildcard().HasAttr(attrs);
    DFPattern full = full_ || ones_ || zeros_;
    pattern_ = op({full, x_}) || op({x_, full});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    Type pre_type = pre->checked_type_;
    ICHECK(pre_type.as<TensorTypeNode>());
    auto dtype = pre_type.as<TensorTypeNode>()->dtype;
    auto x = node_map[x_][0];
    bool is_left = post.as<CallNode>()->args[1] == x;
    Type x_type;
    if (is_left) {
      x_type = call->args[1]->checked_type_;
    } else {
      x_type = call->args[0]->checked_type_;
    }

    if (StructuralEqual()(x_type, pre_type)) {
      Expr value;
      if (node_map.count(full_)) {
        value = node_map[value_][0];
        ICHECK(IsConstScalar(value));
      } else if (node_map.count(ones_)) {
        value = MakeConstantScalar(dtype, 1);
      } else if (node_map.count(zeros_)) {
        value = MakeConstantScalar(dtype, 0);
      } else {
        ICHECK(false) << "Didn't find a full op while matching full + elementwise";
      }
      if (is_left) {
        return Call(call->op, {value, x}, call->attrs, call->type_args, call->span);
      } else {
        return Call(call->op, {x, value}, call->attrs, call->type_args, call->span);
      }
    }
    return post;
  }

 private:
  /*! \brief binary argument */
  DFPattern x_;
  /*! \brief data ops get shape from */
  DFPattern data_;
  /*! \brief constant input */
  DFPattern value_;
  /*! \brief full op */
  DFPattern full_;
  /*! \brief ones op */
  DFPattern ones_;
  /*! \brief zeros op */
  DFPattern zeros_;
};

/*!
 * \brief Converts `*_like` operators to their explicit shape equivalent (e.g. `zeros_like(x, y)` to
 * `zeros(x, y.shape)`), when the target shape is concrete. This removes unnecessary dependencies
 * and can enable more opportunities for operator fusion.
 */
class ConcretizeLikeRewrite : public DFPatternRewrite {
 public:
  explicit ConcretizeLikeRewrite(const Op& op) {
    ICHECK(op->num_inputs == 1 || op->num_inputs == 2)
        << "ConcretizeLike does not handle operators that aren't unary or binary, got: " << op;
    like_pat_ = IsWildcard();
    data_pat_ = IsWildcard();
    if (op->num_inputs == 1) {
      pattern_ = IsExpr(op)({like_pat_});
    } else {
      pattern_ = IsExpr(op)({data_pat_, like_pat_});
    }
  }

  virtual bool Check(const Expr& pre, const Expr& post,
                     const Map<DFPattern, Array<Expr>>& node_map) const {
    const CallNode* call_node = pre.as<CallNode>();
    ICHECK(call_node);

    if (!call_node->checked_type().as<TensorTypeNode>()) {
      return false;
    }

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
        // shape is not static, don't concretize
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
};

class ConcretizeOnesLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeOnesLikeRewrite() : ConcretizeLikeRewrite(Op::Get("ones_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeOnes(shape, dtype);
  }
};

class ConcretizeFullLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeFullLikeRewrite() : ConcretizeLikeRewrite(Op::Get("full_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    // `like_pat_` here is `fill_value`
    return MakeFull(node_map[like_pat_][0], shape, dtype);
  }
};

class ConcretizeReshapeLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeReshapeLikeRewrite() : ConcretizeLikeRewrite(Op::Get("reshape_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeReshape(node_map[data_pat_][0], shape);
  }
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
};

class ConcretizeBroadcastToLikeRewrite : public ConcretizeLikeRewrite {
 public:
  ConcretizeBroadcastToLikeRewrite() : ConcretizeLikeRewrite(Op::Get("broadcast_to_like")) {}

  Expr Concretize(const Map<DFPattern, Array<Expr>>& node_map, Array<Integer> shape,
                  DataType dtype) const override {
    return MakeBroadCastTo(node_map[data_pat_][0], shape);
  }
};

/*! \brief Eliminates expressions that are equivalent to identity. */
class EliminateIdentityRewrite : public DFPatternRewrite {
 public:
  EliminateIdentityRewrite() {
    x_ = IsWildcard();
    const_ = IsConstant();

    DFPattern add_op = IsOp("add");
    DFPattern mul_op = IsOp("multiply");
    DFPattern zeros_expr = IsOp("zeros")({}) || IsOp("zeros_like")({IsWildcard()}) || const_;
    DFPattern ones_expr = IsOp("ones")({}) || IsOp("ones_like")({IsWildcard()}) || const_;

    // add and multiply are commutative so we don't need another pattern for reversed args
    DFPattern add_id = add_op({x_, zeros_expr});
    DFPattern mul_id = mul_op({x_, ones_expr});

    DFPattern sub_id = IsOp("subtract")({x_, zeros_expr});
    DFPattern div_id = IsOp("divide")({x_, ones_expr});

    pattern_ = add_id || mul_id || sub_id || div_id;
  }

  bool CheckConstant(const OpNode* op, const ConstantNode* constant) const {
    if (!IsScalar(GetRef<Expr>(constant))) {
      return false;
    }
    auto value = TryToScalar(constant->data, 0);
    if (!value) {
      // unsupported dtype
      return false;
    }
    if (op->name == "add" || op->name == "subtract") {
      return value.value() == 0.0;
    } else if (op->name == "multiply" || op->name == "divide") {
      return value.value() == 1.0;
    }
    return false;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    Type pre_type = pre->checked_type_;
    ICHECK(pre_type.as<TensorTypeNode>());
    auto x = node_map[x_][0];
    bool is_left = post.as<CallNode>()->args[1] == x;
    Type x_type;
    if (is_left) {
      x_type = call->args[1]->checked_type_;
    } else {
      x_type = call->args[0]->checked_type_;
    }

    if (node_map.count(const_)) {
      // the other argument is a Constant in this case
      const ConstantNode* constant = node_map[const_][0].as<ConstantNode>();
      const OpNode* op = call->op.as<OpNode>();
      ICHECK(constant);
      ICHECK(op);
      if (!CheckConstant(op, constant)) {
        return post;
      }
    }

    if (StructuralEqual()(x_type, pre_type)) {
      return x;
    }

    return post;
  }

 private:
  DFPattern x_;
  DFPattern const_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  // the rewrites will be applied in the given order, and repeated until fixed point
  DFPatternRewriteComposer composer;
  composer.AddRewrite<ConcretizeZerosLikeRewrite>();
  composer.AddRewrite<ConcretizeOnesLikeRewrite>();
  composer.AddRewrite<ConcretizeFullLikeRewrite>();
  composer.AddRewrite<ConcretizeReshapeLikeRewrite>();
  composer.AddRewrite<ConcretizeCollapseSumLikeRewrite>();
  composer.AddRewrite<ConcretizeBroadcastToLikeRewrite>();
  composer.AddRewrite<EliminateIdentityRewrite>();
  composer.AddRewrite<SimplifyReshape>();
  composer.AddRewrite<SimplifyTranspose>();
  composer.AddRewrite<SimplifyCast>();
  composer.AddRewrite<FullElementwise>();
  return RewritePatterns(composer.MakeCallbacks(), expr, mod);
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
