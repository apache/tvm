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
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class ConcretizeLikeRewrite {
 public:
  ConcretizeLikeRewrite() {
    concrete_map_[Op::Get("reshape_like")] = [](Expr data, Array<Integer> shape, DataType dtype) {
      return MakeReshape(data, shape);
    };
    concrete_map_[Op::Get("zeros_like")] = [](Expr data, Array<Integer> shape, DataType dtype) {
      return MakeZeros(shape, dtype);
    };
    concrete_map_[Op::Get("ones_like")] = [](Expr data, Array<Integer> shape, DataType dtype) {
      return MakeOnes(shape, dtype);
    };
    concrete_map_[Op::Get("collapse_sum_like")] = [](Expr data, Array<Integer> shape,
                                                     DataType dtype) {
      ICHECK_LE(shape.size(), std::numeric_limits<int64_t>::max());
      static const Op& op = Op::Get("collapse_sum_to");
      auto attrs = make_object<InitOpAttrs>();
      auto cshape =
          MakeConstantTensor(DataType::Int(32), {static_cast<int64_t>(shape.size())}, shape);
      attrs->shape = shape;
      return Call(op, {data, cshape}, Attrs(attrs));
    };
    concrete_map_[Op::Get("broadcast_to_like")] = [](Expr data, Array<Integer> shape,
                                                     DataType dtype) {
      return MakeBroadCastTo(data, shape);
    };

    for (const auto& pr : concrete_map_) {
      if (!op_pat_.defined()) {
        op_pat_ = IsExpr(pr.first);
      } else {
        op_pat_ = op_pat_ || IsExpr(pr.first);
      }
    }

    data_pat_ = IsWildcard();
    like_pat_ = IsWildcard();
    unary_like_pat_ = (IsOp("zeros_like") || IsOp("ones_like"))({like_pat_});
    binary_like_pat_ = (IsOp("reshape_like") || IsOp("collapse_sum_like") ||
                        IsOp("broadcast_to_like"))({data_pat_, like_pat_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const {
    // we will rewrite iff the like argument has fully concrete shape
    const CallNode* call_node = post.as<CallNode>();
    ICHECK(call_node);
    const OpNode* op_node = call_node->op.as<OpNode>();
    ICHECK(op_node);
    const Op op_ref = GetRef<Op>(op_node);
    ICHECK(concrete_map_.count(op_ref) > 0);

    Expr like = node_map[like_pat_][0];

    if (!like->checked_type_.defined()) {
      // TODO(@altanh): maybe because of the input being rewritten?
      return post;
    }

    // skip trying to support this for now (ironic, as I was the one who added the feature)
    if (const auto* attrs = call_node->attrs.as<ReshapeLikeAttrs>()) {
      if (attrs->lhs_begin != 0 || attrs->rhs_begin != 0 || attrs->lhs_end.defined() ||
          attrs->rhs_end.defined()) {
        return post;
      }
    }

    CHECK(like->checked_type_.defined())
        << "ConcretizeLike requires checked types to be populated, please run type inference";
    const TensorTypeNode* like_ty = like->checked_type().as<TensorTypeNode>();
    ICHECK(like_ty) << "got non-Tensor argument type " << PrettyPrint(like->checked_type());

    Array<Integer> cshape;
    for (const auto& dim : like_ty->shape) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        cshape.push_back(Integer(GetRef<IntImm>(imm)));
        continue;
      }
      return post;
    }

    if (call_node->args.size() == 2) {
      return concrete_map_.at(op_ref)(node_map[data_pat_][0], cshape, like_ty->dtype);
    }
    return concrete_map_.at(op_ref)(Expr(), cshape, like_ty->dtype);
  }

  DFPattern UnaryPattern() const { return unary_like_pat_; }

  DFPattern BinaryPattern() const { return binary_like_pat_; }

 private:
  using FMake = std::function<Expr(Expr, Array<Integer>, DataType)>;
  std::unordered_map<Op, FMake, ObjectPtrHash, ObjectPtrEqual> concrete_map_;
  DFPattern op_pat_;
  DFPattern data_pat_;
  DFPattern like_pat_;
  DFPattern unary_like_pat_;
  DFPattern binary_like_pat_;
};

namespace transform {

Pass ConcretizeLike() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) {
        ConcretizeLikeRewrite rw;
        auto callback_func = PackedFunc([&rw](TVMArgs args, TVMRetValue* rv) {
          Expr pre = args[0];
          Expr post = args[1];
          Map<DFPattern, Array<Expr>> node_map = args[2];
          *rv = rw.Callback(pre, post, node_map);
        });
        Array<DFPatternCallback> callbacks = {
            DFPatternCallback(rw.UnaryPattern(), callback_func, true),
            DFPatternCallback(rw.BinaryPattern(), callback_func, true)};
        return Downcast<Function>(RewritePatterns(callbacks, f, m));
      };
  return CreateFunctionPass(pass_func, 0, "ConcretizeLike", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ConcretizeLike").set_body_typed(ConcretizeLike);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
