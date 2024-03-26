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
  software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file dynamic_to_static.cc
 * \brief Rewrite Dynamic Operations to Static operations where possible
 */
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class DynamicToStaticMutator : public MixedModeMutator {
 public:
  DynamicToStaticMutator(IRModule mod, Function func) : mod_(mod), func_(func) {
    op_map_ = {
        {Op::Get("dyn.reshape"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[1].as<ConstantNode>()) {
             ICHECK_EQ(shape->data->ndim, 1);
             return MakeReshape(call_node->args[0], ToVector(shape->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.squeeze"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* axis = args[1].as<ConstantNode>()) {
             ICHECK_EQ(axis->data->ndim, 1);
             return MakeSqueeze(call_node->args[0], ToVector(axis->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.tile"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* reps = args[1].as<ConstantNode>()) {
             ICHECK_EQ(reps->data->ndim, 1);
             return MakeTile(call_node->args[0], ToVector(reps->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.topk"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* k = args[1].as<ConstantNode>()) {
             const TopKAttrs* param = call_node->attrs.as<TopKAttrs>();
             ICHECK(param);
             return MakeTopK(call_node->args[0], static_cast<int>(ToScalar(k->data, 0)),
                             param->axis, param->ret_type, param->is_ascend, param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.broadcast_to"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[1].as<ConstantNode>()) {
             ICHECK_EQ(shape->data->ndim, 1);
             return MakeBroadCastTo(call_node->args[0], ToVector(shape->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.zeros"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[0].as<ConstantNode>()) {
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             ICHECK(param);
             return MakeZeros(ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.ones"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[0].as<ConstantNode>()) {
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             ICHECK(param);
             return MakeOnes(ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.one_hot"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* depth = args[3].as<ConstantNode>()) {
             const OneHotAttrs* param = call_node->attrs.as<OneHotAttrs>();
             ICHECK(param);
             return MakeOneHot(call_node->args[0], call_node->args[1], call_node->args[2],
                               static_cast<int>(ToScalar(depth->data, 0)), param->axis,
                               param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.image.resize2d"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* size = args[1].as<ConstantNode>()) {
             if (const ConstantNode* roi = args[2].as<ConstantNode>()) {
               const Resize2DAttrs* param = call_node->attrs.as<Resize2DAttrs>();
               ICHECK(param);
               auto size_int = ToVector(size->data);
               Array<PrimExpr> size_prim;
               for (size_t i = 0; i < size_int.size(); ++i) {
                 size_prim.push_back(size_int[i]);
               }
               auto roi_vec = ToFloatVector(roi->data);
               Array<FloatImm> roi_prim;
               for (size_t i = 0; i < roi_vec.size(); ++i) {
                 roi_prim.push_back(roi_vec[i]);
               }
               return MakeResize2D(call_node->args[0], size_prim, roi_prim, param->layout,
                                   param->method, param->coordinate_transformation_mode,
                                   param->rounding_method, param->cubic_alpha, param->cubic_exclude,
                                   param->extrapolation_value, param->out_dtype);
             }
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.full"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[1].as<ConstantNode>()) {
             ICHECK_EQ(shape->data->ndim, 1);
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             ICHECK(param);
             return MakeFull(call_node->args[0], ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.upsampling"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           const ConstantNode* scale_h = args[1].as<ConstantNode>();
           const ConstantNode* scale_w = args[2].as<ConstantNode>();
           if (scale_h && scale_w) {
             ICHECK_EQ(scale_h->data->ndim, 0);
             ICHECK_EQ(scale_w->data->ndim, 0);
             const UpSamplingAttrs* param = call_node->attrs.as<UpSamplingAttrs>();
             ICHECK(param);
             return MakeUpSampling(call_node->args[0], ToScalar(scale_h->data),
                                   ToScalar(scale_w->data), param->layout, param->method,
                                   param->align_corners);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.upsampling3d"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           const ConstantNode* scale_d = args[1].as<ConstantNode>();
           const ConstantNode* scale_h = args[2].as<ConstantNode>();
           const ConstantNode* scale_w = args[3].as<ConstantNode>();
           if (scale_d && scale_h && scale_w) {
             ICHECK_EQ(scale_d->data->ndim, 0);
             ICHECK_EQ(scale_h->data->ndim, 0);
             ICHECK_EQ(scale_w->data->ndim, 0);
             const UpSampling3DAttrs* param = call_node->attrs.as<UpSampling3DAttrs>();
             ICHECK(param);
             return MakeUpSampling3D(call_node->args[0], ToScalar(scale_d->data),
                                     ToScalar(scale_h->data), ToScalar(scale_w->data),
                                     param->layout, param->method,
                                     param->coordinate_transformation_mode);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.pad"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           const ConstantNode* pad_width = args[1].as<ConstantNode>();
           const ConstantNode* pad_fill = args[2].as<ConstantNode>();
           if (pad_width && pad_fill) {
             ICHECK_EQ(pad_fill->data->ndim, 0);   // pad_val is 1d
             ICHECK_EQ(pad_width->data->ndim, 2);  // pad_width is 2d

             const PadAttrs* param = call_node->attrs.as<PadAttrs>();
             ICHECK(param);

             Expr pad_value = args[2];
             return MakePad(call_node->args[0], ToMatrix(pad_width->data), pad_value,
                            param->pad_mode);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.strided_slice"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           const ConstantNode* begin = args[1].as<ConstantNode>();
           const ConstantNode* end = args[2].as<ConstantNode>();
           const ConstantNode* stride = args[3].as<ConstantNode>();
           if (begin && end && stride) {
             ICHECK_EQ(begin->data->ndim, 1);
             ICHECK_EQ(end->data->ndim, 1);
             ICHECK_EQ(stride->data->ndim, 1);
             const StridedSliceAttrs* param = call_node->attrs.as<StridedSliceAttrs>();
             ICHECK(param);
             return MakeStridedSlice(call_node->args[0], ToVector(begin->data), ToVector(end->data),
                                     ToVector(stride->data), param->slice_mode);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.sparse_to_dense"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           const ConstantNode* output_shape = args[3].as<ConstantNode>();
           if (output_shape) {
             ICHECK_EQ(output_shape->data->ndim, 1);
             return MakeSparseToDense(call_node->args[0], ToVector(output_shape->data),
                                      call_node->args[1], call_node->args[2]);
           }
           return Expr(nullptr);
         }},
    };
    Map<BaseFunc, GlobalVar> vars;
    for (auto kv : mod_->functions) {
      vars.Set(kv.second, kv.first);
    }
    gv_ = vars[func_];
  }

  Expr GetCurExpr(const Expr& original_expr) {
    if (original_expr.as<FunctionNode>()) {
      return mod_->Lookup(gv_);
    } else {
      return mod_->Lookup(gv_).as<FunctionNode>()->body;
    }
  }

  Expr PrepareInput(const Expr& expr) {
    BaseFunc func;
    if (auto func_node = expr.as<BaseFunc>()) {
      func = func_node.value();
    } else {
      func = relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod_));
    }
    mod_->Update(gv_, func);

    mod_ = transform::FoldConstant()(mod_);
    transform::InferTypeLocal(GetCurExpr(expr));
    mod_ = transform::FoldConstant()(mod_);
    transform::InferTypeLocal(GetCurExpr(expr));

    Expr out;
    if (expr.as<FunctionNode>()) {
      out = mod_->Lookup(gv_);
    } else {
      out = mod_->Lookup(gv_).as<FunctionNode>()->body;
    }
    return out;
  }

  std::vector<Expr> PrepareArgs(const CallNode* call_node) {
    std::vector<Expr> args;
    for (auto arg : call_node->args) {
      if (arg.as<ConstantNode>()) {
        args.emplace_back(arg);
      } else {
        args.emplace_back(PrepareInput(arg));
      }
    }
    return args;
  }

 private:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (op_map_.count(call_node->op)) {
        auto out = op_map_[call_node->op](call_node);
        if (out.defined()) {
          return out;
        }
      }
    }
    return post;
  }

  Expr DispatchVisitExpr(const Expr& expr) override {
    auto post = MixedModeMutator::DispatchVisitExpr(expr);
    if (auto op = post.as<FunctionNode>()) {
      return Function(op->params, op->body, NullValue<Type>(), op->type_params, op->attrs);
    }
    return post;
  }

  std::unordered_map<Expr, std::function<Expr(const CallNode*)>, ObjectPtrHash, ObjectPtrEqual>
      op_map_;
  IRModule mod_;
  Function func_;
  GlobalVar gv_;
};

Expr DynamicToStatic(Function f, IRModule m) {
  DynamicToStaticMutator mutator(m, f);
  Expr expr = mutator.Mutate(f);
  Expr out = mutator.PrepareInput(expr);
  return out;
}

namespace transform {

Pass DynamicToStatic() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DynamicToStatic(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "DynamicToStatic", {});
}

TVM_REGISTER_GLOBAL("relay._transform.DynamicToStatic").set_body_typed([]() {
  return DynamicToStatic();
});

}  // namespace transform
}  // namespace relay
}  // namespace tvm
