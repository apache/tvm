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

#include "pattern_util.h"

namespace tvm {
namespace relay {

class DynamicToStaticMutator : public MixedModeMutator {
 public:
  DynamicToStaticMutator() {
    op_map_ = {
        {Op::Get("dyn.reshape"),
         [](const CallNode* call_node) {
           if (const ConstantNode* shape = call_node->args[1].as<ConstantNode>()) {
             CHECK_EQ(shape->data->ndim, 1);
             return MakeReshape(call_node->args[0], ToVector(shape->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.tile"),
         [](const CallNode* call_node) {
           if (const ConstantNode* reps = call_node->args[1].as<ConstantNode>()) {
             CHECK_EQ(reps->data->ndim, 1);
             return MakeTile(call_node->args[0], ToVector(reps->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.topk"),
         [](const CallNode* call_node) {
           if (const ConstantNode* k = call_node->args[1].as<ConstantNode>()) {
             const TopKAttrs* param = call_node->attrs.as<TopKAttrs>();
             CHECK(param);
             return MakeTopK(call_node->args[0], static_cast<int>(ToScalar(k->data, 0)),
                             param->axis, param->ret_type, param->is_ascend, param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.broadcast_to"),
         [](const CallNode* call_node) {
           if (const ConstantNode* shape = call_node->args[1].as<ConstantNode>()) {
             CHECK_EQ(shape->data->ndim, 1);
             return MakeBroadCastTo(call_node->args[0], ToVector(shape->data));
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.zeros"),
         [](const CallNode* call_node) {
           if (const ConstantNode* shape = call_node->args[0].as<ConstantNode>()) {
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             CHECK(param);
             return MakeZeros(ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.ones"),
         [](const CallNode* call_node) {
           if (const ConstantNode* shape = call_node->args[0].as<ConstantNode>()) {
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             CHECK(param);
             return MakeOnes(ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.one_hot"),
         [](const CallNode* call_node) {
           if (const ConstantNode* depth = call_node->args[3].as<ConstantNode>()) {
             const OneHotAttrs* param = call_node->attrs.as<OneHotAttrs>();
             CHECK(param);
             return MakeOneHot(call_node->args[0], call_node->args[1], call_node->args[2],
                               static_cast<int>(ToScalar(depth->data, 0)), param->axis,
                               param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.image.resize"),
         [](const CallNode* call_node) {
           if (const ConstantNode* size = call_node->args[1].as<ConstantNode>()) {
             const ResizeAttrs* param = call_node->attrs.as<ResizeAttrs>();
             CHECK(param);
             auto size_int = ToVector(size->data);
             Array<PrimExpr> size_prim;
             for (size_t i = 0; i < size_int.size(); ++i) {
               size_prim.push_back(size_int[i]);
             }
             return MakeResize(call_node->args[0], size_prim, param->layout, param->method,
                               param->coordinate_transformation_mode, param->out_dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.full"),
         [](const CallNode* call_node) {
           if (const ConstantNode* shape = call_node->args[1].as<ConstantNode>()) {
             CHECK_EQ(shape->data->ndim, 1);
             const InitOpAttrs* param = call_node->attrs.as<InitOpAttrs>();
             CHECK(param);
             return MakeFull(call_node->args[0], ToVector(shape->data), param->dtype);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.upsampling"),
         [](const CallNode* call_node) {
           const ConstantNode* scale_h = call_node->args[1].as<ConstantNode>();
           const ConstantNode* scale_w = call_node->args[2].as<ConstantNode>();
           if (scale_h && scale_w) {
             CHECK_EQ(scale_h->data->ndim, 0);
             CHECK_EQ(scale_w->data->ndim, 0);
             const UpSamplingAttrs* param = call_node->attrs.as<UpSamplingAttrs>();
             CHECK(param);
             return MakeUpSampling(call_node->args[0], ToScalar(scale_h->data),
                                   ToScalar(scale_w->data), param->layout, param->method,
                                   param->align_corners);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.upsampling3d"),
         [](const CallNode* call_node) {
           const ConstantNode* scale_d = call_node->args[1].as<ConstantNode>();
           const ConstantNode* scale_h = call_node->args[2].as<ConstantNode>();
           const ConstantNode* scale_w = call_node->args[3].as<ConstantNode>();
           if (scale_d && scale_h && scale_w) {
             CHECK_EQ(scale_d->data->ndim, 0);
             CHECK_EQ(scale_h->data->ndim, 0);
             CHECK_EQ(scale_w->data->ndim, 0);
             const UpSampling3DAttrs* param = call_node->attrs.as<UpSampling3DAttrs>();
             CHECK(param);

             return MakeUpSampling3D(call_node->args[0], ToScalar(scale_d->data),
                                     ToScalar(scale_h->data), ToScalar(scale_w->data),
                                     param->layout, param->method,
                                     param->coordinate_transformation_mode);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.nn.pad"),
         [](const CallNode* call_node) {
           const ConstantNode* pad_width = call_node->args[1].as<ConstantNode>();
           const ConstantNode* pad_fill = call_node->args[2].as<ConstantNode>();
           if (pad_width && pad_fill) {
             CHECK_EQ(pad_fill->data->ndim, 0);   // pad_val is 1d
             CHECK_EQ(pad_width->data->ndim, 2);  // pad_width is 2d

             const PadAttrs* param = call_node->attrs.as<PadAttrs>();
             CHECK(param);
             return MakePad(call_node->args[0], ToMatrix(pad_width->data), ToScalar(pad_fill->data),
                            param->pad_mode);
           }
           return Expr(nullptr);
         }},
        {Op::Get("dyn.strided_slice"),
         [](const CallNode* call_node) {
           const ConstantNode* begin = call_node->args[1].as<ConstantNode>();
           const ConstantNode* end = call_node->args[2].as<ConstantNode>();
           const ConstantNode* stride = call_node->args[3].as<ConstantNode>();
           if (begin && end && stride) {
             CHECK_EQ(begin->data->ndim, 1);
             CHECK_EQ(end->data->ndim, 1);
             CHECK_EQ(stride->data->ndim, 1);
             const StridedSliceAttrs* param = call_node->attrs.as<StridedSliceAttrs>();
             CHECK(param);
             return MakeStridedSlice(call_node->args[0], ToVector(begin->data), ToVector(end->data),
                                     ToVector(stride->data), param->slice_mode);
           }
           return Expr(nullptr);
         }},
    };
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
};

Expr DynamicToStatic(Function f, IRModule m) {
  Expr pre = f;
  Expr expr = f;
  auto fold_const = transform::FoldConstant();
  auto infer_type = transform::InferType();
  DynamicToStaticMutator mutator;
  Map<BaseFunc, GlobalVar> vars;
  for (auto kv : m->functions) {
    vars.Set(kv.second, kv.first);
  }
  const auto gv = vars[f];
  int i = 0;
  do {
    pre = expr;
    // TODO(mbrookhart): Is it possible to run these passes JUST on the current function?
    m = infer_type(m);
    m = fold_const(m);
    expr = mutator.Mutate(m->functions[gv]);
    m->Update(gv, Downcast<BaseFunc>(expr));
    i += 1;
  } while (pre != expr && i < 1000);
  return expr;
}

namespace transform {

Pass ConvertDynamicToStatic() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DynamicToStatic(f, m));
      };
  return CreateFunctionPass(pass_func, 3, "DynamicToStatic", {});
}

TVM_REGISTER_GLOBAL("relay._transform.DynamicToStatic").set_body_typed([]() {
  return ConvertDynamicToStatic();
});

}  // namespace transform
}  // namespace relay
}  // namespace tvm
