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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "pattern_util.h"

namespace tvm {
namespace relay {

class DynamicToStaticMutator : public MixedModeMutator {
 public:
<<<<<<< HEAD
  DynamicToStaticMutator() {}
=======
  DynamicToStaticMutator()
      : dyn_reshape_op_(Op::Get("dyn.reshape")),
        dyn_tile_op_(Op::Get("dyn.tile")),
        dyn_topk_op_(Op::Get("dyn.topk")),
        dyn_broadcast_to_op_(Op::Get("dyn.broadcast_to")) {}
>>>>>>> bdc05de53... Dynamic BroadcastTo

 private:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    const CallNode* call_node = post.as<CallNode>();
    if (call_node->op == Op::Get("dyn.reshape")) {
      if (const ConstantNode* shape = call_node->args[1].as<ConstantNode>()) {
        CHECK_EQ(shape->data->ndim, 1);
        return MakeReshape(call_node->args[0], ToVector(shape->data));
      }
    } else if (call_node->op == Op::Get("dyn.tile")) {
      if (const ConstantNode* reps = call_node->args[1].as<ConstantNode>()) {
        CHECK_EQ(reps->data->ndim, 1);
        return MakeTile(call_node->args[0], ToVector(reps->data));
      }
    } else if (call_node->op == Op::Get("dyn.topk")) {
      if (const ConstantNode* k = call_node->args[1].as<ConstantNode>()) {
        const TopKAttrs* param = call_node->attrs.as<TopKAttrs>();
        CHECK(param);
        return MakeTopK(call_node->args[0], static_cast<int>(ToScalar(k->data, 0)), param->axis,
                        param->ret_type, param->is_ascend, param->dtype);
      }
    }
    if (call_node->op == dyn_broadcast_to_op_) {
      if (const ConstantNode* shape = call_node->args[1].as<ConstantNode>()) {
        auto attrs = make_object<InitOpAttrs>();
        CHECK_EQ(shape->data->ndim, 1);

        // put shape in attrs
        attrs->shape = ToVector(shape->data);
        static const Op& broadcast_to = Op::Get("broadcast_to");
        // pass in one arg to static broadcast to
        return Call(broadcast_to, {call_node->args[0]}, Attrs(attrs), {});
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
<<<<<<< HEAD
=======

  const Op& dyn_reshape_op_;
  const Op& dyn_tile_op_;
  const Op& dyn_topk_op_;
  const Op& dyn_broadcast_to_op_; 
>>>>>>> bdc05de53... Dynamic BroadcastTo
};

Expr DynamicToStatic(Function f, IRModule m) {
  Expr pre = f;
  Expr expr = f;
  auto fold_const = transform::FoldConstant();
  auto infer_type = transform::InferType();
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
    expr = DynamicToStaticMutator().Mutate(m->functions[gv]);
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
