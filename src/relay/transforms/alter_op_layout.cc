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
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>
#include <tuple>
#include <vector>
#include <functional>
#include <string>
#include <utility>
#include <unordered_map>

#include "transform_layout.h"
#include "pattern_util.h"

namespace tvm {
namespace relay {

namespace alter_op_layout {

/*!
 * \brief Container to instantiate a Node for alter op layouts.
 */
class AlterTransformMemorizerNode : public TransformMemorizerNode {
 public:
  static constexpr const char* _type_key = "relay.alter_op_layout.AlterTransformMemorizerNode";
};

/*!
 * \brief Container that provides the transformation function for alter layout..
 */
class AlterTransformMemorizer : public TransformMemorizer {
 public:
  AlterTransformMemorizer() {}
  explicit AlterTransformMemorizer(ObjectPtr<Object> n) : TransformMemorizer(n) {}

  AlterTransformMemorizerNode* operator->() {
    return static_cast<AlterTransformMemorizerNode*>(get_mutable());
  }

  /*!
   * \brief Defines the call transformation for AlterOpLayout pass. The new layouts are defined by
   * used for different targets using a packed func.
   * \param ref_call The original call.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) override {
    static auto falter_layout = Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");
    Op op = Downcast<Op>(ref_call->op);

    Expr new_e;
    bool modified = false;
    if (falter_layout.count(op)) {
      tvm::Array<tvm::te::Tensor> tinfos;
      for (auto expr : ref_call->args) {
        auto ttype = expr->type_as<TensorTypeNode>();
        tinfos.push_back(tvm::te::placeholder(ttype->shape, ttype->dtype));
      }
      // TODO(@kevinthesun, @icemelon9): This won't work if inputs/outputs are dynamic shapes.
      //   Probably we need to disable the AlterOpLayout when compiling dynamic models.
      Expr altered_value = falter_layout[op](ref_call->attrs, new_args, tinfos,
                                             ref_call->checked_type());
      if (altered_value.defined()) {
        new_e = altered_value;
        modified = true;
      }
    }
    if (!modified) {
      new_e = Call(ref_call->op, new_args, ref_call->attrs);
    }

    const CallNode* new_call = new_e.as<CallNode>();
    CHECK(new_call) << "Can only replace the original operator with another call node";
    return GetRef<Call>(new_call);
  }

  using ContainerType = AlterTransformMemorizerNode;
};

/*!
 * Limitations:
 * 1. The altered op should have the same number of arguments as the previous one.
 * 2. Do not support nested tuple arguments.
 */
Expr AlterOpLayout(const Expr& expr) {
  AlterTransformMemorizer alterMemorizer(make_object<AlterTransformMemorizerNode>());
  auto fcontext = [&](const Call& call) -> ObjectRef { return alterMemorizer; };

  return ForwardRewrite(expr, LayoutRewriter<AlterTransformMemorizer>, fcontext);
}

}  // namespace alter_op_layout

namespace transform {

Pass AlterOpLayout() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(relay::alter_op_layout::AlterOpLayout(f));
  };
  return CreateFunctionPass(pass_func, 3, "AlterOpLayout", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AlterOpLayout")
.set_body_typed(AlterOpLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
