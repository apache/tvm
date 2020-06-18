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
 * \file convert_op_layout.cc
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

namespace convert_op_layout {

/*!
 * \brief Container for the transformations for ConvertLayout.
 */
class ConvertTransformMemorizerNode : public TransformMemorizerNode {
 public:
  /*!
   * \brief Initializes the desired_layout.
   * \param desired_layout The desired layout.
   */
  explicit ConvertTransformMemorizerNode(const std::string& desired_layout)
      : desired_layout_(desired_layout) {}

  /*! \brief The desired layout for the Convert Layout pass */
  std::string desired_layout_;
};

/*!
 * \brief Container that provides the transformation function for convert layout.
 */
class ConvertTransformMemorizer : public TransformMemorizer {
 public:
  ConvertTransformMemorizer() {}
  explicit ConvertTransformMemorizer(ObjectPtr<Object> n) : TransformMemorizer(n) {}

  ConvertTransformMemorizerNode* operator->() {
    return static_cast<ConvertTransformMemorizerNode*>(get_mutable());
  }

  /*!
   * \brief Defines the call transformation for ConvertLayout pass. The new layouts should be the
   * desired layout as specified by the user.
   * \param ref_call The original call.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) override {
    static auto fconvert_layout = Op::GetAttr<FTVMConvertOpLayout>("FTVMConvertOpLayout");
    Op op = Downcast<Op>(ref_call->op);

    Expr new_e;
    bool modified = false;
    if (fconvert_layout.count(op)) {
      tvm::Array<tvm::te::Tensor> tinfos;
      for (auto expr : ref_call->args) {
        auto ttype = expr->type_as<TensorTypeNode>();
        tinfos.push_back(tvm::te::placeholder(ttype->shape, ttype->dtype));
      }
      Expr altered_value =
          fconvert_layout[op](ref_call->attrs, new_args, tinfos, operator->()->desired_layout_);
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

  using ContainerType = ConvertTransformMemorizerNode;
};

/*!
 * Limitations:
 * 1. The altered op should have the same number of arguments as the previous one.
 * 2. Do not support nested tuple arguments.
 */
Expr ConvertLayout(const Expr& expr, const std::string& desired_layout) {
  ConvertTransformMemorizer transformMemorizer(
      make_object<ConvertTransformMemorizerNode>(desired_layout));
  auto fcontext = [&](const Call& call) -> ObjectRef { return transformMemorizer; };

  return ForwardRewrite(expr, LayoutRewriter<ConvertTransformMemorizer>, fcontext);
}

}  // namespace convert_op_layout

namespace transform {

Pass ConvertLayout(const std::string& desired_layout) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::convert_op_layout::ConvertLayout(f, desired_layout));
      };
  return CreateFunctionPass(
      pass_func, 3, "ConvertLayout", {"InferType", "CanonicalizeOps"});
}

TVM_REGISTER_GLOBAL("relay._transform.ConvertLayout").set_body_typed(ConvertLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
