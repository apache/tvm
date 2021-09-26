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
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pattern_utils.h"
#include "transform_layout.h"

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
   * \param desired_layouts Specify mapping of op_name to array of desired layouts for each input.
   *                        For example: Map("nn.conv2d", Array("NHWC", "OHWI")),
   *                        this specifies the desired layout for data then kernel for nn.conv2d.
   */
  explicit ConvertTransformMemorizerNode(Map<String, Array<String>> desired_layouts)
      : desired_layouts_(std::move(desired_layouts)) {}

  /*!
   * \brief Defines the call transformation for ConvertLayout pass. The new layouts should be the
   * desired layout as specified by the user.
   * \param ref_call The original call.
   * \param new_attrs Updated attributes consistent with new layouts.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  Call CallWithNewLayouts(const Call& ref_call, Attrs new_attrs,
                          const std::vector<Expr>& new_args) override {
    static auto fconvert_layout = Op::GetAttrMap<FTVMConvertOpLayout>("FTVMConvertOpLayout");
    Op op = Downcast<Op>(ref_call->op);
    Expr new_e;
    bool modified = false;
    if (fconvert_layout.count(op)) {
      auto desired_layouts = desired_layouts_;
      if (desired_layouts.find(op->name) != desired_layouts.end()) {
        tvm::Array<tvm::te::Tensor> tinfos;
        for (auto& expr : ref_call->args) {
          if (expr->checked_type()->IsInstance<TupleTypeNode>()) {
            auto tuple_ttype_node = expr->type_as<TupleTypeNode>();
            for (auto& ttype : tuple_ttype_node->fields) {
              auto ttype_node = ttype.as<TensorTypeNode>();
              tinfos.push_back(tvm::te::placeholder(ttype_node->shape, ttype_node->dtype));
            }
          } else {
            auto ttype = expr->type_as<TensorTypeNode>();
            tinfos.push_back(tvm::te::placeholder(ttype->shape, ttype->dtype));
          }
        }

        Array<String> op_desired_layouts = desired_layouts.at(op->name);
        Expr altered_value = fconvert_layout[op](new_attrs, new_args, tinfos, op_desired_layouts);
        if (altered_value.defined()) {
          new_e = altered_value;
          modified = true;
        }
      } else {
        LOG(WARNING) << "Desired layout(s) not specified for op: " << op->name;
      }
    }
    if (!modified) {
      new_e = Call(ref_call->op, new_args, new_attrs);
    }

    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call) << "Can only replace the original operator with another call node";
    return Call(new_call->op, new_call->args, new_call->attrs, new_call->type_args, ref_call->span);
  }

  Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) override {
    return CallWithNewLayouts(ref_call, ref_call->attrs, new_args);
  }

  /*! \brief A mapping of op_name to array of desired layouts for each input. */
  Map<String, Array<String>> desired_layouts_;
};

/*!
 * \brief Container that provides the transformation function for convert layout.
 */
class ConvertTransformMemorizer : public TransformMemorizer {
 public:
  ConvertTransformMemorizer() = default;
  explicit ConvertTransformMemorizer(ObjectPtr<Object> n) : TransformMemorizer(n) {}

  ConvertTransformMemorizerNode* operator->() {
    return static_cast<ConvertTransformMemorizerNode*>(get_mutable());
  }

  using ContainerType = ConvertTransformMemorizerNode;
};

/*!
 * Limitations:
 * 1. The altered op should have the same number of arguments as the previous one.
 * 2. Do not support nested tuple arguments.
 */
Expr ConvertLayout(const Expr& expr, const Map<String, Array<String>>& desired_layouts) {
  ConvertTransformMemorizer transformMemorizer(
      make_object<ConvertTransformMemorizerNode>(desired_layouts));
  auto fcontext = [&](const Call& call) -> ObjectRef { return transformMemorizer; };

  return ForwardRewrite(expr, LayoutRewriter<ConvertTransformMemorizer>, fcontext);
}

}  // namespace convert_op_layout

namespace transform {

Pass ConvertLayout(const Map<String, Array<String>>& desired_layouts) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::convert_op_layout::ConvertLayout(f, desired_layouts));
      };
  return CreateFunctionPass(pass_func, 3, "ConvertLayout", {"InferType", "CanonicalizeOps"});
}

TVM_REGISTER_GLOBAL("relay._transform.ConvertLayout").set_body_typed(ConvertLayout);

TVM_REGISTER_GLOBAL("relay._transform.InferCorrectLayoutOutput")
    .set_body_typed([](Array<Layout> input_layouts, Array<Layout> output_layouts, Attrs new_attrs) {
      return InferCorrectLayoutOutput(input_layouts, output_layouts, new_attrs);
    });

TVM_REGISTER_NODE_TYPE(InferCorrectLayoutOutputNode);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
