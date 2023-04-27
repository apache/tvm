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
 * \file pad.cc
 * \brief Implementation of operator pad
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/op.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>

#include <vector>

#include "../make_op.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.nn.pad
TVM_REGISTER_NODE_TYPE(PadAttrs);

InferCorrectLayoutOutput PadInferCorrectLayout(const Attrs& attrs,
                                               const Array<Layout>& new_in_layouts,
                                               const Array<Layout>& old_in_layouts,
                                               const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<PadAttrs>();
  CHECK(attrs_ptr);
  ObjectPtr<PadAttrs> params = make_object<PadAttrs>(*attrs_ptr);

  Layout ret_data;
  // If new_in_layouts are defined, this code tries to modify the layout.
  bool is_layout_modified = new_in_layouts.defined();
  if (new_in_layouts.defined()) {
    // Create a map of axis to param_width. For the new layout, a new param_width is generated using
    // the map. The new layout is rejected, if the padding is happening along the axis which was
    // split.

    // 1) Create a map from axis to param_width using old layout.
    std::map<std::string, tvm::Array<Integer>> axis_pad_width;
    int index_counter = 0;
    ICHECK_EQ(new_in_layouts.size(), 2);
    ICHECK_EQ(old_in_layouts.size(), 2);
    for (auto iter_var : old_in_layouts[0]->axes) {
      const auto& old_layout_axis = LayoutAxis::Get(iter_var);
      axis_pad_width.emplace(old_layout_axis.name(), params->pad_width[index_counter]);
      index_counter++;
    }

    // 2) Create new pad width by walking over the new layout and using the map.
    tvm::Array<tvm::Array<Integer>> new_pad_width;
    for (auto iter_var : new_in_layouts[0]->axes) {
      const auto& new_layout_axis = LayoutAxis::Get(iter_var);
      auto axis_name = new_layout_axis.name();
      if (axis_pad_width.count(axis_name) != 0 && new_layout_axis.IsPrimal()) {
        // This is primal axis. So, directly use the original pad_width.
        new_pad_width.push_back(axis_pad_width.at(axis_name));
      } else {
        // This is the axis that got split. So, check that pad_width was [0, 0] originally.
        const auto& dual_axis = new_layout_axis.ToPrimal();
        auto dual_axis_name = dual_axis.name();
        ICHECK(axis_pad_width.count(dual_axis_name))
            << "Missing axis " << dual_axis << " in " << old_in_layouts[0].name();
        new_pad_width.push_back(axis_pad_width.at(dual_axis_name));

        // If any pad_width element is not zero, do not change the layout.
        for (auto width : axis_pad_width.at(dual_axis_name)) {
          if (auto* width_imm = width.as<IntImmNode>()) {
            if (width_imm->value != 0) {
              is_layout_modified = false;
            }
          } else {
            is_layout_modified = false;
          }
        }
      }
    }

    // If the above conditions satisfied, we can set the newly created pad_width and use the new
    // layout.
    if (is_layout_modified) {
      ret_data = new_in_layouts[0];
      params->pad_width = new_pad_width;
    }
  }

  if (!is_layout_modified) {
    if (old_in_layouts.defined()) {
      ICHECK_EQ(old_in_layouts.size(), 2);
      ret_data = old_in_layouts[0];
    } else {
      ret_data = Layout::Undef();
    }
  }

  // The pad value is always a scalar
  Layout ret_pad_value = Layout("1");
  return InferCorrectLayoutOutput({ret_data, ret_pad_value}, {ret_data}, Attrs(params));
}

bool PadRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
            const TypeReporter& reporter) {
  // types = [pad_data_type, pad_value_type, ret_type]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const PadAttrs* param = attrs.as<PadAttrs>();
  ICHECK(param != nullptr);

  // check that pad widths match lengths
  ICHECK(data->shape.size() == param->pad_width.size())
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->shape.size() << " dimensions "
      << "and there are " << param->pad_width.size() << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->pad_width.size(); i++) {
    ICHECK(param->pad_width[i].size() == 2)
        << "Each pad width element should be a pair but at index " << i << " there are "
        << param->pad_width[i].size() << " elements.";

    auto width1 = tir::as_const_int(param->pad_width[i][0]);
    auto width2 = tir::as_const_int(param->pad_width[i][1]);
    ICHECK(width1 != nullptr);
    ICHECK(width2 != nullptr);

    if (!data->shape[i].as<tir::AnyNode>()) {
      auto padding = tir::make_const(data->shape[i].dtype(), *width1 + *width2);
      oshape.push_back(data->shape[i] + padding);
      if (tir::as_const_int(data->shape[i])) {
        ICHECK(topi::detail::GetConstInt(data->shape[i] + padding) >= 0)
            << "Output shape post padding should be positive but got " << data->shape[i] + padding;
      }
    } else {
      oshape.push_back(data->shape[i]);
    }
  }

  reporter->Assign(types[2], TensorType(Array<IndexExpr>(oshape), data->dtype));
  return true;
}

Array<te::Tensor> PadCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  const auto* param = attrs.as<PadAttrs>();
  ICHECK(param != nullptr);

  auto pad_width = param->pad_width;
  ICHECK(pad_width.size() == inputs[0].ndim() && pad_width[0].size() == 2) << "Illegal pad_width";
  Array<IndexExpr> pad_before;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_before.push_back(pad_width[i][0]);
  }
  Array<IndexExpr> pad_after;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_after.push_back(pad_width[i][1]);
  }
  te::Tensor cast_pad_value = topi::cast(inputs[1], inputs[0]->dtype);
  const PrimExpr& pad_value = cast_pad_value(Array<PrimExpr>(inputs[1]->shape.size(), 0));
  return Array<te::Tensor>{topi::pad(inputs[0], pad_before, pad_after, pad_value, "T_pad",
                                     topi::kElementWise, param->pad_mode)};
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakePad(Expr data, Array<Array<Integer>> pad_width, Expr pad_value, String pad_mode) {
  auto attrs = make_object<PadAttrs>();
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  static const Op& op = Op::Get("nn.pad");
  return Call(op, {data, pad_value}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.pad").set_body_typed(MakePad);

RELAY_REGISTER_OP("nn.pad")
    .describe(R"code(Pad for n-D tensor.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<PadAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("pad_val", "Tensor", "The value to fill the padded area with")
    .set_support_level(2)
    .add_type_rel("Pad", PadRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PadInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FTVMCompute>("FTVMCompute", PadCompute);

// relay.nn.mirror_pad
TVM_REGISTER_NODE_TYPE(MirrorPadAttrs);

bool MirrorPadRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const MirrorPadAttrs* param = attrs.as<MirrorPadAttrs>();
  ICHECK(param != nullptr);

  // check that pad widths match lengths
  ICHECK(data->shape.size() == param->pad_width.size())
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->shape.size() << " dimensions "
      << "and there are " << param->pad_width.size() << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->pad_width.size(); i++) {
    ICHECK(param->pad_width[i].size() == 2)
        << "Each pad width element should be a pair but at index " << i << " there are "
        << param->pad_width[i].size() << " elements.";

    auto width1 = tir::as_const_int(param->pad_width[i][0]);
    auto width2 = tir::as_const_int(param->pad_width[i][1]);
    ICHECK(width1 != nullptr);
    ICHECK(width2 != nullptr);

    ICHECK(*width1 >= 0) << "Param width elements should be positive but first pad width at "
                         << "index " << i << " is " << *width1 << ".";
    ICHECK(*width2 >= 0) << "Param width elements should be positive but first pad width at "
                         << "index " << i << " is " << *width2 << ".";

    auto padding = tir::make_const(data->shape[i].dtype(), *width1 + *width2);
    oshape.push_back(data->shape[i] + padding);
  }

  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), data->dtype));
  return true;
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakeMirrorPad(Expr data, Array<Array<IndexExpr>> pad_width, String mode) {
  auto attrs = make_object<MirrorPadAttrs>();
  attrs->mode = mode;
  attrs->pad_width = std::move(pad_width);
  static const Op& op = Op::Get("nn.mirror_pad");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.mirror_pad").set_body_typed(MakeMirrorPad);

RELAY_REGISTER_OP("nn.mirror_pad")
    .describe(R"code(MirrorPad for n-D tensor.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<MirrorPadAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("MirrorPad", MirrorPadRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
