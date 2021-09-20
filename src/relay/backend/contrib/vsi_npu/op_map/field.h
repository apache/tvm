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
#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_FIELD_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_FIELD_H_

#include "helper.h"
#include "tim/vx/tensor.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {
void shape_setup(const Call& c, uint32_t arg_idx, tim::vx::ShapeType& result_shape);

template <uint32_t Idx, uint32_t Scale_Idx, uint32_t Zp_Idx,
          tim::vx::QuantType QType = tim::vx::QuantType::ASYMMETRIC>
struct Field_Quant_Operand {
  static const uint32_t arg_pos = Idx;

  static tim::vx::TensorSpec AsTimVxTensorSpec(const Call& c, const Call& c1) {
    tim::vx::ShapeType shape;
    tim::vx::DataType dataType;
    tim::vx::TensorAttribute role;
    std::vector<float> scales;
    std::vector<int32_t> zps;

    Expr expr = c->args[Idx];
    auto dtype = expr->checked_type().as<TensorTypeNode>()->dtype;
    role = expr->IsInstance<ConstantNode>() ? tim::vx::TensorAttribute::CONSTANT
                                            : tim::vx::TensorAttribute::TRANSIENT;

    dataType = GetTvxType(dtype);

    shape_setup(c, Idx, shape);
    AsConstant(c1->args[Scale_Idx], scales);
    AsConstant(c1->args[Zp_Idx], zps);
    tim::vx::Quantization quant_spec;
    if (scales.size() == 1) {
      quant_spec = tim::vx::Quantization(QType, scales[0], zps[0]);
    } else {
      for (uint32_t i = 1; i < scales.size(); i++) {
        zps.push_back(0);
      }
      quant_spec = tim::vx::Quantization(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0, scales, zps);
    }

    tim::vx::TensorSpec spec(dataType, shape, role, quant_spec);
    return spec;
  }
};

template <uint32_t Idx>
struct Field_NoQuant_Operand {
  static const uint32_t arg_pos = Idx;

  static tim::vx::TensorSpec AsTimVxTensorSpec(const Call& c) {
    tim::vx::ShapeType shape;
    tim::vx::DataType dataType;
    tim::vx::TensorAttribute role;

    Expr expr = c->args[Idx];

    auto dtype = expr->checked_type().as<TensorTypeNode>()->dtype;
    role = expr->IsInstance<ConstantNode>() ? tim::vx::TensorAttribute::CONSTANT
                                            : tim::vx::TensorAttribute::TRANSIENT;

    dataType = GetTvxType(dtype);

    shape_setup(c, Idx, shape);
    tim::vx::TensorSpec spec(dataType, shape, role);
    return spec;
  }
};

template <uint32_t Idx, uint32_t Scale_Idx, uint32_t Zp_Idx,
          tim::vx::QuantType QType = tim::vx::QuantType::ASYMMETRIC>
struct Field_TUPLE_QUANT_OPERAND {
  static const uint32_t arg_pos = Idx;

  static std::vector<tim::vx::TensorSpec> AsTimVxTensorSpec(const Call& c, const Call& c1) {
    auto input_node_tensors_type = c->args[Idx]->checked_type().as<TupleTypeNode>()->fields;
    auto input_node_scales = c->args[Scale_Idx].as<TupleNode>()->fields;
    auto input_node_zps = c->args[Zp_Idx].as<TupleNode>()->fields;

    std::vector<tim::vx::TensorSpec> specs;
    uint32_t input_node_num = input_node_tensors_type.size();
    for (uint32_t i = 0; i < input_node_num; i++) {
      tim::vx::ShapeType shape;
      tim::vx::DataType dataType;
      std::vector<float> scales;
      std::vector<int32_t> zps;

      std::transform(
          input_node_tensors_type[i].as<TensorTypeNode>()->shape.rbegin(),
          input_node_tensors_type[i].as<TensorTypeNode>()->shape.rend(), std::back_inserter(shape),
          [](const PrimExpr& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });

      AsConstant<float>(input_node_scales[i], scales);
      AsConstant<int>(input_node_zps[i], zps);

      auto dtype = input_node_tensors_type[i].as<TensorTypeNode>()->dtype;
      dataType = GetTvxType(dtype);

      auto quant_spec = tim::vx::Quantization(QType, scales[0], zps[0]);
      tim::vx::TensorSpec spec(dataType, shape, tim::vx::TensorAttribute::TRANSIENT, quant_spec);
      specs.push_back(spec);
    }
    return specs;
  }
};

}  // namespace op_map
}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif