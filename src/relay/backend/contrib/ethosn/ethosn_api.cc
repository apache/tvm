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

#include "ethosn_api.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

EthosnError EthosnAPI::Concatenate(const Expr& expr, ConcatenateParams* params) {
  Call call = Downcast<Call>(expr);
  const auto& attrs = call->attrs.as<ConcatenateAttrs>();
  params->concat_info.m_Axis = attrs->axis;

  float output_s;
  int output_zp;
  EthosnError err = AsConstant<float>(call->args[3], &output_s);
  err += AsConstant<int>(call->args[4], &output_zp);
  params->concat_info.m_OutputQuantizationInfo = sl::QuantizationInfo(output_zp, output_s);

  auto input_scales = call->args[1].as<TupleNode>()->fields;
  auto input_zero_points = call->args[2].as<TupleNode>()->fields;
  auto input_tensors = call->args[0]->checked_type().as<TupleTypeNode>()->fields;

  int index = 0;
  for (auto input_scale : input_scales) {
    auto input_dtype = input_tensors[index].as<TensorTypeNode>();
    auto input_zero_point = input_zero_points[index];
    float scale;
    int zp;
    err += AsConstant<float>(input_scale, &scale);
    err += AsConstant<int>(input_zero_point, &zp);
    sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
    sl::DataType input_data_type;
    err += Tvm2Npu(input_dtype->shape, &input_tensor_shape);
    err += Tvm2Npu(input_dtype->dtype, &input_data_type);
    params->input_infos.emplace_back(sl::TensorInfo(input_tensor_shape, input_data_type,
                                                    sl::DataFormat::NHWC,
                                                    sl::QuantizationInfo(zp, scale)));
    index++;
  }
  return err;
}

EthosnError EthosnAPI::Split(const Expr& expr, SplitParams* params) {
  Call call = Downcast<Call>(expr);
  const auto* input_tensor_type = call->args[0]->checked_type().as<TensorTypeNode>();
  const auto& attrs = call->attrs.as<SplitAttrs>();

  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_data_type;
  EthosnError err = Tvm2Npu(input_tensor_type->shape, &input_tensor_shape);
  err += Tvm2Npu(input_tensor_type->dtype, &input_data_type);
  params->input_info =
      sl::TensorInfo(input_tensor_shape, input_data_type, params->input_info.m_DataFormat,
                     params->input_info.m_QuantizationInfo);
  params->split_info.m_Axis = attrs->axis;
  if (attrs->indices_or_sections->IsInstance<IntImmNode>()) {
    auto sections = Downcast<IntImm>(attrs->indices_or_sections)->value;
    int size = input_tensor_shape[attrs->axis] / sections;
    for (int i = 0; i < sections; i++) {
      params->split_info.m_Sizes.push_back(size);
    }
  } else {
    auto indices = Downcast<tvm::Array<Integer>>(attrs->indices_or_sections);
    int last_index = 0;
    for (const auto& i : indices) {
      params->split_info.m_Sizes.push_back(i->value - last_index);
      last_index = i->value;
    }
    int axis_size = input_tensor_shape[attrs->axis];
    params->split_info.m_Sizes.push_back(axis_size - last_index);
  }
  return err;
}

EthosnError EthosnAPI::Tvm2Npu(const Array<IndexExpr>& shape, sl::TensorShape* npu_shape) {
  EthosnError err = AsArray<IndexExpr, uint32_t>(shape, npu_shape);
  if (npu_shape->front() != 1) {
    err += EthosnError(ErrStrm() << "batch size=" << npu_shape->front() << ", batch size must = 1");
  }
  return err;
}

EthosnError EthosnAPI::Tvm2Npu(const tvm::DataType& dtype, sl::DataType* data_type) {
  if (dtype.is_scalar() == 1) {
    if (dtype.is_uint() && dtype.bits() == 8) {
      *data_type = sl::DataType::UINT8_QUANTIZED;
      return EthosnError();
    } else if (dtype.is_int() && dtype.bits() == 32) {
      *data_type = sl::DataType::INT32_QUANTIZED;
      return EthosnError();
    }
  }
  return EthosnError(ErrStrm() << "dtype=\'" << dtype << "\', dtype must be either uint8 or int32");
}

// Convert an array of IntImmNodes into ValueT
// IndexT type of Array indexing variable
// ValueT type of resulting value
template <typename IndexT, typename ValueT>
EthosnError EthosnAPI::AsArray(const Array<IndexT>& arr, std::array<ValueT, 4>* v) {
  if (arr.size() > 4)
    return EthosnError(ErrStrm() << "dimensions=" << arr.size() << ", dimensions must be <= 4");
  for (size_t i = 0; i < std::min(arr.size(), 4ul); i++) {
    const PrimExpr& a = arr[i];
    const auto* intImm = a.as<IntImmNode>();
    if (intImm->value > std::numeric_limits<ValueT>::max()) {
      return EthosnError(ErrStrm() << "axis size=" << intImm->value << ", axis size must be <= "
                                   << std::numeric_limits<ValueT>::max());
    }
    (*v)[i] = static_cast<ValueT>(intImm->value);
  }
  return EthosnError();
}

// Get a T from a constant represented by a NDArray.
template <typename T>
EthosnError EthosnAPI::AsConstant(const Expr& expr, T* out) {
  if (!expr->IsInstance<ConstantNode>()) {
    return EthosnError("expected constant data");
  }
  runtime::NDArray data = Downcast<Constant>(expr)->data;
  *out = *static_cast<T*>(data->data);
  return EthosnError();
}

TVM_REGISTER_GLOBAL("relay.ethos-n.support.concatenate")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ConcatenateParams params;
      auto err = EthosnAPI::Concatenate(call, &params);
      *rv = !err && sl::IsConcatenationSupported(params.input_infos, params.concat_info);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.split")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      SplitParams params;
      auto err = EthosnAPI::Split(call, &params);
      *rv = !err && sl::IsSplitSupported(params.input_info, params.split_info);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.query").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
#if defined ETHOSN_HW
  *rv = true;
#else
  *rv = false;
#endif
});

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
