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

#include "capabilities.h"
#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

std::unique_ptr<sl::CompiledNetwork> EthosnAPI::Compile(std::shared_ptr<sl::Network> network,
                                                        const sl::CompilationOptions& options) {
  std::vector<std::unique_ptr<sl::CompiledNetwork>> compiled_network =
      sl::Compile(*network, options);
  CHECK_GE(compiled_network.size(), 1) << "Ethos-N compiler failed to compile network";

  return std::move(compiled_network[0]);
}

struct EthosnCompilerConfigNode : public tvm::AttrsNode<EthosnCompilerConfigNode> {
  int variant;
  bool strategy0;
  bool strategy1;
  bool strategy3;
  bool strategy4;
  bool strategy6;
  bool strategy7;
  bool dump_ram;
  bool initial_sram_dump;
  bool block_config_16x16;
  bool block_config_32x8;
  bool block_config_8x32;
  bool block_config_8x8;
  bool enable_intermediate_compression;
  bool disable_winograd;
  bool dump_debug_files;
  String debug_dir;
  bool enable_cascading;

  TVM_DECLARE_ATTRS(EthosnCompilerConfigNode, "ext.attrs.EthosnCompilerConfigNode") {
    TVM_ATTR_FIELD(variant)
        .describe("0 for Ethos-N77, 1 for Ethos-N57, 2 for Ethos-N37. See Ethos-N documentation.")
        .set_default(0);
    TVM_ATTR_FIELD(strategy0).set_default(true);
    TVM_ATTR_FIELD(strategy1).set_default(true);
    TVM_ATTR_FIELD(strategy3).set_default(true);
    TVM_ATTR_FIELD(strategy4).set_default(true);
    TVM_ATTR_FIELD(strategy6).set_default(true);
    TVM_ATTR_FIELD(strategy7).set_default(true);
    TVM_ATTR_FIELD(dump_ram).set_default(false);
    TVM_ATTR_FIELD(initial_sram_dump).set_default(false);
    TVM_ATTR_FIELD(block_config_16x16).set_default(true);
    TVM_ATTR_FIELD(block_config_32x8).set_default(true);
    TVM_ATTR_FIELD(block_config_8x32).set_default(true);
    TVM_ATTR_FIELD(block_config_8x8).set_default(true);
    TVM_ATTR_FIELD(enable_intermediate_compression).set_default(true);
    TVM_ATTR_FIELD(disable_winograd).set_default(false);
    TVM_ATTR_FIELD(dump_debug_files).set_default(false);
    TVM_ATTR_FIELD(debug_dir).set_default(".");
    TVM_ATTR_FIELD(enable_cascading).set_default(false);
  }
};

class EthosnCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(EthosnCompilerConfig, Attrs, EthosnCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(EthosnCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.ethos-n.options", EthosnCompilerConfig);

sl::CompilationOptions EthosnAPI::CreateOptions() {
  auto ctx = transform::PassContext::Current();
  auto cfg = ctx->GetConfig<EthosnCompilerConfig>("relay.ext.ethos-n.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<EthosnCompilerConfig>();
  }

  sl::CompilationOptions options(variants[cfg.value()->variant]);
  options.m_Strategy0 = cfg.value()->strategy0;
  options.m_Strategy1 = cfg.value()->strategy1;
  options.m_Strategy3 = cfg.value()->strategy3;
  options.m_Strategy4 = cfg.value()->strategy4;
  options.m_Strategy6 = cfg.value()->strategy6;
  options.m_Strategy7 = cfg.value()->strategy7;
  options.m_DebugInfo.m_DumpRam = cfg.value()->dump_ram;
  options.m_DebugInfo.m_InitialSramDump = cfg.value()->initial_sram_dump;
  options.m_BlockConfig16x16 = cfg.value()->block_config_16x16;
  options.m_BlockConfig32x8 = cfg.value()->block_config_32x8;
  options.m_BlockConfig8x32 = cfg.value()->block_config_8x32;
  options.m_BlockConfig8x8 = cfg.value()->block_config_8x8;
  options.m_EnableIntermediateCompression = cfg.value()->enable_intermediate_compression;
  options.m_DisableWinograd = cfg.value()->disable_winograd;
  options.m_DebugInfo.m_DumpDebugFiles = cfg.value()->dump_debug_files;
  options.m_DebugInfo.m_DebugDir = cfg.value()->debug_dir;
  options.m_EnableCascading = cfg.value()->enable_cascading;
  return options;
}

bool EthosnAPI::IsEthosnOp(const Call& call, const std::string& op_name) {
  if (call->op->IsInstance<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    CHECK(op.defined());
    return op == Op::Get(op_name);
  } else {
    return false;
  }
}

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
