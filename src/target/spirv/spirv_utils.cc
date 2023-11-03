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
 * \file spirv_utils.cc
 * \brief Build SPIRV block
 */
// Use libspirv for parsing and validating code.
#include "spirv_utils.h"

#if TVM_ENABLE_SPIRV
#include <libspirv.h>

#include "codegen_spirv.h"
#endif

#include <tvm/tir/transform.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

#include "../../runtime/spirv/spirv_shader.h"
#include "../../support/utils.h"

namespace tvm {
namespace codegen {

#if TVM_ENABLE_SPIRV

class SPIRVTools {
 public:
  explicit SPIRVTools(Target target) {
    uint32_t vulkan_version =
        target->GetAttr<Integer>("vulkan_api_version").value_or(VK_API_VERSION_1_0).IntValue();
    uint32_t spirv_version =
        target->GetAttr<Integer>("max_spirv_version").value_or(0x10000).IntValue();

    spv_target_env validation_version;
    if (target->kind->name == "opencl") {
      validation_version = SPV_ENV_OPENCL_2_2;
    } else {
      if (vulkan_version >= VK_API_VERSION_1_2) {
        validation_version = SPV_ENV_VULKAN_1_2;
      } else if (vulkan_version >= VK_API_VERSION_1_1 && spirv_version >= 0x10400) {
        validation_version = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
      } else if (vulkan_version >= VK_API_VERSION_1_1) {
        validation_version = SPV_ENV_VULKAN_1_1;
      } else {
        validation_version = SPV_ENV_VULKAN_1_0;
      }
    }
    ctx_ = spvContextCreate(validation_version);
  }

  ~SPIRVTools() { spvContextDestroy(ctx_); }

  std::string BinaryToText(const std::vector<uint32_t>& bin) {
    spv_text text = nullptr;
    spv_diagnostic diagnostic = nullptr;
    spv_const_binary_t spv_bin{bin.data(), bin.size()};

    spv_result_t res =
        spvBinaryToText(ctx_, spv_bin.code, spv_bin.wordCount,
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT,
                        &text, &diagnostic);

    ICHECK_EQ(res, SPV_SUCCESS) << " line=" << diagnostic->position.line
                                << " column=" << diagnostic->position.column
                                << " index=" << diagnostic->position.index
                                << " error:" << diagnostic->error;
    spvDiagnosticDestroy(diagnostic);

    std::string ret(text->str);
    spvTextDestroy(text);
    return ret;
  }

  void ValidateShader(const std::vector<uint32_t>& bin) {
    spv_const_binary_t spv_bin{bin.data(), bin.size()};

    spv_diagnostic diagnostic = nullptr;
    spv_result_t res = spvValidate(ctx_, &spv_bin, &diagnostic);

    ICHECK_EQ(res, SPV_SUCCESS) << " index=" << diagnostic->position.index
                                << " error:" << diagnostic->error;

    spvDiagnosticDestroy(diagnostic);
  }

 private:
  spv_context ctx_;
};

std::pair<std::unordered_map<std::string, runtime::SPIRVShader>, std::string> LowerToSPIRV(
    IRModule mod, Target target) {
  using tvm::runtime::Registry;
  using tvm::runtime::SPIRVShader;

  std::ostringstream code_data;
  SPIRVTools spirv_tools(target);
  std::unordered_map<std::string, SPIRVShader> smap;

  const auto* postproc = Registry::Get("tvm_callback_vulkan_postproc");

  mod = tir::transform::PointerValueTypeRewrite()(std::move(mod));

  CodeGenSPIRV cg(target);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSPIRV: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSPIRV: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined())
        << "CodeGenSPIRV: Expect PrimFunc to have the global_symbol attribute";

    std::string f_name = global_symbol.value();
    std::string entry = f_name;

    SPIRVShader shader = cg.BuildFunction(f, entry);

    if (auto path = std::getenv("TVM_VULKAN_DEBUG_SHADER_SAVEPATH")) {
      if (*path) {
        std::stringstream ss;
        ss << path << "/" << f_name << "_";
        std::string prefix = ss.str();

        std::ofstream(prefix + "tir.txt") << f;
        std::ofstream(prefix + "spv.txt") << spirv_tools.BinaryToText(shader.data);
        std::ofstream(prefix + "spv.spv", std::ios::binary)
            .write(reinterpret_cast<const char*>(shader.data.data()),
                   sizeof(shader.data[0]) * shader.data.size());
      }
    }

    if (!support::BoolEnvironmentVar("TVM_VULKAN_DISABLE_SHADER_VALIDATION")) {
      spirv_tools.ValidateShader(shader.data);
    }

    if (postproc != nullptr) {
      TVMByteArray arr;
      arr.data = reinterpret_cast<const char*>(dmlc::BeginPtr(shader.data));
      arr.size = shader.data.size() * sizeof(uint32_t);
      std::string transformed = (*postproc)(arr, target);
      ICHECK_EQ(transformed.length() % 4U, 0U);
      shader.data.resize(transformed.size() / 4U);
      std::copy(transformed.begin(), transformed.end(),
                reinterpret_cast<char*>(dmlc::BeginPtr(shader.data)));
    }
    code_data << spirv_tools.BinaryToText(shader.data);
    smap[f_name] = std::move(shader);
  }

  return std::make_pair(smap, code_data.str());
}

#else

std::pair<std::unordered_map<std::string, runtime::SPIRVShader>, std::string> LowerToSPIRV(
    IRModule mod, Target target) {
  LOG(FATAL)
      << "LowerToSPIRV is called but SPIRV codegen is not enabled. Please set -DUSE_VULKAN=ON.";
  return {};
}

#endif

}  // namespace codegen
}  // namespace tvm
