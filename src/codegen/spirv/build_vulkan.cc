/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vulkan.cc
 * \brief Build SPIRV block
 */
// Use libspirv for parsing and validating code.
#include <libspirv.h>
#include <dmlc/memory_io.h>
#include <tvm/ir_pass.h>

#include "codegen_spirv.h"
#include "../build_common.h"
#include "../../runtime/vulkan/vulkan_module.h"

namespace tvm {
namespace codegen {

class SPIRVTools {
 public:
  SPIRVTools() {
    ctx_ = spvContextCreate(SPV_ENV_VULKAN_1_0);
  }
  ~SPIRVTools() {
    spvContextDestroy(ctx_);
  }
  std::string BinaryToText(const std::vector<uint32_t>& bin) {
    spv_text text = nullptr;
    spv_diagnostic diagnostic;
    spv_const_binary_t spv_bin{bin.data(), bin.size()};
    spv_result_t res;

    res = spvBinaryToText(
       ctx_, spv_bin.code, spv_bin.wordCount,
      SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
           SPV_BINARY_TO_TEXT_OPTION_INDENT,
        &text, &diagnostic);

    CHECK_EQ(res, SPV_SUCCESS)
        << " line=" << diagnostic->position.line
        << " column=" << diagnostic->position.column
        << " index=" << diagnostic->position.index
        << " error:" << diagnostic->error;

    std::string ret(text->str);
    spvTextDestroy(text);
    return ret;
  }

 private:
  spv_context ctx_;
};

runtime::Module BuildSPIRV(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  using tvm::runtime::VulkanShader;

  std::ostringstream code_data;
  static SPIRVTools spirv_tools;
  std::unordered_map<std::string, VulkanShader> smap;

  const auto* postproc = Registry::Get("tvm_callback_vulkan_postproc");

  CodeGenSPIRV cg;
  for (LoweredFunc f : funcs) {
    f = PointerValueTypeRewrite(f);
    VulkanShader shader;
    shader.data = cg.BuildFunction(f);

    if (postproc != nullptr) {
      TVMByteArray arr;
      arr.data = reinterpret_cast<const char*>(dmlc::BeginPtr(shader.data));
      arr.size = shader.data.size() * sizeof(uint32_t);
      std::string transformed = (*postproc)(arr);
      CHECK_EQ(transformed.length() % 4U, 0U);
      shader.data.resize(transformed.size() / 4U);
      std::copy(transformed.begin(), transformed.end(),
                reinterpret_cast<char*>(dmlc::BeginPtr(shader.data)));
    }
    code_data << spirv_tools.BinaryToText(shader.data);
    smap[f->name] = std::move(shader);
  }
  return runtime::VulkanModuleCreate(
     smap, ExtractFuncInfo(funcs), code_data.str());
}

TVM_REGISTER_API("codegen.build_vulkan")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSPIRV(args[0]);
  });

}  // namespace codegen
}  // namespace tvm
