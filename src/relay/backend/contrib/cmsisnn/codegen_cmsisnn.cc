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
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../runtime/file_utils.h"
#include "../../../../target/source/codegen_c.h"
#include "../../../qnn/utils.h"

namespace tvm {
namespace runtime {

using namespace tir;

class CodeGenCMSISNN : public tvm::codegen::CodeGenC {
 public:
  void Init(bool output_ssa) {
    decl_stream << "#include <stdio.h>\n";
    decl_stream << "#include <stdlib.h>\n";
    decl_stream << "#include <dlpack/dlpack.h>\n";
    decl_stream << "#include <tvm/runtime/crt/module.h>\n";
    decl_stream << "#include <arm_nnfunctions.h>\n";
    CodeGenC::Init(output_ssa);
  }

  /*!
   * \brief Emit code that offloads a subgraph to the Cortex-M
   *
   * \return string of code that offloads a subgraph to the Cortex-M
   */
  void AddFunction(const PrimFunc& prim_func) {
    PrintExternCPrefix(stream);
    CodeGenC::AddFunction(prim_func);
    PrintExternCPostfix(stream);
  }

 private:
  void VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
    if (!op->op.same_as(builtin::call_extern())) {
      return;
    }
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    if (cmsis_func_name == "arm_softmax_s8") {
      EmitSoftmax(op);
    }
    return;
  }

  /*!  * \brief Creates a cplusplus guard prefix for extern "C" printing */
  void PrintExternCPrefix(std::ostringstream& ss) {
    PrintIndent();
    ss << "#ifdef __cplusplus\n";
    ss << "extern \"C\" {\n";
    ss << "#endif\n";
  }

  /*!  * \brief Creates a cplusplus guard postfix for extern "C" printing */
  void PrintExternCPostfix(std::ostringstream& ss) {
    PrintIndent();
    ss << "#ifdef __cplusplus\n";
    ss << "}\n";
    ss << "#endif\n";
  }

  /*!  * \brief Emits CMSIS-NN code block for softmax */
  void EmitSoftmax(const CallNode* op) {
    // @tir.call_extern("arm_softmax_s8", buffer_0, num_rows, row_size, scale, buffer_1, dtype=int8)
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    int32_t num_rows = op->args[2].as<IntImmNode>()->value;
    int32_t row_size = op->args[3].as<IntImmNode>()->value;
    float quant_scale = op->args[4].as<FloatImmNode>()->value;

    // calculate multiplier and shift for CMSIS-NN softmax API
    // Note: tfl micro assumptions
    // TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
    // TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    // TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
    double beta = 1.0;
    int32_t input_bits = 5;
    double beta_multiplier = (beta * quant_scale * (1 << (31 - input_bits)));
    beta_multiplier = std::min<double>(beta_multiplier, (1ll << 31) - 1.0);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(beta_multiplier);
    int32_t mult = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);
    int32_t diff_min = (1 << 5) - 1;
    diff_min <<= (31 - 5);
    diff_min >>= shift;
    diff_min *= -1;

    PrintIndent();
    stream << "int32_t num_rows = " << num_rows << ";\n";
    PrintIndent();
    stream << "int32_t row_size = " << row_size << ";\n";
    PrintIndent();
    stream << "int32_t mult = " << mult << ";\n";
    PrintIndent();
    stream << "int32_t shift = " << shift << ";\n";
    PrintIndent();
    stream << "int32_t diff_min = " << diff_min << ";\n";
    PrintIndent();
    stream << cmsis_func_name << "(buffer,";
    PrintIndent();
    stream << " num_rows, row_size, mult, shift, diff_min, buffer1);\n";
    PrintIndent();
    stream << "return;\n";
  }
};

class CMSISNNModuleNode : public runtime::ModuleNode {
 public:
  CMSISNNModuleNode(const std::string& code, const std::string& fmt,
                    const Array<String>& func_names)
      : code_(code), fmt_(fmt), func_names_(func_names) {}

  std::string GetSource(const std::string& format) final { return code_; }

  const char* type_key() const { return "c"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_[0]; });
    } else if (name == "get_func_names") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->func_names_; });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "c" || fmt == "cu") {
      ICHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
  Array<String> func_names_;
};

class CMSISNNModule : public Module {
 public:
  CMSISNNModule() {}
  explicit CMSISNNModule(ObjectPtr<Object> n) : Module(n) {}
  inline CMSISNNModuleNode* operator->();
  inline const CMSISNNModuleNode* operator->() const;
};

inline CMSISNNModuleNode* CMSISNNModule::operator->() {
  return static_cast<CMSISNNModuleNode*>(get_mutable());
}

static Module CMSISNNModuleNodeCreate(IRModule mod) {
  bool output_ssa = false;
  CodeGenCMSISNN cg;
  Array<String> function_names;
  cg.Init(output_ssa);
  ICHECK(mod->functions.size() == 1) << "Supports modules with single PrimFunc.";
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenCHost: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined())
        << "CodeGenCHost: Expect PrimFunc to have the global_symbol attribute";
    function_names.push_back(global_symbol.value());
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  auto n = make_object<CMSISNNModuleNode>(code, "c", function_names);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.cmsisnn.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CMSISNNModuleNodeCreate(args[0]);
});

}  // namespace runtime
}  // namespace tvm
