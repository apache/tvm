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
#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
using namespace tir;
namespace relay {
namespace contrib {
namespace cmsisnn {

class CodeGenCMSISNN : public codegen::CodeGenCHost {
 public:
  void Init(bool output_ssa, bool emit_asserts, std::string target_str) {
    decl_stream << "#include <stdio.h>\n";
    decl_stream << "#include <stdlib.h>\n";
    decl_stream << "#include <dlpack/dlpack.h>\n";
    decl_stream << "#include <tvm/runtime/crt/module.h>\n";
    decl_stream << "#include <arm_nnfunctions.h>\n";
    decl_stream << "#include <arm_nn_types.h>\n";
    CodeGenCHost::Init(output_ssa, emit_asserts, target_str);
  }

  /*!
   * \brief Emit code that offloads a subgraph to the Cortex-M
   *
   * \return string of code that offloads a subgraph to the Cortex-M
   */
  void AddFunction(const PrimFunc& prim_func) { CodeGenC::AddFunction(prim_func); }

 private:
  /*!  * \brief Emit the CMSIS-NN context buffer */
  void VisitStmt_(const AllocateNode* op) {
    context_buffer_name_ = op->buffer_var->name_hint;
    context_buffer_size_ = op->constant_allocation_size();
    CodeGenC::VisitStmt_(op);
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern */
  void VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
    if (!op->op.same_as(builtin::call_extern())) {
      CodeGenCHost::VisitExpr_(op, os);
      return;
    }
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    if (cmsis_func_name == "arm_softmax_s8" || cmsis_func_name == "arm_elementwise_mul_s8" ||
        cmsis_func_name == "arm_elementwise_add_s8") {
      CodeGenC::VisitExpr_(op, os);
    } else if (cmsis_func_name == "arm_convolve_wrapper_s8") {
      EmitConv2D(op);
    }
    return;
  }

  /*!  * \brief Emits cmsis_nn_context struct */
  std::string EmitCMSISNNContext(std::ostream& os, std::string buf_name, int buf_size) {
    std::string struct_name = "context";
    PrintIndent();
    os << "cmsis_nn_context " << struct_name << "= {" << buf_name << "," << buf_size << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_conv_params struct */
  std::string EmitCMSISNNConvParams(std::ostream& os, int32_t input_offset, int32_t output_offset,
                                    int32_t stride_w, int32_t stride_h, int32_t padding_w,
                                    int32_t padding_h, int32_t dilation_w, int32_t dilation_h,
                                    int32_t clip_min, int32_t clip_max) {
    std::string struct_name = "conv_params";
    PrintIndent();
    os << "cmsis_nn_tile stride = {" << stride_w << "," << stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding = {" << padding_w << "," << padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile dilation = {" << dilation_w << "," << dilation_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << clip_min << "," << clip_max << "};\n";
    PrintIndent();
    os << "cmsis_nn_conv_params " << struct_name << " = {" << input_offset << ", " << output_offset
       << ", stride, padding, dilation, activation};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_per_channel_quant_params struct */
  std::string EmitCMSISNNPerChannelQuantParams(std::ostream& os, std::string multiplier,
                                               std::string shift) {
    std::string struct_name = "quant_params";
    PrintIndent();
    os << "cmsis_nn_per_channel_quant_params " << struct_name << " = {" << multiplier << ", "
       << shift << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_dims struct */
  std::string EmitCMSISNNDims(std::ostream& os, std::string tensor_type, int32_t n, int32_t h,
                              int32_t w, int32_t c) {
    std::string struct_name = tensor_type + "_dims";
    PrintIndent();
    os << "cmsis_nn_dims " << struct_name << " = {" << n << "," << h << "," << w << "," << c
       << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern */
  void EmitConv2D(const CallNode* op) {
    static const int max_num_args = 35;
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    bool bias_enabled = false;
    if (op->args.size() == max_num_args) {
      bias_enabled = true;
    }

    auto get_var_name = [](const CallNode* op, int id) {
      return op->args[id].as<VarNode>()->name_hint.c_str();
    };
    auto get_arg_value = [](const CallNode* op, int id) {
      return op->args[id].as<IntImmNode>()->value;
    };
    int arg_id = 0;
    std::string input_data = get_var_name(op, ++arg_id);
    std::string filter_data = get_var_name(op, ++arg_id);
    std::string multiplier = get_var_name(op, ++arg_id);
    std::string bias_data("0x0");
    if (bias_enabled) {
      bias_data = get_var_name(op, ++arg_id);
    }
    std::string shift = get_var_name(op, ++arg_id);
    std::string output_data = get_var_name(op, ++arg_id);

    std::string context_buffer_name = op->args[++arg_id].as<StringImmNode>()->value;
    int context_buffer_size = get_arg_value(op, ++arg_id);
    int input_offset = get_arg_value(op, ++arg_id);
    int output_offset = get_arg_value(op, ++arg_id);
    int stride_w = get_arg_value(op, ++arg_id);
    int stride_h = get_arg_value(op, ++arg_id);
    int padding_w = get_arg_value(op, ++arg_id);
    int padding_h = get_arg_value(op, ++arg_id);
    int dilation_w = get_arg_value(op, ++arg_id);
    int dilation_h = get_arg_value(op, ++arg_id);
    int clip_min = get_arg_value(op, ++arg_id);
    int clip_max = get_arg_value(op, ++arg_id);
    int input_n = get_arg_value(op, ++arg_id);
    int input_h = get_arg_value(op, ++arg_id);
    int input_w = get_arg_value(op, ++arg_id);
    int input_c = get_arg_value(op, ++arg_id);
    int filter_n = get_arg_value(op, ++arg_id);
    int filter_h = get_arg_value(op, ++arg_id);
    int filter_w = get_arg_value(op, ++arg_id);
    int filter_c = get_arg_value(op, ++arg_id);
    int bias_n = get_arg_value(op, ++arg_id);
    int bias_h = get_arg_value(op, ++arg_id);
    int bias_w = get_arg_value(op, ++arg_id);
    int bias_c = get_arg_value(op, ++arg_id);
    int output_n = get_arg_value(op, ++arg_id);
    int output_h = get_arg_value(op, ++arg_id);
    int output_w = get_arg_value(op, ++arg_id);
    int output_c = get_arg_value(op, ++arg_id);

    std::string context = EmitCMSISNNContext(stream, context_buffer_name, context_buffer_size);
    std::string conv_params =
        EmitCMSISNNConvParams(stream, input_offset, output_offset, stride_w, stride_h, padding_w,
                              padding_h, dilation_w, dilation_h, clip_min, clip_max);
    std::string quant_params = EmitCMSISNNPerChannelQuantParams(stream, multiplier, shift);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_n, input_h, input_w, input_c);
    std::string filter_dim =
        EmitCMSISNNDims(stream, "filter", filter_n, filter_h, filter_w, filter_c);
    std::string bias_dim = EmitCMSISNNDims(stream, "bias", bias_n, bias_h, bias_w, bias_c);
    std::string output_dim =
        EmitCMSISNNDims(stream, "output", output_n, output_h, output_w, output_c);

    PrintIndent();
    stream << "arm_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << conv_params << ", ";
    stream << "&" << quant_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", " << filter_data << ", ";
    stream << "&" << bias_dim << ", " << bias_data << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    PrintIndent();
    stream << "if (status != ARM_MATH_SUCCESS) {\n";
    PrintIndent();
    PrintIndent();
    stream << "return -1;\n";
    PrintIndent();
    stream << "}\n";
  }

 private:
  std::string context_buffer_name_ = "NULL";
  int context_buffer_size_ = 0;
};

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenCMSISNN codegen;
  Array<String> function_names;
  codegen.Init(output_ssa, emit_asserts, target->str());
  for (auto kv : mod->functions) {
    auto prim_func = Downcast<PrimFunc>(kv.second);
    auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    function_names.push_back(global_symbol.value());
    codegen.AddFunction(prim_func);
  }
  std::string code = codegen.Finish();
  return codegen::CSourceModuleCreate(code, "c", function_names);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
