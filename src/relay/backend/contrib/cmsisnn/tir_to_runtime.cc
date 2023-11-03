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
#include <tvm/ir/transform.h>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../runtime/file_utils.h"
#include "../../../../target/source/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"
#include "compiler_attrs.h"

namespace tvm {
using namespace tir;
namespace relay {
namespace contrib {
namespace cmsisnn {

class CodeGenCMSISNN : public codegen::CodeGenCHost {
 public:
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl, std::string target_str,
            bool debug_last_error) {
    this->debug_last_error = debug_last_error;
    std::unordered_set<std::string> devices;
    devices.insert("cmsis-nn");
    CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl, target_str, devices);
  }

 private:
  /*!  * \brief Enable storing the last error */
  bool debug_last_error;

  /*!  * \brief CMSIS-NN context buffer info */
  struct CMSISNNContextBuffer {
    std::string name;
    int size;
  };

  /*!  * \brief CMSIS-NN buffer dimensions */
  struct CMSISNNDims {
    int n;
    int h;
    int w;
    int c;
  };

  /*!  * \brief CMSIS-NN Conv2D and Depthwise parameters */
  struct Conv2DParams {
    int input_offset;
    int output_offset;
    int stride_w;
    int stride_h;
    int padding_w;
    int padding_h;
    int dilation_w;
    int dilation_h;
    int clip_min;
    int clip_max;
    int depth_multiplier;
  };

  /*!  * \brief CMSIS-NN Conv2D and Depthwise parameters */
  struct FCParams {
    int input_offset;
    int filter_offset;
    int output_offset;
    int clip_min;
    int clip_max;
    int multiplier;
    int shift;
  };

  struct PoolParams {
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int clip_min;
    int clip_max;
  };

  struct CMSISNNSoftmaxLutS16 {
    std::string exp_lut_name;
    std::string one_by_one_lut_name;
  };

  using codegen::CodeGenCHost::VisitStmt_;

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern */
  void VisitExpr_(const CallNode* op, std::ostream& os) final {
    if (!op->op.same_as(builtin::call_extern())) {
      CodeGenCHost::VisitExpr_(op, os);
      return;
    }

    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    if (cmsis_func_name == "arm_softmax_s8" || cmsis_func_name == "arm_elementwise_mul_s8" ||
        cmsis_func_name == "arm_elementwise_add_s8" ||
        cmsis_func_name == "arm_elementwise_mul_s16" ||
        cmsis_func_name == "arm_elementwise_add_s16") {
      CodeGenC::VisitExpr_(op, os);
    } else if (cmsis_func_name == "arm_convolve_wrapper_s8" ||
               cmsis_func_name == "arm_convolve_wrapper_s16" ||
               cmsis_func_name == "arm_depthwise_conv_wrapper_s8" ||
               cmsis_func_name == "arm_depthwise_conv_wrapper_s16") {
      EmitConv2D(op);
    } else if (cmsis_func_name == "arm_fully_connected_s8" ||
               cmsis_func_name == "arm_fully_connected_s16") {
      EmitFullyConnected(op);
    } else if (cmsis_func_name == "arm_avgpool_s8" || cmsis_func_name == "arm_avgpool_s16" ||
               cmsis_func_name == "arm_max_pool_s8" || cmsis_func_name == "arm_max_pool_s16") {
      EmitPool2D(op);
    } else if (cmsis_func_name == "arm_softmax_s16") {
      EmitSoftmaxInt16(op);
    }
    return;
  }

  /*!  * \brief Emits cmsis_nn_context struct */
  std::string EmitCMSISNNContext(std::ostream& os, CMSISNNContextBuffer context_buffer) {
    std::string struct_name = "context";
    PrintIndent();
    os << "cmsis_nn_context " << struct_name << "= {" << context_buffer.name << ","
       << context_buffer.size << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_conv_params struct */
  std::string EmitCMSISNNConvParams(std::ostream& os, Conv2DParams params) {
    std::string struct_name = "cmsis_nn_conv_params";
    std::string instance_name = "conv_params";
    if (params.depth_multiplier != -1) {
      struct_name = "cmsis_nn_dw_conv_params";
    }
    PrintIndent();
    os << "cmsis_nn_tile stride = {" << params.stride_w << "," << params.stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding = {" << params.padding_w << "," << params.padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile dilation = {" << params.dilation_w << "," << params.dilation_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {" << params.input_offset << ", "
       << params.output_offset;
    if (params.depth_multiplier != -1) {
      os << ", " << params.depth_multiplier;
    }
    os << ", stride, padding, dilation, activation};\n";
    return instance_name;
  }

  /*!  * \brief Emits cmsis_nn_fc_params struct */
  std::string EmitCMSISNNFCParams(std::ostream& os, FCParams params) {
    std::string struct_name = "cmsis_nn_fc_params";
    std::string instance_name = "fc_params";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {" << params.input_offset << ", "
       << params.filter_offset << ", " << params.output_offset;
    os << ", activation};\n";
    return instance_name;
  }

  /*!  * \brief Emits cmsis_nn_pool_params struct */
  std::string EmitCMSISNNPoolParams(std::ostream& os, PoolParams params) {
    std::string struct_name = "cmsis_nn_pool_params";
    std::string instance_name = "pool_params";
    PrintIndent();
    os << "cmsis_nn_tile stride = {" << params.stride_w << "," << params.stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding = {" << params.padding_w << "," << params.padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {stride, padding, activation};\n";
    return instance_name;
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

  /*!  * \brief Emits cmsis_nn_per_tensor_quant_params struct */
  std::string EmitCMSISNNPerTensorQuantParams(std::ostream& os, int multiplier, int shift) {
    std::string struct_name = "quant_params";
    PrintIndent();
    os << "cmsis_nn_per_tensor_quant_params " << struct_name << " = {" << multiplier << ", "
       << shift << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_dims struct */
  std::string EmitCMSISNNDims(std::ostream& os, std::string tensor_type, CMSISNNDims dims) {
    std::string struct_name = tensor_type + "_dims";
    PrintIndent();
    os << "cmsis_nn_dims " << struct_name << " = {" << dims.n << "," << dims.h << "," << dims.w
       << "," << dims.c << "};\n";
    return struct_name;
  }
  /*!  * \brief Emits cmsis_nn_softmax_params struct */
  std::string EmitCMSISNNSoftmaxLutS16(std::ostream& os, CMSISNNSoftmaxLutS16 softmax_params) {
    std::string struct_name = "softmax_params";
    PrintIndent();
    os << "cmsis_nn_softmax_lut_s16 " << struct_name << "= {" << softmax_params.exp_lut_name << ", "
       << softmax_params.one_by_one_lut_name << "};\n";
    return struct_name;
  }

  /*!  * \brief Deduces variable name from call_extern argument resting at id */
  std::string VarNameFromArg(const CallNode* op, int id) {
    return op->args[id].as<VarNode>()->name_hint.c_str();
  }

  /*!  * \brief Deduces value from call_extern argument resting at id */
  int ValueFromArg(const CallNode* op, int id) { return op->args[id].as<IntImmNode>()->value; }

  /*!  * \brief extracts CMSIS-NN context buffer information */
  CMSISNNContextBuffer extract_context_buffer_info(const CallNode* op, int base_pos) {
    CMSISNNContextBuffer context_buffer;

    // The argument could be a Var if it is allocated to hold the
    // context buffer OR it will be a StringImm with "NULL"
    if (op->args[base_pos]->IsInstance<VarNode>()) {
      context_buffer.name = op->args[base_pos].as<VarNode>()->name_hint;
    } else {
      context_buffer.name = op->args[base_pos].as<StringImmNode>()->value;
    }
    context_buffer.size = ValueFromArg(op, base_pos + 1);
    return context_buffer;
  }

  /*!  * \brief extracts CMSIS-NN conv2d parameters from call_extern */
  Conv2DParams extract_conv2d_params(const CallNode* op, int base_pos) {
    Conv2DParams conv2d_params;
    conv2d_params.input_offset = ValueFromArg(op, base_pos);
    conv2d_params.output_offset = ValueFromArg(op, ++base_pos);
    conv2d_params.stride_w = ValueFromArg(op, ++base_pos);
    conv2d_params.stride_h = ValueFromArg(op, ++base_pos);
    conv2d_params.padding_w = ValueFromArg(op, ++base_pos);
    conv2d_params.padding_h = ValueFromArg(op, ++base_pos);
    conv2d_params.dilation_w = ValueFromArg(op, ++base_pos);
    conv2d_params.dilation_h = ValueFromArg(op, ++base_pos);
    conv2d_params.clip_min = ValueFromArg(op, ++base_pos);
    conv2d_params.clip_max = ValueFromArg(op, ++base_pos);
    conv2d_params.depth_multiplier = ValueFromArg(op, ++base_pos);
    return conv2d_params;
  }

  /*!  * \brief extracts CMSIS-NN FC parameters from call_extern */
  FCParams extract_fc_params(const CallNode* op, int base_pos) {
    FCParams fc_params;
    fc_params.input_offset = ValueFromArg(op, base_pos);
    fc_params.filter_offset = ValueFromArg(op, ++base_pos);
    fc_params.output_offset = ValueFromArg(op, ++base_pos);
    fc_params.clip_min = ValueFromArg(op, ++base_pos);
    fc_params.clip_max = ValueFromArg(op, ++base_pos);
    fc_params.multiplier = ValueFromArg(op, ++base_pos);
    fc_params.shift = ValueFromArg(op, ++base_pos);
    return fc_params;
  }

  /*!  * \brief extracts CMSIS-NN Pooling parameters from call_extern */
  PoolParams extract_pool_params(const CallNode* op, int base_pos) {
    PoolParams pool_params;
    pool_params.stride_h = ValueFromArg(op, base_pos);
    pool_params.stride_w = ValueFromArg(op, ++base_pos);
    pool_params.padding_h = ValueFromArg(op, ++base_pos);
    pool_params.padding_w = ValueFromArg(op, ++base_pos);
    pool_params.clip_min = ValueFromArg(op, ++base_pos);
    pool_params.clip_max = ValueFromArg(op, ++base_pos);
    return pool_params;
  }

  /*!  * \brief extracts CMSIS-NN buffer dimensions from call_extern */
  CMSISNNDims extract_buffer_dims(const CallNode* op, int base_pos) {
    CMSISNNDims dims;
    dims.n = ValueFromArg(op, base_pos);
    dims.h = ValueFromArg(op, ++base_pos);
    dims.w = ValueFromArg(op, ++base_pos);
    dims.c = ValueFromArg(op, ++base_pos);
    return dims;
  }
  /*!  * \brief extracts CMSIS-NN softmax LUTs from call_extern */
  CMSISNNSoftmaxLutS16 extract_softmax_softmax_lut_s16(const CallNode* op, int exp_lut_pos,
                                                       int one_by_one_lut_pos) {
    CMSISNNSoftmaxLutS16 softmax_params;
    softmax_params.exp_lut_name = op->args[exp_lut_pos].as<VarNode>()->name_hint;
    softmax_params.one_by_one_lut_name = op->args[one_by_one_lut_pos].as<VarNode>()->name_hint;
    return softmax_params;
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising convolution */
  void EmitConv2D(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      CONV2D_PARAMS_POS = 3,
      INPUT_DIM_POS = 14,
      FILTER_DIM_POS = 18,
      BIAS_DIM_POS = 22,
      OUTPUT_DIM_POS = 26,
      MAX_NUM_ARGS = 36
    };

    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string filter_data = VarNameFromArg(op, ++arg_id);
    std::string multiplier = VarNameFromArg(op, ++arg_id);
    std::string bias_data("NULL");
    if (op->args.size() == CallExternArgPos::MAX_NUM_ARGS) {
      bias_data = VarNameFromArg(op, ++arg_id);
    }
    std::string shift = VarNameFromArg(op, ++arg_id);
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int conv2d_params_pos = arg_id + CallExternArgPos::CONV2D_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int bias_dim_pos = arg_id + CallExternArgPos::BIAS_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    Conv2DParams conv2d_params = extract_conv2d_params(op, conv2d_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims bias_dims = extract_buffer_dims(op, bias_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    // Emit CMSIS-NN API arguments
    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string conv_params = EmitCMSISNNConvParams(stream, conv2d_params);
    std::string quant_params = EmitCMSISNNPerChannelQuantParams(stream, multiplier, shift);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string bias_dim = EmitCMSISNNDims(stream, "bias", bias_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    // Emit CMSIS-NN API
    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << conv_params << ", ";
    stream << "&" << quant_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", " << filter_data << ", ";
    stream << "&" << bias_dim << ", " << bias_data << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising fully connected */
  void EmitFullyConnected(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      FC_PARAMS_POS = 3,
      INPUT_DIM_POS = 10,
      FILTER_DIM_POS = 14,
      BIAS_DIM_POS = 18,
      OUTPUT_DIM_POS = 22,
      MAX_NUM_ARGS = 30
    };

    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string filter_data = VarNameFromArg(op, ++arg_id);
    std::string bias_data("NULL");
    if (op->args.size() == CallExternArgPos::MAX_NUM_ARGS) {
      bias_data = VarNameFromArg(op, ++arg_id);
    }
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int fc_params_pos = arg_id + CallExternArgPos::FC_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int bias_dim_pos = arg_id + CallExternArgPos::BIAS_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    FCParams fc_params = extract_fc_params(op, fc_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims bias_dims = extract_buffer_dims(op, bias_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    // Emit CMSIS-NN API arguments
    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string cmsisnn_fc_params = EmitCMSISNNFCParams(stream, fc_params);
    std::string quant_params =
        EmitCMSISNNPerTensorQuantParams(stream, fc_params.multiplier, fc_params.shift);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string bias_dim = EmitCMSISNNDims(stream, "bias", bias_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << cmsisnn_fc_params << ", ";
    stream << "&" << quant_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", " << filter_data << ", ";
    stream << "&" << bias_dim << ", " << bias_data << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising pooling ops */
  void EmitPool2D(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      POOL_PARAMS_POS = 3,
      INPUT_DIM_POS = 9,
      FILTER_DIM_POS = 13,
      OUTPUT_DIM_POS = 17,
      MAX_NUM_ARGS = 23
    };
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int pool_params_pos = arg_id + CallExternArgPos::POOL_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    PoolParams pool_params = extract_pool_params(op, pool_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string cmsisnn_pool_params = EmitCMSISNNPoolParams(stream, pool_params);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << cmsisnn_pool_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  void EmitSoftmaxInt16(const CallNode* op) {
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    int num_rows = ValueFromArg(op, ++arg_id);
    int row_size = ValueFromArg(op, ++arg_id);
    int multiplier = ValueFromArg(op, ++arg_id);
    int shift = ValueFromArg(op, ++arg_id);
    // extracting LUT names from call_extern
    CMSISNNSoftmaxLutS16 softmax_params_buffer =
        extract_softmax_softmax_lut_s16(op, arg_id + 1, arg_id + 2);
    arg_id += 2;
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // Emit CMSIS-NN API arguments
    std::string softmax_params = EmitCMSISNNSoftmaxLutS16(stream, softmax_params_buffer);

    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << input_data << ", ";
    stream << num_rows << ", ";
    stream << row_size << ", ";
    stream << multiplier << ", ";
    stream << shift << ", ";
    stream << "&" << softmax_params << ", ";
    stream << output_data << ");\n";
    EmitErrorCheck();
  }

  void EmitErrorCheck() {
    auto emit_error = [&](std::string error) {
      if (this->debug_last_error) {
        stream << "TVMAPISetLastError(\"" << error << "\"); ";
      }
    };

    PrintIndent();
    stream << "switch (!status) {\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_SUCCESS: break;\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_ARG_ERROR: ";
    emit_error("ARM_CMSIS_NN_ARG_ERROR");
    stream << "return -1;\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_NO_IMPL_ERROR: ";
    emit_error("ARM_CMSIS_NN_NO_IMPL_ERROR");
    stream << "return -1;\n";
    PrintIndent();
    stream << "}\n";
  }
};

static CMSISNNCompilerConfig GetCompilerAttrs() {
  auto ctx = tvm::tir::transform::PassContext::Current();
  Optional<CMSISNNCompilerConfig> cfg =
      ctx->GetConfig<CMSISNNCompilerConfig>("relay.ext.cmsisnn.options");
  if (!cfg.defined()) {
    return AttrsWithDefaultValues<CMSISNNCompilerConfig>();
  }
  return cfg.value();
}

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = false;
  bool debug_last_error = GetCompilerAttrs()->debug_last_error;
  CodeGenCMSISNN codegen;
  codegen.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), debug_last_error);

  std::vector<std::pair<tvm::GlobalVar, tvm::PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    funcs.push_back({gvar, Downcast<PrimFunc>(base_func)});
  }

  std::sort(funcs.begin(), funcs.end(),
            [](std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_a,
               std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_b) {
              std::string name_hint_a = kv_a.first->name_hint;
              std::string name_hint_b = kv_b.first->name_hint;
              size_t name_a_length = name_hint_a.length();
              size_t name_b_length = name_hint_b.length();
              if (name_a_length < name_b_length) return true;
              if (name_a_length > name_b_length) return false;
              return name_hint_a < name_hint_b;
            });

  for (auto [gvar, prim_func] : funcs) {
    codegen.AddFunction(gvar, prim_func);
  }
  std::string code = codegen.Finish();

  Array<String> function_names;
  for (auto [gvar, prim_func] : funcs) {
    function_names.push_back(codegen.GetFunctionName(gvar));
  }

  return codegen::CSourceModuleCreate(code, "c", function_names);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
