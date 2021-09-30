
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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../qnn/utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

class RelayToTIRVisitor : public MixedModeVisitor {
 public:
  explicit RelayToTIRVisitor(String func_name) : func_name_(func_name) {}

  tir::PrimFunc GetReplacementPrimFunc() { return primfunc_; }

 private:
  template <typename T>
  const T ArgumentToConstantValue(const Expr& arg) {
    const ConstantNode* constant_node = arg.as<ConstantNode>();
    return static_cast<const T*>(constant_node->data->data)[0];
  }

  void CreatePrimFuncForExtern(Array<tir::Var> func_signature,
                               tvm::Array<PrimExpr> call_extern_args) {
    Map<String, ObjectRef> dict_attrs;
    dict_attrs.Set("global_symbol", func_name_);
    dict_attrs.Set("tir.noalias", Bool(true));

    tir::Stmt body = tir::Evaluate(
        tvm::tir::Call(DataType::Int(8), tir::builtin::call_extern(), call_extern_args));

    primfunc_ = tir::PrimFunc(func_signature, body, VoidType(), Map<tir::Var, tir::Buffer>(),
                              DictAttrs(dict_attrs));
  }

  void EmitSoftMax(const Expr& expr) {
    auto* quantize_call = expr.as<CallNode>();
    auto* softmax_call = quantize_call->args[0].as<CallNode>();
    auto* dequant_call = softmax_call->args[0].as<CallNode>();
    const float quant_scale = ArgumentToConstantValue<float>(dequant_call->args[1]);

    // assuming layout as NHWC
    auto shape = quantize_call->type_as<TensorTypeNode>()->shape;
    int trailing_dim = shape.size() - 1;
    int row_size = shape[trailing_dim].as<tir::IntImmNode>()->value;
    int num_rows = 1;
    for (int i = 0; i < trailing_dim; ++i) {
      num_rows *= shape[i].as<tir::IntImmNode>()->value;
    }

    // calculate multiplier and shift for CMSIS-NN softmax API
    // Note: TensorFlow Lite Micro assumptions
    // Output zero point and scale are fixed to -128 and 1 / 256
    // https://github.com/tensorflow/tflite-micro/blob/d97cd0908d8cf5021e9d86f05a49888bee28c2a4/tensorflow/lite/micro/kernels/softmax_common.cc#L47
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

    auto in_var = tir::Var("input", DataType::Handle(8));
    auto out_var = tir::Var("output", DataType::Handle(8));

    Array<tir::Var> func_signature{in_var, out_var};

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_softmax_s8"),    in_var,
        IntImm(DataType::Int(32), num_rows), IntImm(DataType::Int(32), row_size),
        IntImm(DataType::Int(32), mult),     IntImm(DataType::Int(32), shift),
        IntImm(DataType::Int(32), diff_min), out_var};

    CreatePrimFuncForExtern(func_signature, args);
  }

  void EmitMul(const Expr& expr) {
    auto* mul_call = expr.as<CallNode>();

    const float input_0_scale = ArgumentToConstantValue<float>(mul_call->args[2]);
    const int32_t input_0_zero_point = ArgumentToConstantValue<int32_t>(mul_call->args[3]);
    const float input_1_scale = ArgumentToConstantValue<float>(mul_call->args[4]);
    const int32_t input_1_zero_point = ArgumentToConstantValue<int32_t>(mul_call->args[5]);
    const float output_scale = ArgumentToConstantValue<float>(mul_call->args[6]);
    const int32_t output_zero_point = ArgumentToConstantValue<int32_t>(mul_call->args[7]);

    double quantized_multiplier = static_cast<double>(input_0_scale) *
                                  static_cast<double>(input_1_scale) /
                                  static_cast<double>(output_scale);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(quantized_multiplier);
    int32_t output_multiplier = std::get<0>(mult_shift_pair);
    int32_t output_shift = std::get<1>(mult_shift_pair);

    PrimExpr tensor_size = mul_call->type_as<TensorTypeNode>()->Size();

    tir::Var input_0("input_0", DataType::Handle(8));
    tir::Var input_1("input_1", DataType::Handle(8));
    tir::Var output("output", DataType::Handle(8));

    Array<tir::Var> func_signature{input_0, input_1, output};

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_elementwise_mul_s8"),
        input_0,
        input_1,
        IntImm(DataType::Int(32), -input_0_zero_point),
        IntImm(DataType::Int(32), -input_1_zero_point),
        output,
        IntImm(DataType::Int(32), output_zero_point),
        IntImm(DataType::Int(32), output_multiplier),
        IntImm(DataType::Int(32), output_shift),
        IntImm(DataType::Int(32), std::numeric_limits<int8_t>::min()),
        IntImm(DataType::Int(32), std::numeric_limits<int8_t>::max()),
        tensor_size,
    };

    CreatePrimFuncForExtern(func_signature, args);
  }

  void VisitExpr_(const CallNode* call) final {
    auto* func = call->op.as<FunctionNode>();
    if (func == nullptr) {
      return;
    }

    auto comp_name = func->GetAttr<String>(attr::kComposite);
    if (comp_name.defined() && comp_name == "cmsisnn.quantized_softmax") {
      EmitSoftMax(func->body);
    }
    if (comp_name.defined() && comp_name == "cmsisnn.quantized_mul") {
      EmitMul(func->body);
    }
  }

 public:
  String func_name_;
  tir::PrimFunc primfunc_;
};

IRModule GenerateTIR(IRModule mod) {
  String func_name;
  Function func;

  // Obtain external Relay Function that needs to be translated into TIR
  ICHECK(mod->functions.size() == 1) << "Supports modules with single external Relay function.";
  for (auto kv : mod->functions) {
    func = Downcast<Function>(kv.second);
    func_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
  }

  // Prepare PrimFunc from Relay Function
  auto relay_to_tir = RelayToTIRVisitor(func_name);
  relay_to_tir.VisitExpr(func->body);

  // Build the TIR IRModule from the generated PrimFunc
  Map<GlobalVar, BaseFunc> var_func_map;
  var_func_map.Set(GlobalVar(func_name), relay_to_tir.GetReplacementPrimFunc());
  return IRModule(var_func_map);
}

transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return GenerateTIR(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIR", {});
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
