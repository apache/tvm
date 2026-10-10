
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
 * \file backend/trn/op/target_builtin.cc
 *
 *  builtin intrinsic operators specific to Trainium target.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/runtime/base.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op_attr_types.h>

#include <initializer_list>
#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

namespace {
static ffi::Array<Var> RegionNoBodyParams(const CallNode*) { return {}; }

void RegisterNKIIntrinsicAliases();
}  // namespace

void RegisterTRNTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  RegisterNKIIntrinsicAliases();

  OpDef("tirx.nki.tensorized_instruction", "Tensorize the NKI instructions in the region body.")
      .signature()
      .set_attr<FRegionGetBodyParams>(tvm::op_attr::kRegionGetBodyParams,
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>(tvm::tirx::op_attr::kDeviceIntrinsicNamespace,
                                           ffi::String("nki"))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.nki.tensorized_instruction"));
}

namespace {

struct NKIIntrinsicNames {
  std::string canonical;
  std::string printer;
};

TVM_FFI_NO_INLINE NKIIntrinsicNames MakeNKIIntrinsicNames(const char* op_name) {
  std::string prefix = "nki_";
  std::string suffix(op_name);
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }
  return {"tirx.nki." + suffix, "tirx.nki." + suffix};
}

TVM_FFI_NO_INLINE void RegisterNKIIntrinsicAttrs(OpDef& def, const std::string& printer_name) {
  def.set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>(tvm::tirx::op_attr::kDeviceIntrinsicNamespace,
                                           ffi::String("nki"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String(printer_name));
}

template <typename... Specs>
void RegisterNKIIntrinsic(const char* op_name, const Specs&... specs) {
  NKIIntrinsicNames names = MakeNKIIntrinsicNames(op_name);
  OpDef def(names.canonical);
  def.signature(specs...);
  RegisterNKIIntrinsicAttrs(def, names.printer);
}

void RegisterNKIIntrinsicAliases() {
  RegisterNKIIntrinsic("nki_activation", sig::arg("result"), sig::arg("data"), sig::arg("opcode"),
                       sig::arg("bias"), sig::arg("scale"));
  RegisterNKIIntrinsic("nki_activation_reduce", sig::arg("reduce_res"), sig::arg("act_res"),
                       sig::arg("data"), sig::arg("opcode"), sig::arg("reduce_opcode"),
                       sig::arg("bias"), sig::arg("scale"));
  RegisterNKIIntrinsic("nki_affine_select", sig::arg("result"), sig::arg("pred"),
                       sig::arg("true_value"), sig::arg("false_value"));
  RegisterNKIIntrinsic("nki_identity", sig::arg("result"), sig::arg("size"));
  RegisterNKIIntrinsic("nki_load", sig::arg("res"), sig::arg("data"));
  RegisterNKIIntrinsic("nki_matmul", sig::arg("res"), sig::arg("lhs"), sig::arg("rhs"),
                       sig::arg("accum"));
  RegisterNKIIntrinsic("nki_memset", sig::arg("result"), sig::arg("value"));
  RegisterNKIIntrinsic("nki_reciprocal", sig::arg("result"), sig::arg("data"));
  RegisterNKIIntrinsic("nki_scalar_tensor_scalar", sig::arg("result"), sig::arg("data"),
                       sig::arg("operand0"), sig::arg("operand1"), sig::arg("opcode0"),
                       sig::arg("opcode1"), sig::arg("reverse0"), sig::arg("reverse1"));
  RegisterNKIIntrinsic("nki_scalar_tensor_tensor", sig::arg("result"), sig::arg("data"),
                       sig::arg("operand0"), sig::arg("operand1"), sig::arg("opcode0"),
                       sig::arg("opcode1"), sig::arg("reverse0"), sig::arg("reverse1"));
  RegisterNKIIntrinsic("nki_store", sig::arg("res"), sig::arg("data"));
  RegisterNKIIntrinsic("nki_tensor_copy", sig::arg("res"), sig::arg("data"));
  RegisterNKIIntrinsic("nki_tensorreduce", sig::arg("result"), sig::arg("data"), sig::arg("opcode"),
                       sig::arg("negate"), sig::var_args("args"));
  RegisterNKIIntrinsic("nki_tensorscalar", sig::arg("result"), sig::arg("operand0"),
                       sig::arg("operand1"), sig::arg("opcode"), sig::arg("reverse"));
  RegisterNKIIntrinsic("nki_tensorscalar_reduce", sig::arg("reduce_res"),
                       sig::arg("tensorscalar_res"), sig::arg("operand0"), sig::arg("operand1"),
                       sig::arg("opcode"), sig::arg("reduce_opcode"), sig::arg("reverse"));
  RegisterNKIIntrinsic("nki_tensortensor", sig::arg("result"), sig::arg("operand0"),
                       sig::arg("operand1"), sig::arg("opcode"));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterTRNTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
