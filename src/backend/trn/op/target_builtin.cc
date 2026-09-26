
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
#include <tvm/ir/prim/builtin.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <initializer_list>
#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

namespace {
void RegisterNKIIntrinsicAliases();
}

void RegisterTRNTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  RegisterNKIIntrinsicAliases();
}

namespace {

struct NKIIntrinsicRegistration {
  const char* name;
  std::initializer_list<const char*> args;
  bool allow_extra_args;
};

void RegisterNKIIntrinsic(const NKIIntrinsicRegistration& reg) {
  std::string prefix = "nki_";
  std::string suffix(reg.name);
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string canonical_op_name = "tirx.nki." + suffix;
  ffi::String namespace_attr("nki");
  ffi::String printer_name("nki." + suffix);
  int64_t effect = static_cast<int64_t>(CallEffectKind::kOpaque);

  OpDef def(canonical_op_name);
  for (const char* name : reg.args) {
    def.arg(name, "");
  }
  if (reg.allow_extra_args) {
    def.allow_extra_args();
  }
  def.set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr)
      .set_attr<TCallEffectKind>("TCallEffectKind", effect)
      .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name);
}

const NKIIntrinsicRegistration kNKIIntrinsics[] = {
    {"nki_activation", {"result", "data", "opcode", "bias", "scale"}, false},
    {"nki_activation_reduce",
     {"reduce_res", "act_res", "data", "opcode", "reduce_opcode", "bias", "scale"},
     false},
    {"nki_affine_select", {"result", "pred", "true_value", "false_value"}, false},
    {"nki_identity", {"result", "size"}, false},
    {"nki_load", {"res", "data"}, false},
    {"nki_matmul", {"res", "lhs", "rhs", "accum"}, false},
    {"nki_memset", {"result", "value"}, false},
    {"nki_reciprocal", {"result", "data"}, false},
    {"nki_scalar_tensor_scalar",
     {"result", "data", "operand0", "operand1", "opcode0", "opcode1", "reverse0", "reverse1"},
     false},
    {"nki_scalar_tensor_tensor",
     {"result", "data", "operand0", "operand1", "opcode0", "opcode1", "reverse0", "reverse1"},
     false},
    {"nki_store", {"res", "data"}, false},
    {"nki_tensor_copy", {"res", "data"}, false},
    {"nki_tensorreduce", {"result", "data", "opcode", "negate"}, true},
    {"nki_tensorscalar", {"result", "operand0", "operand1", "opcode", "reverse"}, false},
    {"nki_tensorscalar_reduce",
     {"reduce_res", "tensorscalar_res", "operand0", "operand1", "opcode", "reduce_opcode",
      "reverse"},
     false},
    {"nki_tensortensor", {"result", "operand0", "operand1", "opcode"}, false},
};

void RegisterNKIIntrinsicAliases() {
  for (const auto& reg : kNKIIntrinsics) {
    RegisterNKIIntrinsic(reg);
  }
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterTRNTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
