
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
#include <tvm/runtime/base.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

namespace {
void RegisterNKIIntrinsicAliases();
}

void RegisterTRNTargetBuiltins() {
  // clang-format off
static bool registered = false;
if (registered) return;
registered = true;

RegisterNKIIntrinsicAliases();
  // clang-format on
}

namespace {

void RegisterNKIIntrinsic(const char* name) {
  std::string prefix = "nki_";
  std::string suffix(name);
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string canonical_op_name = "tirx.nki." + suffix;
  ffi::String namespace_attr("nki");
  ffi::String printer_name("nki." + suffix);
  int64_t effect = static_cast<int64_t>(CallEffectKind::kOpaque);

  auto register_one = [&](const std::string& op_name) {
    OpRegEntry::RegisterOrGet(op_name)
        .set_name()
        .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"),
                                  /*plevel=*/15)
        .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr,
                                             /*plevel=*/15)
        .set_attr<TCallEffectKind>("TCallEffectKind", effect, /*plevel=*/15)
        .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name, /*plevel=*/15);
  };

  register_one(canonical_op_name);
}

const char* kNKIIntrinsics[] = {
    "nki_activation",
    "nki_activation_reduce",
    "nki_affine_select",
    "nki_identity",
    "nki_load",
    "nki_matmul",
    "nki_memset",
    "nki_reciprocal",
    "nki_scalar_tensor_scalar",
    "nki_scalar_tensor_tensor",
    "nki_store",
    "nki_tensor_copy",
    "nki_tensorreduce",
    "nki_tensorscalar",
    "nki_tensorscalar_reduce",
    "nki_tensortensor",
};

void RegisterNKIIntrinsicAliases() {
  for (const char* op_name : kNKIIntrinsics) {
    RegisterNKIIntrinsic(op_name);
  }
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterTRNTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
