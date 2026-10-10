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
 * \file tirx/op/abi.cc
 * \brief TIRx abi operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace tirx {

void CallFFIKernelAttr::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<CallFFIKernelAttr>()
      .def_ro("launch_params", &CallFFIKernelAttr::launch_params,
              refl::DefaultValue(ffi::Array<ffi::String>()))
      .def_ro("launch_fields", &CallFFIKernelAttr::launch_fields,
              refl::DefaultValue(ffi::Array<ffi::String>()))
      .def_ro("num_kernel_args", &CallFFIKernelAttr::num_kernel_args, refl::DefaultValue(-1))
      .def_ro("kernel_attrs", &CallFFIKernelAttr::kernel_attrs,
              refl::DefaultValue(ffi::Map<ffi::String, int64_t>()));
}

Type InferTypeStackAlloca(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 1U, ValueError) << "Stack allocation requires a dtype name";
  ffi::String dtype = call->args[0].as_or_throw<StringImm>()->value;
  if (dtype == "shape") return PointerType(PrimType::Int(64));
  if (dtype == "arg_tcode") return PointerType(PrimType::Int(32));
  if (dtype == "tensormap") return PointerType(TensorMapType());
  return PointerType(PrimType::Void());
}

const Op& call_extern_op() {
  static const Op op = Op::Get("tirx.call_extern");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_extern")
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& call_pure_extern_op() {
  static const Op op = Op::Get("tirx.call_pure_extern");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_pure_extern")
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_pure_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& call_llvm_intrin_op() {
  static const Op op = Op::Get("tirx.call_llvm_intrin");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_llvm_intrin")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_llvm_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& call_llvm_pure_intrin_op() {
  static const Op op = Op::Get("tirx.call_llvm_pure_intrin");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_llvm_pure_intrin")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_llvm_pure_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);
}

const Op& call_spirv_pure_glsl450_op() {
  static const Op op = Op::Get("tirx.call_spirv_pure_glsl450");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_spirv_pure_glsl450")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."),
                 sig::var_args<PrimExpr>("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& abi_field_get_op() {
  static const Op op = Op::Get("tirx.abi_field_get");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.abi_field_get")
      .signature(sig::arg("arr", "The array."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg<IntExpr>("field", "The field index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.abi_field_get"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));
}

const Op& abi_field_set_op() {
  static const Op op = Op::Get("tirx.abi_field_set");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.abi_field_set")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("arr", "The array."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg<IntExpr>("field", "The field index."),
                 sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.abi_field_set"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));
}

const Op& throw_last_error_op() {
  static const Op op = Op::Get("tirx.throw_last_error");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.throw_last_error")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.throw_last_error"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& stack_alloca_op() {
  static const Op op = Op::Get("tirx.stack_alloca");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.stack_alloca")
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeStackAlloca>())
      .signature(sig::arg("dtype_str", "The data type name."),
                 sig::arg<IntExpr>("num", "The number of entries."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.stack_alloca"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& stack_make_shape_op() {
  static const Op op = Op::Get("tirx.stack_make_shape");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.stack_make_shape")
      .set_attr<TFixedReturnType>("TFixedReturnType", PointerType(PrimType::Int(64)))
      .signature(sig::var_args<IntExpr>("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.stack_make_shape"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& stack_make_dltensor_op() {
  static const Op op = Op::Get("tirx.stack_make_dltensor");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.stack_make_dltensor")
      .set_attr<TFixedReturnType>("TFixedReturnType", PointerType(PrimType::Void()))
      .signature(sig::arg("data", "The input data."), sig::arg("shape", "The shape."),
                 sig::arg("strides", "The strides."),
                 sig::arg<IntExpr>("ndim", "The number of dimensions."),
                 sig::arg("arr_dtype", "The array data type."),
                 sig::arg<IntExpr>("elem_offset", "The element offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.stack_make_dltensor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& call_packed_op() {
  static const Op op = Op::Get("tirx.call_packed");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_packed")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_packed"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  CallFFIKernelAttr::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tirx.CallFFIKernelAttr",
      [](ffi::Array<ffi::String> launch_params, ffi::Array<ffi::String> launch_fields,
         int64_t num_kernel_args, ffi::Map<ffi::String, int64_t> kernel_attrs) {
        auto attrs = ffi::make_object<CallFFIKernelAttr>();
        attrs->launch_params = std::move(launch_params);
        attrs->launch_fields = std::move(launch_fields);
        attrs->num_kernel_args = num_kernel_args;
        attrs->kernel_attrs = std::move(kernel_attrs);
        return Attrs(attrs);
      });
}

const Op& call_ffi_kernel_op() {
  static const Op op = Op::Get("tirx.call_ffi_kernel");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_ffi_kernel")
      .signature(sig::arg("kernel", "The kernel."), sig::var_args("args"),
                 sig::call_attrs<CallFFIKernelAttr>())
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_ffi_kernel"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& call_cpacked_op() {
  static const Op op = Op::Get("tirx.call_cpacked");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_cpacked")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_cpacked"));
}

const Op& call_packed_lowered_op() {
  static const Op op = Op::Get("tirx.call_packed_lowered");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_packed_lowered")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."),
                 sig::arg("args_stack", "The argument stack."),
                 sig::arg<IntExpr>("begin", "The start index."),
                 sig::arg<IntExpr>("end", "The end index."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_packed_lowered"));
}

const Op& call_cpacked_lowered_op() {
  static const Op op = Op::Get("tirx.call_cpacked_lowered");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_cpacked_lowered")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."),
                 sig::arg("args_stack", "The argument stack."),
                 sig::arg<IntExpr>("begin", "The start index."),
                 sig::arg<IntExpr>("end", "The end index."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_cpacked_lowered"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.alloc_workspace")
      .set_attr<TFixedReturnType>("TFixedReturnType", PointerType(PrimType::Void()))
      .signature(sig::arg<IntExpr>("device_type", "The device type."),
                 sig::arg<IntExpr>("device_id", "The device index."),
                 sig::arg<IntExpr>("nbytes", "The number of bytes."),
                 sig::arg<IntExpr>("dtype_code_hint", "The data type code hint."),
                 sig::arg<IntExpr>("dtype_bits_hint", "The data type bit-width hint."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.alloc_workspace"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendAllocWorkspace")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.free_workspace")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg<IntExpr>("device_type", "The device type."),
                 sig::arg<IntExpr>("device_id", "The device index."),
                 sig::arg("ptr", "The pointer."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.free_workspace"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendFreeWorkspace")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace tirx
}  // namespace tvm
