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
 * \file tirx/op/builtin.cc
 *
 *  builtin intrinsic operators.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/cpu.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op/vector.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace tirx {

static ffi::Array<Var> RegionNoBodyParams(const CallNode*) { return {}; }

void CallFFIKernelAttr::RegisterReflection() {
  ffi::reflection::ObjectDef<CallFFIKernelAttr>().def_ro("launch_params",
                                                         &CallFFIKernelAttr::launch_params);
}

ffi::Array<Var> LaunchThreadBodyParams(const CallNode* call) {
  auto tag = call->args[0].as_or_throw<StringImm>();
  PrimType dtype = call->args[1].as_or_throw<IntExpr>().ty();
  TVM_FFI_CHECK(!tag->value.empty(), ValueError)
      << "launch_thread expects a nonempty StringImm thread tag";
  TVM_FFI_CHECK_GT(dtype.bits(), 1, ValueError)
      << "launch_thread extent must have a scalar integer type wider than one bit";
  return {PrimVar("", dtype)};
}

template <size_t N>
static Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

template <bool Combine>
Type InferTypeVectorPart(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), Combine ? 2U : 1U, ValueError);
  PrimType input = call->args[0]->ty.as_or_throw<PrimType>();
  if constexpr (Combine) {
    TVM_FFI_CHECK(ffi::StructuralEqual()(input, call->args[1]->ty), TypeError)
        << "vectorcombine requires equal input vector types";
  }
  int lanes = input.IsScalableVector() ? input.VScaleFactor() : input.lanes();
  if constexpr (!Combine) {
    TVM_FFI_CHECK(lanes > 1 && lanes % 2 == 0, TypeError)
        << "vector half requires an even lane count";
  }
  int result_lanes = Combine ? lanes * 2 : lanes / 2;
  return input.IsScalableVector()
             ? PrimType::ScalableVector(input.code(), input.bits(), result_lanes)
             : input.WithLanes(result_lanes);
}

template <size_t DataIndex, int ElementIndex>
Type InferTypePointerOffset(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), DataIndex, ValueError);
  Type element_type = PrimType::Void();
  if constexpr (ElementIndex >= 0) {
    TVM_FFI_CHECK_GT(call->args.size(), static_cast<size_t>(ElementIndex), ValueError);
    element_type = call->args[ElementIndex]->ty;
    if (element_type.as<MissingType>()) return Type::Missing();
  }
  auto pointer = call->args[DataIndex]->ty.as<PointerType>();
  return PointerType(element_type, pointer ? pointer.value()->storage_scope : "global");
}

Type InferTypeAccessPointer(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->ty_args.size(), 1U, ValueError);
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  auto pointer = call->args[0]->ty.as<PointerType>();
  return PointerType(call->ty_args[0], pointer ? pointer.value()->storage_scope : "global");
}

Type InferTypeStackAlloca(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 1U, ValueError) << "Stack allocation requires a dtype name";
  ffi::String dtype = call->args[0].as_or_throw<StringImm>()->value;
  if (dtype == "shape") return PointerType(PrimType::Int(64));
  if (dtype == "arg_tcode") return PointerType(PrimType::Int(32));
  if (dtype == "tensormap") return PointerType(TensorMapType());
  return PointerType(PrimType::Void());
}

Type InferTypeAddressOf(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 1U, ValueError) << "Address type requires an object";
  if (const auto* load = call->args[0].as<TensorLoadNode>()) {
    return load->source.as_or_throw<TensorVar>().DataPointerType();
  }
  Var variable = call->args[0].as_or_throw<Var>();
  if (auto pointer = variable->ty.as<PointerType>();
      pointer && pointer.value()->element_type.as<TensorMapType>()) {
    return PrimType::UInt(64);
  }
  return PointerType(variable->ty.as_or_throw<PrimType>());
}

Type InferTypeMaskedLoad(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 2U, ValueError)
      << "Masked load type requires a buffer and index operands";
  TensorVar buffer = call->args[0].as_or_throw<TensorVar>();
  ffi::Array<PrimExpr> indices;
  for (size_t i = 1; i + 1 < call->args.size(); ++i) {
    indices.push_back(call->args[i].as_or_throw<PrimExpr>());
  }
  // Ordinary load typing computes vector elements and scalable index lanes.
  return MakeTensorLoad(buffer, indices).ty();
}

ffi::Expected<Type> InferTypeTensorDataPtr(const CallNode* call) noexcept try {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError)
      << "tirx.tensor_data_ptr expects one TensorVar argument";
  Type inferred = call->args[0].as_or_throw<TensorVar>().DataPointerType();
  if (call->ty.same_as(inferred) || ffi::StructuralEqual()(call->ty, inferred)) return call->ty;
  return inferred;
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

// Essential buffer properties follow operands; the supplied result type retains layout metadata.
template <int shape_index>
ffi::Expected<Type> InferTypeBuffer(const CallNode* call) noexcept try {
  TVM_FFI_CHECK_EQ(call->args.size(), shape_index + 3U, ValueError);
  tvm::Tuple shape = call->args[shape_index].as_or_throw<tvm::Tuple>();
  DLDataType dtype = call->args[shape_index + 1].as_or_throw<DataTypeImm>()->value;
  ffi::String scope = call->args[shape_index + 2].as_or_throw<StringImm>()->value;
  auto original = call->ty.as_or_throw<TensorType>();
  if (ffi::StructuralEqual()(shape->fields, original->shape) && dtype == original->dtype->dtype &&
      scope == original->storage_scope) {
    return original;
  }
  auto inferred = ffi::make_object<TensorTypeNode>(*original.get());
  inferred->shape =
      shape->fields.Map([](const Expr& extent) { return extent.as_or_throw<PrimExpr>(); });
  inferred->dtype = PrimType(dtype);
  inferred->storage_scope = scope;
  return TensorType(std::move(inferred));
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

ffi::Expected<void> ValidateDeclTensor(const CallNode* call) noexcept try {
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  auto buffer = call->ty.as_or_throw<TensorType>();
  ffi::String scope = call->args[3].as_or_throw<StringImm>()->value;
  if (scope == "tmem") {
    TVM_FFI_CHECK_EQ(buffer->allocated_addr.size(), 1U, ValueError)
        << "For `tmem` scope, decl_tensor requires exactly one `allocated_addr` PrimExpr";
  } else if (scope.empty() || scope == "global" || scope == "shared" || scope == "shared.dyn" ||
             scope == "local") {
    TVM_FFI_CHECK(buffer->allocated_addr.empty(), ValueError)
        << "For `" << scope << "` scope, decl_tensor does not accept `allocated_addr`";
  }
  return {};
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

void ValidateBuiltinRegion(const RegionStmtNode* region) {
  TVM_FFI_CHECK(region->result_vars.empty() && region->attrs->dict.empty(), ValueError)
      << region->op->name << " expects no results or attributes";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  CallFFIKernelAttr::RegisterReflection();
  ffi::reflection::GlobalDef().def("tirx.CallFFIKernelAttr",
                                   [](ffi::Array<ffi::String> launch_params) {
                                     auto attrs = ffi::make_object<CallFFIKernelAttr>();
                                     attrs->launch_params = std::move(launch_params);
                                     return Attrs(attrs);
                                   });

  // Script metadata extends the canonical primitive operators registered by IR.

  OpDef("prim.likely")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.likely"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.if_then_else")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.if_then_else"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.vscale")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vscale"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.ceil")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ceil"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.clz")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.clz"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.pow")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.pow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.fabs")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fabs"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.fmod")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fmod"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.floor")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.floor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.round")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.round"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.nearbyint")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nearbyint"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.trunc")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.trunc"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.exp")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.exp2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.exp10")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.erf")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.erf"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.tanh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.sigmoid")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sigmoid"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.sqrt")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.rsqrt")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.rsqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.log")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.log1p")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log1p"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.log10")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.tan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.cos")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.cosh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.sin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.sinh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.asin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.acos")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.atan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.acosh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.asinh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.atanh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.atan2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.nextafter")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nextafter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.hypot")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.hypot"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.copysign")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.copysign"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.ldexp")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ldexp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.isnan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.isnan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.popcount")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.popcount"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.fma")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("prim.assume")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.assume"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(reinterpret_op, "tirx.reinterpret")
TVM_DEFINE_CACHED_OP_GETTER(gpu_thread_return_op, "tirx.gpu_thread_return")
TVM_DEFINE_CACHED_OP_GETTER(gpu_thread_filter_op, "tirx.gpu_thread_filter")
TVM_DEFINE_CACHED_OP_GETTER(gpu_active_thread_selector_op, "tirx.gpu_active_thread_selector")
TVM_DEFINE_CACHED_OP_GETTER(address_of_op, "tirx.address_of")
TVM_DEFINE_CACHED_OP_GETTER(isnullptr_op, "tirx.isnullptr")
TVM_DEFINE_CACHED_OP_GETTER(call_extern_op, "tirx.call_extern")
TVM_DEFINE_CACHED_OP_GETTER(call_pure_extern_op, "tirx.call_pure_extern")
TVM_DEFINE_CACHED_OP_GETTER(call_llvm_intrin_op, "tirx.call_llvm_intrin")
TVM_DEFINE_CACHED_OP_GETTER(call_llvm_pure_intrin_op, "tirx.call_llvm_pure_intrin")
TVM_DEFINE_CACHED_OP_GETTER(call_spirv_pure_glsl450_op, "tirx.call_spirv_pure_glsl450")
TVM_DEFINE_CACHED_OP_GETTER(prefetch_op, "tirx.prefetch")
TVM_DEFINE_CACHED_OP_GETTER(access_ptr_op, "tirx.access_ptr")
TVM_DEFINE_CACHED_OP_GETTER(ptr_byte_offset_op, "tirx.ptr_byte_offset")
TVM_DEFINE_CACHED_OP_GETTER(static_handle_op, "tirx.static_handle")
TVM_DEFINE_CACHED_OP_GETTER(handle_add_byte_offset_op, "tirx.handle_add_byte_offset")
TVM_DEFINE_CACHED_OP_GETTER(abi_field_get_op, "tirx.abi_field_get")
TVM_DEFINE_CACHED_OP_GETTER(abi_field_set_op, "tirx.abi_field_set")
TVM_DEFINE_CACHED_OP_GETTER(throw_last_error_op, "tirx.throw_last_error")
TVM_DEFINE_CACHED_OP_GETTER(stack_alloca_op, "tirx.stack_alloca")
TVM_DEFINE_CACHED_OP_GETTER(stack_make_shape_op, "tirx.stack_make_shape")
TVM_DEFINE_CACHED_OP_GETTER(stack_make_dltensor_op, "tirx.stack_make_dltensor")
TVM_DEFINE_CACHED_OP_GETTER(call_packed_op, "tirx.call_packed")
TVM_DEFINE_CACHED_OP_GETTER(call_ffi_kernel_op, "tirx.call_ffi_kernel")
TVM_DEFINE_CACHED_OP_GETTER(call_cpacked_op, "tirx.call_cpacked")
TVM_DEFINE_CACHED_OP_GETTER(gpu_thread_invariant_op, "tirx.gpu_thread_invariant")
TVM_DEFINE_CACHED_OP_GETTER(call_packed_lowered_op, "tirx.call_packed_lowered")
TVM_DEFINE_CACHED_OP_GETTER(call_cpacked_lowered_op, "tirx.call_cpacked_lowered")
TVM_DEFINE_CACHED_OP_GETTER(gpu_storage_sync_op, "tirx.gpu_storage_sync")
TVM_DEFINE_CACHED_OP_GETTER(cpu_parallel_barrier_op, "tirx.cpu_parallel_barrier")
TVM_DEFINE_CACHED_OP_GETTER(kernel_replace_point_op, "tirx.kernel_replace_point")
TVM_DEFINE_CACHED_OP_GETTER(gpu_warp_shuffle_op, "tirx.gpu_warp_shuffle")
TVM_DEFINE_CACHED_OP_GETTER(gpu_warp_shuffle_up_op, "tirx.gpu_warp_shuffle_up")
TVM_DEFINE_CACHED_OP_GETTER(gpu_warp_shuffle_down_op, "tirx.gpu_warp_shuffle_down")
TVM_DEFINE_CACHED_OP_GETTER(gpu_warp_shuffle_xor_op, "tirx.gpu_warp_shuffle_xor")
TVM_DEFINE_CACHED_OP_GETTER(gpu_warp_activemask_op, "tirx.gpu_warp_activemask")
TVM_DEFINE_CACHED_OP_GETTER(gpu_thread_allreduce_op, "tirx.gpu_thread_allreduce")
TVM_DEFINE_CACHED_OP_GETTER(vectorhigh_op, "tirx.vectorhigh")
TVM_DEFINE_CACHED_OP_GETTER(vectorlow_op, "tirx.vectorlow")
TVM_DEFINE_CACHED_OP_GETTER(vectorcombine_op, "tirx.vectorcombine")
TVM_DEFINE_CACHED_OP_GETTER(gpu_dp4a_op, "tirx.gpu_dp4a")
TVM_DEFINE_CACHED_OP_GETTER(atomic_add_op, "tirx.atomic_add")
TVM_DEFINE_CACHED_OP_GETTER(assume_aligned_op, "tirx.assume_aligned")
TVM_DEFINE_CACHED_OP_GETTER(undef_op, "tirx.undef")
TVM_DEFINE_CACHED_OP_GETTER(get_active_lane_mask_op, "tirx.get_active_lane_mask")
TVM_DEFINE_CACHED_OP_GETTER(masked_load_op, "tirx.masked_load")
TVM_DEFINE_CACHED_OP_GETTER(masked_store_op, "tirx.masked_store")
TVM_DEFINE_CACHED_OP_GETTER(ignore_loop_partition_op, "tirx.ignore_loop_partition")
TVM_DEFINE_CACHED_OP_GETTER(alloc_tensor_op, "tirx.alloc_tensor")
TVM_DEFINE_CACHED_OP_GETTER(decl_tensor_op, "tirx.decl_tensor")
TVM_DEFINE_CACHED_OP_GETTER(tensor_data_ptr_op, "tirx.tensor_data_ptr")

// Region operations.
TVM_DEFINE_CACHED_OP_GETTER(launch_thread_op, "tirx.launch_thread")
TVM_DEFINE_CACHED_OP_GETTER(device_entry_op, "tirx.device_entry")
TVM_DEFINE_CACHED_OP_GETTER(device_context_op, "tirx.device_context")
TVM_DEFINE_CACHED_OP_GETTER(compute_scope_op, "tirx.compute_scope")
TVM_DEFINE_CACHED_OP_GETTER(parallel_launch_op, "tirx.parallel_launch")

#undef TVM_DEFINE_CACHED_OP_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.reinterpret")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.reinterpret"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.gpu_thread_return")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_thread_return"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kControlJump));

  // tirx.gpu_thread_filter: escape hatch for non-canonical thread-set filter predicates
  // used as an If condition. (var, cond) -- ``var`` names the
  // active-set axis the compiler should collapse to a singleton if it cannot
  // statically analyze ``cond``. Canonical predicates (see
  // ``analysis/filter_canonical.h``) should appear bare in ``if`` conditions
  // without this wrapper.

  OpDef("tirx.gpu_thread_filter")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Bool())
      .signature(sig::arg<PrimVar>("var", "The thread-axis variable."),
                 sig::arg<PrimExpr>("pred", "The predicate."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_thread_filter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.gpu_active_thread_selector")
      .signature(sig::arg<PrimVar>("var", "The thread-axis variable."),
                 sig::arg<PrimExpr>("pred", "The predicate."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.gpu_active_thread_selector"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.address_of")
      .signature(sig::arg("obj", "The referenced object."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeAddressOf>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.address_of"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.isnullptr")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Bool())
      .signature(sig::arg("x", "The input value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.isnullptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.call_extern")
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.call_pure_extern")
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_pure_extern"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.call_llvm_intrin")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_llvm_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.call_llvm_pure_intrin")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_llvm_pure_intrin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.call_spirv_pure_glsl450")
      .signature(sig::arg<IntExpr>("intrin_id", "The intrinsic identifier."),
                 sig::var_args<PrimExpr>("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.prefetch")
      .signature(sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("rw", "The read/write mode."),
                 sig::arg<IntExpr>("locality", "The locality hint."),
                 sig::arg<IntExpr>("cache_type", "The cache policy."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.access_ptr")
      .signature(sig::ty_arg<PrimType>("access_dtype", "The accessed element type."),
                 sig::arg("data", "The input data."), sig::arg<IntExpr>("offset", "The offset."),
                 sig::arg<IntExpr>("extent", "The extent."),
                 sig::arg<IntExpr>("rw_mask", "The read/write mask."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.access_ptr"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeAccessPointer>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.ptr_byte_offset")
      .signature(sig::arg("data", "Base pointer."),
                 sig::arg<IntExpr>("byte_offset", "Offset in bytes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ptr_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.static_handle")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));

  OpDef("tirx.handle_add_byte_offset")
      .signature(sig::arg("handle", "The handle."), sig::arg<IntExpr>("offset", "The offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.handle_add_byte_offset"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypePointerOffset<0, -1>>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.abi_field_get")
      .signature(sig::arg("arr", "The array."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg<IntExpr>("field", "The field index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.abi_field_get"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.abi_field_set")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("arr", "The array."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg<IntExpr>("field", "The field index."),
                 sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.abi_field_set"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.throw_last_error")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.throw_last_error"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.stack_alloca")
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeStackAlloca>())
      .signature(sig::arg("dtype_str", "The data type name."),
                 sig::arg<IntExpr>("num", "The number of entries."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.stack_alloca"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.stack_make_shape")
      .set_attr<TFixedReturnType>("TFixedReturnType", PointerType(PrimType::Int(64)))
      .signature(sig::var_args<IntExpr>("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.stack_make_shape"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

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

  OpDef("tirx.call_packed")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_packed"));

  OpDef("tirx.call_ffi_kernel")
      .signature(sig::arg("kernel", "The kernel."), sig::var_args("args"),
                 sig::call_attrs<CallFFIKernelAttr>())
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_ffi_kernel"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.call_cpacked")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_cpacked"));

  OpDef("tirx.gpu_thread_invariant")
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .signature(sig::arg<PrimExpr>("cond", "The condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_thread_invariant"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.call_packed_lowered")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."),
                 sig::arg("args_stack", "The argument stack."),
                 sig::arg<IntExpr>("begin", "The start index."),
                 sig::arg<IntExpr>("end", "The end index."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_packed_lowered"));

  OpDef("tirx.call_cpacked_lowered")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("func_name", "The function name."),
                 sig::arg("args_stack", "The argument stack."),
                 sig::arg<IntExpr>("begin", "The start index."),
                 sig::arg<IntExpr>("end", "The end index."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.call_cpacked_lowered"));

  // TODO(tvm-team) revisit storage sync once we have a good memory hierachy structure.

  OpDef("tirx.gpu_storage_sync")
      .signature(sig::arg<StringImm>("storage_scope", "The storage scope."))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_storage_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cpu_parallel_barrier")
      .signature()
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cpu_parallel_barrier"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.kernel_replace_point")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.kernel_replace_point"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_warp_shuffle")
      .signature(sig::arg<IntExpr>("mask", "The mask."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<IntExpr>("warp_id", "The warp identifier."),
                 sig::arg<IntExpr>("width", "The width."),
                 sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_warp_shuffle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_warp_shuffle_up")
      .signature(
          sig::arg<IntExpr>("mask", "The mask."), sig::arg<PrimExpr>("value", "The value to use."),
          sig::arg<IntExpr>("offset", "The offset."), sig::arg<IntExpr>("width", "The width."),
          sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_warp_shuffle_up"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_warp_shuffle_down")
      .signature(
          sig::arg<IntExpr>("mask", "The mask."), sig::arg<PrimExpr>("value", "The value to use."),
          sig::arg<IntExpr>("offset", "The offset."), sig::arg<IntExpr>("width", "The width."),
          sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_warp_shuffle_down"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_warp_shuffle_xor")
      .signature(sig::arg<IntExpr>("mask", "The mask."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<IntExpr>("lane_mask", "The lane mask."),
                 sig::arg<IntExpr>("width", "The width."),
                 sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_warp_shuffle_xor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_warp_activemask")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_warp_activemask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_thread_allreduce")
      .signature(sig::arg<LambdaExpr>("combine", "The typed combining lambda."),
                 sig::arg<Expr>("identity", "The identity values."),
                 sig::arg<Expr>("values", "The reduction values."),
                 sig::arg<PrimExpr>("predicate", "Whether this thread contributes."),
                 sig::arg<Expr>("destinations", "The destination tensor loads."),
                 sig::arg<Expr>("thread_axes", "The reduction thread axes."))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_thread_allreduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.vectorhigh")
      .signature(sig::arg<PrimExpr>("vec", "The input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<false>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorhigh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.vectorlow")
      .signature(sig::arg<PrimExpr>("vec", "The input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<false>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorlow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.vectorcombine")
      .signature(sig::arg<PrimExpr>("vec1", "The first input vector."),
                 sig::arg<PrimExpr>("vec2", "The second input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<true>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorcombine"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.gpu_load_matrix_sync")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
          sig::arg<IntExpr>("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_load_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.gpu_mma_sync")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("fragment_d", "The D fragment."),
                 sig::arg<IntExpr>("index_d", "The D fragment index."),
                 sig::arg("fragment_a", "The A fragment."),
                 sig::arg<IntExpr>("index_a", "The A fragment index."),
                 sig::arg("fragment_b", "The B fragment."),
                 sig::arg<IntExpr>("index_b", "The B fragment index."),
                 sig::arg("fragment_c", "The C fragment."),
                 sig::arg<IntExpr>("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_mma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_fill_fragment")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_fill_fragment"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_store_matrix_sync")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
          sig::arg<IntExpr>("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_store_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.gpu_dp4a")
      .signature(sig::arg<PrimExpr>("vec1", "The first input vector."),
                 sig::arg<PrimExpr>("vec2", "The second input vector."),
                 sig::arg<PrimExpr>("acc", "The accumulator."))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.gpu_dp4a"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.atomic_add")
      .signature(sig::arg("ptr", "The pointer."), sig::arg("value", "The value to use."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.assume_aligned")
      .signature(sig::arg<TensorVar>("tensor", "The tensor whose base address is aligned."),
                 sig::arg<IntImm>("alignment_bytes", "The constant byte alignment."))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.assume_aligned"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kEmbedInfo));

  OpDef("tirx.undef")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.undef"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.get_active_lane_mask")
      .signature(sig::arg<IntExpr>("base", "The base value."),
                 sig::arg<IntExpr>("limit", "The limit value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.get_active_lane_mask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.masked_load")
      .signature(sig::arg<TensorVar>("buffer", "The buffer."),
                 sig::arg<PrimExpr>("index", "The index."), sig::var_args<PrimExpr>("args"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeMaskedLoad>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.masked_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.masked_store")
      .signature(sig::arg<TensorVar>("buffer", "The buffer."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<PrimExpr>("index", "The index."), sig::var_args<PrimExpr>("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.masked_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));

  OpDef("tirx.ignore_loop_partition")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Bool())
      .signature(sig::arg<PrimExpr>("predicate", "The predicate."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ignore_loop_partition"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.alloc_tensor")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeBuffer<0>>())
      .add_arg("shape", "The tuple of buffer extents.")
      .add_arg("dtype", "The buffer data type.")
      .add_arg("scope", "The storage scope.")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.decl_tensor")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeBuffer<1>>())
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateDeclTensor>())
      .add_arg("data", "The existing data pointer.")
      .add_arg("shape", "The tuple of buffer extents.")
      .add_arg("dtype", "The buffer data type.")
      .add_arg("scope", "The storage scope.")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tensor_data_ptr")
      .signature(sig::arg("tensor", "The tensor variable."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeTensorDataPtr>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  // Region operations.
  OpDef("tirx.device_entry", "Mark a device entry containing scope definitions.")
      .signature()
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("tirx.launch_thread", "Bind a thread index within a body with a launch extent.")
      .signature(sig::arg<StringImm>("tag"), sig::arg<IntExpr>("extent"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&LaunchThreadBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("tirx.device_context", "Supply the device type and ID within a region.")
      .signature(sig::arg<IntExpr>("device_type"), sig::arg<IntExpr>("device_id"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("tirx.compute_scope", "Outline a named CPU compute region.")
      .signature(sig::arg<StringImm>("name"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
  OpDef("tirx.parallel_launch", "Launch a CPU worker team around a region.")
      .signature()
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

}  // namespace tirx
}  // namespace tvm
