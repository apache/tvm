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
 * \file tirx/op/tile.cc
 * \brief Attribute schemas and validation for tensor instructions.
 */

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

struct ScopeAttrs : public AttrsNode {
  ffi::String scope;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<ScopeAttrs>().def_ro("scope", &ScopeAttrs::scope,
                                         refl::DefaultValue(ffi::String("thread")));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.ScopeAttrs", ScopeAttrs, AttrsNode);
};

struct MemoryAttrs : public AttrsNode {
  ffi::String scope;
  int64_t vec_bits;
  ffi::Optional<ffi::String> cache;
  ffi::Optional<ffi::String> l1_evict;
  ffi::Optional<ffi::String> l2_evict;
  ffi::Optional<ffi::String> prefetch_size;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<MemoryAttrs>()
        .def_ro("scope", &MemoryAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("vec_bits", &MemoryAttrs::vec_bits, refl::DefaultValue(int64_t(0)))
        .def_ro("cache", &MemoryAttrs::cache,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("l1_evict", &MemoryAttrs::l1_evict,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("l2_evict", &MemoryAttrs::l2_evict,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("prefetch_size", &MemoryAttrs::prefetch_size,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.MemoryAttrs", MemoryAttrs, AttrsNode);
};

struct AsyncAttrs : public AttrsNode {
  ffi::String scope;
  bool direct;
  ffi::String fill_mode;
  int64_t prefetch_size;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<AsyncAttrs>()
        .def_ro("scope", &AsyncAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("direct", &AsyncAttrs::direct, refl::DefaultValue(false))
        .def_ro("fill_mode", &AsyncAttrs::fill_mode, refl::DefaultValue(ffi::String("")))
        .def_ro("prefetch_size", &AsyncAttrs::prefetch_size, refl::DefaultValue(int64_t(-1)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.AsyncAttrs", AsyncAttrs, AttrsNode);
};

struct TMAAttrs : public AttrsNode {
  ffi::String scope;
  ffi::String descriptor_mode;
  int64_t cta_group;
  ffi::String cache_hint;
  ffi::Optional<ffi::String> tma_dtype;
  ffi::Optional<int64_t> oob;
  bool prefetch_tensormap;
  ffi::Optional<int64_t> tensormap_l2_promotion;
  ffi::Optional<ffi::String> reduce_op;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TMAAttrs>()
        .def_ro("scope", &TMAAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("descriptor_mode", &TMAAttrs::descriptor_mode,
                refl::DefaultValue(ffi::String("auto")))
        .def_ro("cta_group", &TMAAttrs::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("cache_hint", &TMAAttrs::cache_hint, refl::DefaultValue(ffi::String("")))
        .def_ro("tma_dtype", &TMAAttrs::tma_dtype,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("oob", &TMAAttrs::oob, refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("prefetch_tensormap", &TMAAttrs::prefetch_tensormap, refl::DefaultValue(false))
        .def_ro("tensormap_l2_promotion", &TMAAttrs::tensormap_l2_promotion,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("reduce_op", &TMAAttrs::reduce_op,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TMAAttrs", TMAAttrs, AttrsNode);
};

struct TcCopyAttrs : public AttrsNode {
  ffi::String scope;
  ffi::Optional<ffi::String> shape;
  ffi::Optional<ffi::String> multicast;
  int64_t cta_group;
  ffi::Optional<ffi::String> decompress;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TcCopyAttrs>()
        .def_ro("scope", &TcCopyAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("shape", &TcCopyAttrs::shape,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("multicast", &TcCopyAttrs::multicast,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("cta_group", &TcCopyAttrs::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("decompress", &TcCopyAttrs::decompress,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TcCopyAttrs", TcCopyAttrs, AttrsNode);
};

struct TcMmaAttrs : public AttrsNode {
  ffi::String scope;
  int64_t cta_group;
  ffi::Optional<int64_t> mma_m;
  ffi::Optional<int64_t> mma_n;
  ffi::String smem_desc;
  bool is_AB_tf32;
  ffi::Optional<bool> weight_stationary;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TcMmaAttrs>()
        .def_ro("scope", &TcMmaAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("cta_group", &TcMmaAttrs::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("mma_m", &TcMmaAttrs::mma_m,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("mma_n", &TcMmaAttrs::mma_n,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("smem_desc", &TcMmaAttrs::smem_desc, refl::DefaultValue(ffi::String("hoist")))
        .def_ro("is_AB_tf32", &TcMmaAttrs::is_AB_tf32, refl::DefaultValue(false))
        .def_ro("weight_stationary", &TcMmaAttrs::weight_stationary,
                refl::DefaultValue(ffi::Optional<bool>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TcMmaAttrs", TcMmaAttrs, AttrsNode);
};

struct MathAttrs : public AttrsNode {
  ffi::String scope;
  ffi::Optional<ffi::String> rounding_mode;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<MathAttrs>()
        .def_ro("scope", &MathAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("rounding_mode", &MathAttrs::rounding_mode,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.MathAttrs", MathAttrs, AttrsNode);
};

struct TrnAttrs : public AttrsNode {
  ffi::String scope;
  ffi::Optional<int64_t> max_inst_size;
  ffi::String opcode;
  ffi::String reduce_op;
  ffi::String op0;
  ffi::String op1;
  ffi::Array<int64_t> axes;
  bool negate;
  bool reverse1;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TrnAttrs>()
        .def_ro("scope", &TrnAttrs::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("max_inst_size", &TrnAttrs::max_inst_size,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("opcode", &TrnAttrs::opcode, refl::DefaultValue(ffi::String("")))
        .def_ro("reduce_op", &TrnAttrs::reduce_op, refl::DefaultValue(ffi::String("sum")))
        .def_ro("op0", &TrnAttrs::op0, refl::DefaultValue(ffi::String("")))
        .def_ro("op1", &TrnAttrs::op1, refl::DefaultValue(ffi::String("")))
        .def_ro("axes", &TrnAttrs::axes, refl::DefaultValue(ffi::Array<int64_t>()))
        .def_ro("negate", &TrnAttrs::negate, refl::DefaultValue(false))
        .def_ro("reverse1", &TrnAttrs::reverse1, refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TrnAttrs", TrnAttrs, AttrsNode);
};

ffi::Expected<void> ValidateTensorInstruction(const CallNode* call) noexcept try {
  auto op = call->op.as_or_throw<Op>();
  TVM_FFI_CHECK_EQ(call->args.size(), op->args_info.size(), TypeError)
      << op->name << " has an invalid operand count";
  TVM_FFI_CHECK(call->ty == PrimType::Void(), TypeError) << op->name << " must return void";
  TVM_FFI_CHECK(call->ty_args.empty(), TypeError) << op->name << " does not take type arguments";
  TVM_FFI_CHECK(call->attrs.defined() && call->attrs->GetTypeKey() == op->attrs_type_key, TypeError)
      << op->name << " requires its registered static attributes";
  static const auto& validators = Op::GetAttrMap<ffi::Function>("FTensorCallValidate");
  validators[op](ffi::GetRef<Call>(call));
  return {};
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ScopeAttrs::RegisterReflection();
  MemoryAttrs::RegisterReflection();
  AsyncAttrs::RegisterReflection();
  TMAAttrs::RegisterReflection();
  TcCopyAttrs::RegisterReflection();
  TcMmaAttrs::RegisterReflection();
  MathAttrs::RegisterReflection();
  TrnAttrs::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tirx.ConfigureTensorInstruction", [](Op op, ffi::Function validator) {
        OpDef(op->name)
            .set_attr("FTensorCallValidate", validator)
            .set_validator(ffi::reflection::NativeFunctionView<void(
                               const CallNode*)>::FromNative<&ValidateTensorInstruction>(),
                           true);
      });
}

}  // namespace tirx
}  // namespace tvm
