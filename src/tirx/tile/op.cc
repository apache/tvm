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
 * \file tirx/tile/op.cc
 * \brief Attribute schemas and validation for tensor instructions.
 */

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/tile/op.h>

namespace tvm::tirx::tile {

struct ScopeAttrsNode : public AttrsNode {
  ffi::String scope;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<ScopeAttrsNode>().def_ro("scope", &ScopeAttrsNode::scope,
                                         refl::DefaultValue(ffi::String("thread")));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.ScopeAttrs", ScopeAttrsNode, AttrsNode);
};

struct MemoryAttrsNode : public AttrsNode {
  ffi::String scope;
  int64_t vec_bits;
  ffi::Optional<ffi::String> cache;
  ffi::Optional<ffi::String> l1_evict;
  ffi::Optional<ffi::String> l2_evict;
  ffi::Optional<ffi::String> prefetch_size;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<MemoryAttrsNode>()
        .def_ro("scope", &MemoryAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("vec_bits", &MemoryAttrsNode::vec_bits, refl::DefaultValue(int64_t(0)))
        .def_ro("cache", &MemoryAttrsNode::cache,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("l1_evict", &MemoryAttrsNode::l1_evict,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("l2_evict", &MemoryAttrsNode::l2_evict,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("prefetch_size", &MemoryAttrsNode::prefetch_size,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.MemoryAttrs", MemoryAttrsNode, AttrsNode);
};

struct AsyncAttrsNode : public AttrsNode {
  ffi::String scope;
  bool direct;
  ffi::String fill_mode;
  int64_t prefetch_size;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<AsyncAttrsNode>()
        .def_ro("scope", &AsyncAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("direct", &AsyncAttrsNode::direct, refl::DefaultValue(false))
        .def_ro("fill_mode", &AsyncAttrsNode::fill_mode, refl::DefaultValue(ffi::String("")))
        .def_ro("prefetch_size", &AsyncAttrsNode::prefetch_size, refl::DefaultValue(int64_t(-1)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.AsyncAttrs", AsyncAttrsNode, AttrsNode);
};

struct TMAAttrsNode : public AttrsNode {
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
    refl::ObjectDef<TMAAttrsNode>()
        .def_ro("scope", &TMAAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("descriptor_mode", &TMAAttrsNode::descriptor_mode,
                refl::DefaultValue(ffi::String("auto")))
        .def_ro("cta_group", &TMAAttrsNode::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("cache_hint", &TMAAttrsNode::cache_hint, refl::DefaultValue(ffi::String("")))
        .def_ro("tma_dtype", &TMAAttrsNode::tma_dtype,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("oob", &TMAAttrsNode::oob, refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("prefetch_tensormap", &TMAAttrsNode::prefetch_tensormap, refl::DefaultValue(false))
        .def_ro("tensormap_l2_promotion", &TMAAttrsNode::tensormap_l2_promotion,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("reduce_op", &TMAAttrsNode::reduce_op,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TMAAttrs", TMAAttrsNode, AttrsNode);
};

struct TcCopyAttrsNode : public AttrsNode {
  ffi::String scope;
  ffi::Optional<ffi::String> shape;
  ffi::Optional<ffi::String> multicast;
  int64_t cta_group;
  ffi::Optional<ffi::String> decompress;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TcCopyAttrsNode>()
        .def_ro("scope", &TcCopyAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("shape", &TcCopyAttrsNode::shape,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("multicast", &TcCopyAttrsNode::multicast,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)))
        .def_ro("cta_group", &TcCopyAttrsNode::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("decompress", &TcCopyAttrsNode::decompress,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TcCopyAttrs", TcCopyAttrsNode, AttrsNode);
};

struct TcMmaAttrsNode : public AttrsNode {
  ffi::String scope;
  int64_t cta_group;
  ffi::Optional<int64_t> mma_m;
  ffi::Optional<int64_t> mma_n;
  ffi::String smem_desc;
  bool is_AB_tf32;
  ffi::Optional<bool> weight_stationary;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<TcMmaAttrsNode>()
        .def_ro("scope", &TcMmaAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("cta_group", &TcMmaAttrsNode::cta_group, refl::DefaultValue(int64_t(1)))
        .def_ro("mma_m", &TcMmaAttrsNode::mma_m,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("mma_n", &TcMmaAttrsNode::mma_n,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("smem_desc", &TcMmaAttrsNode::smem_desc, refl::DefaultValue(ffi::String("hoist")))
        .def_ro("is_AB_tf32", &TcMmaAttrsNode::is_AB_tf32, refl::DefaultValue(false))
        .def_ro("weight_stationary", &TcMmaAttrsNode::weight_stationary,
                refl::DefaultValue(ffi::Optional<bool>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TcMmaAttrs", TcMmaAttrsNode, AttrsNode);
};

struct MathAttrsNode : public AttrsNode {
  ffi::String scope;
  ffi::Optional<ffi::String> rounding_mode;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<MathAttrsNode>()
        .def_ro("scope", &MathAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("rounding_mode", &MathAttrsNode::rounding_mode,
                refl::DefaultValue(ffi::Optional<ffi::String>(std::nullopt)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.MathAttrs", MathAttrsNode, AttrsNode);
};

struct TrnAttrsNode : public AttrsNode {
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
    refl::ObjectDef<TrnAttrsNode>()
        .def_ro("scope", &TrnAttrsNode::scope, refl::DefaultValue(ffi::String("thread")))
        .def_ro("max_inst_size", &TrnAttrsNode::max_inst_size,
                refl::DefaultValue(ffi::Optional<int64_t>(std::nullopt)))
        .def_ro("opcode", &TrnAttrsNode::opcode, refl::DefaultValue(ffi::String("")))
        .def_ro("reduce_op", &TrnAttrsNode::reduce_op, refl::DefaultValue(ffi::String("sum")))
        .def_ro("op0", &TrnAttrsNode::op0, refl::DefaultValue(ffi::String("")))
        .def_ro("op1", &TrnAttrsNode::op1, refl::DefaultValue(ffi::String("")))
        .def_ro("axes", &TrnAttrsNode::axes, refl::DefaultValue(ffi::Array<int64_t>()))
        .def_ro("negate", &TrnAttrsNode::negate, refl::DefaultValue(false))
        .def_ro("reverse1", &TrnAttrsNode::reverse1, refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.TrnAttrs", TrnAttrsNode, AttrsNode);
};

ffi::Expected<void> ValidateTensorInstruction(const CallNode* call) noexcept try {
  auto op = call->op.as_or_throw<Op>();
  TVM_FFI_CHECK_EQ(call->args.size(), op->args_info.size(), TypeError)
      << op->name << " has an invalid operand count";
  TVM_FFI_CHECK(call->ty == PrimType::Void(), TypeError) << op->name << " must return void";
  TVM_FFI_CHECK(call->ty_args.empty(), TypeError) << op->name << " does not take type arguments";
  TVM_FFI_CHECK(call->attrs.defined() && call->attrs->GetTypeKey() == op->attrs_type_key, TypeError)
      << op->name << " requires its registered static attributes";
  static const auto& validators = Op::GetAttrMap<ffi::Function>(op_attr::kTensorCallValidate);
  validators[op](ffi::GetRef<Call>(call));
  return {};
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

void ConfigureTensorInstruction(Op op, ffi::Function validator) {
  OpDef(op->name)
      .set_attr(op_attr::kTensorCallValidate, validator)
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateTensorInstruction>(),
                     true);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ScopeAttrsNode::RegisterReflection();
  MemoryAttrsNode::RegisterReflection();
  AsyncAttrsNode::RegisterReflection();
  TMAAttrsNode::RegisterReflection();
  TcCopyAttrsNode::RegisterReflection();
  TcMmaAttrsNode::RegisterReflection();
  MathAttrsNode::RegisterReflection();
  TrnAttrsNode::RegisterReflection();
  ffi::reflection::GlobalDef().def("tirx.ConfigureTensorInstruction", ConfigureTensorInstruction);
}

}  // namespace tvm::tirx::tile
