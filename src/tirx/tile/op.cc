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

struct ScopeAttrs : public AttrsNode {
  ffi::String scope;
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<ScopeAttrs>().def_ro("scope", &ScopeAttrs::scope,
                                         refl::DefaultValue(ffi::String("thread")));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.tensor.ScopeAttrs", ScopeAttrs, AttrsNode);
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
  TVM_FFI_CHECK(call->attrs.has_value() && call->attrs->GetTypeKey() == op->attrs_type_key,
                TypeError)
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
  ScopeAttrs::RegisterReflection();
  TrnAttrs::RegisterReflection();
  ffi::reflection::GlobalDef().def("tirx.ConfigureTensorInstruction", ConfigureTensorInstruction);
}

}  // namespace tvm::tirx::tile
