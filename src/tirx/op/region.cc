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
 * \file tirx/op/region.cc
 * \brief TIRx region operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

static ffi::Array<Var> RegionNoBodyParams(const CallNode*) { return {}; }

ffi::Array<Var> LaunchThreadBodyParams(const CallNode* call) {
  auto tag = call->args[0].as_or_throw<StringImm>();
  PrimType dtype = call->args[1].as_or_throw<IntExpr>().ty();
  TVM_FFI_CHECK(!tag->value.empty(), ValueError)
      << "launch_thread expects a nonempty StringImm thread tag";
  TVM_FFI_CHECK_GT(dtype.bits(), 1, ValueError)
      << "launch_thread extent must have a scalar integer type wider than one bit";
  return {PrimVar("", dtype)};
}

void ValidateBuiltinRegion(const RegionStmtNode* region) {
  TVM_FFI_CHECK(region->result_vars.empty() && region->attrs->dict.empty(), ValueError)
      << region->op->name << " expects no results or attributes";
}

const Op& launch_thread_op() {
  static const Op op = Op::Get("tirx.launch_thread");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.launch_thread", "Bind a thread index within a body with a launch extent.")
      .signature(sig::arg<StringImm>("tag"), sig::arg<IntExpr>("extent"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&LaunchThreadBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

const Op& device_entry_op() {
  static const Op op = Op::Get("tirx.device_entry");
  return op;
}

void ValidateDeviceEntry(const RegionStmtNode* region) {
  TVM_FFI_CHECK(region->result_vars.empty() && region->body_params.empty(), ValueError)
      << "device_entry expects no results or body parameters";
  if (auto fields = region->attrs->dict.Get("cuda.launch_fields")) {
    TVM_FFI_CHECK_EQ(fields->as_or_throw<ffi::Array<ffi::String>>().size(), region->args.size(),
                     ValueError)
        << "device_entry launch fields must describe every configuration operand";
  } else {
    TVM_FFI_CHECK(region->args.empty(), ValueError)
        << "device_entry configuration operands require cuda.launch_fields";
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_entry", "Enter a device kernel with independent launch configuration.")
      .signature(sig::var_args("launch_values"), sig::call_attrs<DictAttrsNode>())
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateDeviceEntry>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

const Op& device_context_op() {
  static const Op op = Op::Get("tirx.device_context");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_context", "Supply the device type and ID within a region.")
      .signature(sig::arg<IntExpr>("device_type"), sig::arg<IntExpr>("device_id"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

const Op& compute_scope_op() {
  static const Op op = Op::Get("tirx.compute_scope");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.compute_scope", "Outline a named CPU compute region.")
      .signature(sig::arg<StringImm>("name"))
      .set_attr<FRegionGetBodyParams>("FRegionGetBodyParams",
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>("FRegionValidate",
                                 FRegionValidate::FromNative<&ValidateBuiltinRegion>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

const Op& parallel_launch_op() {
  static const Op op = Op::Get("tirx.parallel_launch");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
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
