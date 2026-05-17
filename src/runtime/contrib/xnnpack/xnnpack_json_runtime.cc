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
 * \file src/runtime/contrib/xnnpack/xnnpack_json_runtime.cc
 * \brief Phase 1 XNNPACK JSON runtime skeleton.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <string>

#include "../json/json_runtime.h"

#include <xnnpack.h>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class XNNPACKJSONRuntime : public JSONRuntimeBase {
 public:
  XNNPACKJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                     const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* kind() const override { return "xnnpack_json"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";

    SetupConstants(consts);

    const xnn_status status = xnn_initialize(nullptr);
    TVM_FFI_ICHECK_EQ(status, xnn_status_success)
        << "Failed to initialize XNNPACK runtime. xnn_initialize returned status " << status;

    // TODO(XNNPACK): XNNPACK may read XNN_EXTRA_BYTES past tensor bounds. Operator lowering must
    // ensure buffers passed to XNNPACK satisfy this padding contract.
    // TODO(XNNPACK): Static weight tensors passed into XNNPACK must outlive XNNPACK subgraphs,
    // runtimes, and operator objects that reference them.
  }

  void Run() override {
    TVM_FFI_THROW(InternalError)
        << "XNNPACK execution is not implemented in Phase 1. No Relax operators are supported.";
  }
};

ffi::Module XNNPACKJSONRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                     const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<XNNPACKJSONRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.XNNPACKJSONRuntimeCreate", XNNPACKJSONRuntimeCreate)
      .def("ffi.Module.load_from_bytes.xnnpack_json",
           JSONRuntimeBase::LoadFromBytes<XNNPACKJSONRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
