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
 *  Runtime-side test helpers used by the RPC test suite.
 *  Registered here (rather than in ``src/support/ffi_testing.cc``) so the
 *  ``minrpc`` server binary — which links only against ``libtvm_runtime``
 *  — can resolve them.
 * \file runtime/rpc/testing.cc
 */
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace runtime {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("rpc.testing.GetShapeSize",
           [](ffi::Shape shape) { return static_cast<int64_t>(shape.size()); })
      .def("rpc.testing.GetShapeElem", [](ffi::Shape shape, int idx) {
        TVM_FFI_ICHECK_LT(idx, shape.size());
        return shape[idx];
      });
}

}  // namespace runtime
}  // namespace tvm
