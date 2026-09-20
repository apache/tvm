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
#ifndef TVM_TIRX_ATTRS_H_
#define TVM_TIRX_ATTRS_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ir/attrs.h>

namespace tvm {
namespace tirx {

/*! \brief Launch metadata for call_ffi_kernel. */
struct CallFFIKernelAttr : public AttrsNode {
  /*!
   * \brief Ordered launch tags describing the suffix of the call arguments.
   *
   * The first call argument is the kernel symbol, followed by kernel operands
   * and launch values. Flag-only tags consume no value; dynamic shared-memory
   * bytes, when present, are last. Runtime expressions remain in Call.args.
   */
  ffi::Array<ffi::String> launch_params;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<CallFFIKernelAttr>().def_ro("launch_params",
                                                           &CallFFIKernelAttr::launch_params);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.CallFFIKernelAttr", CallFFIKernelAttr, AttrsNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_ATTRS_H_
