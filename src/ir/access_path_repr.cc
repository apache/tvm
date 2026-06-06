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
 * \file ir/access_path_repr.cc
 * \brief FFI registration for ffi-repr-based printing.
 *
 * This file:
 *  - Registers node.AsRepr (for backward Python compatibility) via ffi::ReprPrint.
 *
 * Note: __ffi_repr__ hooks for ffi::reflection::AccessPath and AccessStep are
 * registered by tvm-ffi itself (src/ffi/extra/reflection_extra.cc, landed in
 * apache/tvm-ffi#598). The duplicate registrations that previously lived here
 * were removed when bumping tvm-ffi to 59da4c0 to avoid a double-registration
 * abort at library load time.
 *
 * Note: tvm::Dump() has been removed (zero in-tree callers). Use
 * tvm::ffi::ReprPrint(any) directly from gdb instead.
 */
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // node.AsRepr: backward-compatible Python entry point.
  // Python's tvm.runtime._ffi_node_api sets __object_repr__ = AsRepr via init_ffi_api.
  refl::GlobalDef().def("node.AsRepr",
                        [](ffi::Any obj) -> ffi::String { return ffi::ReprPrint(obj); });
}
}  // namespace tvm
