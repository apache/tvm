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

/*! \file src/codegen/c/target_kind.cc
 * \brief C target schema and LLVM-independent canonicalization.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/target/target_kind.h>

#include "../canonicalizer/arm_mprofile.h"

namespace tvm {
TVM_FFI_STATIC_INIT_BLOCK() {
  TargetKindDef("c")
      .set_default_device_type(kDLCPU)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("march")
      .def_option<int64_t>("workspace-byte-alignment")
      .def_option<int64_t>("constants-byte-alignment")
      .set_default_keys({"cpu"})
      .set_target_canonicalizer([](ffi::Map<ffi::String, ffi::Any> target) {
        namespace mprofile = tvm::target::canonicalizer::mprofile;
        return mprofile::IsArch(target) ? mprofile::Canonicalize(target) : target;
      });
}
}  // namespace tvm
