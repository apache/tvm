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

/*! \file src/codegen/llvm/target_kind.cc
 * \brief LLVM target schema and canonicalizer registration, available without LLVM.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/target/target_kind.h>

#include "canonicalizer/canonicalize.h"

namespace tvm {
TVM_FFI_STATIC_INIT_BLOCK() {
  TargetKindDef("llvm")
      .set_default_device_type(kDLCPU)
      .def_option<ffi::Array<ffi::String>>("mattr")
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<ffi::String>("mfloat-abi")
      .def_option<ffi::String>("mabi")
      .def_option<int64_t>("num-cores")
      // Fast math flags, see https://llvm.org/docs/LangRef.html#fast-math-flags
      .def_option<bool>("fast-math")  // implies all the below
      .def_option<bool>("fast-math-nnan")
      .def_option<bool>("fast-math-ninf")
      .def_option<bool>("fast-math-nsz")
      .def_option<bool>("fast-math-arcp")
      .def_option<bool>("fast-math-contract")
      .def_option<bool>("fast-math-reassoc")
      .def_option<int64_t>("opt-level")
      // LLVM command line flags, see below
      .def_option<ffi::Array<ffi::String>>("cl-opt")
      // TVM & LLVM custom vector bit width
      .def_option<int64_t>("vector-width")
      .set_default_keys({"cpu"})
      // Force the external codegen kind attribute to be registered, even if no external
      // codegen targets are enabled by the TVM build.
      .set_target_canonicalizer(tvm::target::canonicalizer::llvm::Canonicalize);

  // Note regarding the "cl-opt" attribute:
  // Each string in the array has the format
  //   -optionname[[:type]=value]
  // where
  //   * optionname is the actual LLVM option (e.g. "unroll-threshold")
  //   * type is one of "bool", "int", "uint", or "string"
  //   * value is the corresponding option value (for "bool" type is can be 0 or "false"
  //     for false value, or 1 or "true" for true value)
  // If type is omitted, it is assumed to be "bool". If value is omitted, it is assumed
  // to be "true".
  //
  // The type must match the option type in LLVM. To find the type, search the LLVM
  // repository (https://github.com/llvm/llvm-project) for optionname, and look for
  // its definition: it will be a declaration of a variable of type cl::opt<T> with
  // optionname being an argument to the constructor. The T in the declaration is
  // the type.
  // For example, for unroll-threshold, we get the following declaration:
  // static cl::opt<unsigned>
  //     UnrollThreshold("unroll-threshold", cl::Hidden,
  //                     cl::desc("The cost threshold for loop unrolling"));
  // Hence the type is "uint".
}
}  // namespace tvm
