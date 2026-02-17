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
#include "canonicalize.h"

#include <string>

#include "arm_aprofile.h"
#include "arm_mprofile.h"

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {

ffi::Optional<ffi::String> DetectSystemTriple() {
#ifdef TVM_LLVM_VERSION
  auto pf = tvm::ffi::Function::GetGlobal("target.llvm_get_system_triple");
  ICHECK(pf.has_value()) << "The target llvm_get_system_triple was not found, "
                            "please compile with USE_LLVM = ON";
  return (*pf)().cast<ffi::String>();
#endif
  return {};
}

ffi::Map<ffi::String, ffi::Any> Canonicalize(ffi::Map<ffi::String, ffi::Any> target) {
  ffi::String kind = Downcast<ffi::String>(target.Get("kind").value());
  ffi::Optional<ffi::String> mtriple = Downcast<ffi::Optional<ffi::String>>(target.Get("mtriple"));
  ffi::Optional<ffi::String> mcpu = Downcast<ffi::Optional<ffi::String>>(target.Get("mcpu"));

  // Try to fill in the blanks by detecting target information from the system
  if (kind == "llvm" && !mtriple.has_value() && !mcpu.has_value()) {
    ffi::String system_triple = DetectSystemTriple().value_or("");
    target.Set("mtriple", system_triple);
  }

  if (mprofile::IsArch(target)) {
    return mprofile::Canonicalize(target);
  }

  if (aprofile::IsArch(target)) {
    return aprofile::Canonicalize(target);
  }

  return target;
}

}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm
