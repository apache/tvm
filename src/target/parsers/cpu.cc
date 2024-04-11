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
#include "cpu.h"

#include <string>

#include "aprofile.h"
#include "mprofile.h"

namespace tvm {
namespace target {
namespace parsers {
namespace cpu {

Optional<String> DetectSystemTriple() {
#ifdef TVM_LLVM_VERSION
  auto pf = tvm::runtime::Registry::Get("target.llvm_get_system_triple");
  ICHECK(pf != nullptr) << "The target llvm_get_system_triple was not found, "
                           "please compile with USE_LLVM = ON";
  return (*pf)();
#endif
  return {};
}

TargetJSON ParseTarget(TargetJSON target) {
  String kind = Downcast<String>(target.Get("kind"));
  Optional<String> mtriple = Downcast<Optional<String>>(target.Get("mtriple"));
  Optional<String> mcpu = Downcast<Optional<String>>(target.Get("mcpu"));

  // Try to fill in the blanks by detecting target information from the system
  if (kind == "llvm" && !mtriple.defined() && !mcpu.defined()) {
    String system_triple = DetectSystemTriple().value_or("");
    target.Set("mtriple", system_triple);
  }

  if (mprofile::IsArch(target)) {
    return mprofile::ParseTarget(target);
  }

  if (aprofile::IsArch(target)) {
    return aprofile::ParseTarget(target);
  }

  return target;
}

}  // namespace cpu
}  // namespace parsers
}  // namespace target
}  // namespace tvm
