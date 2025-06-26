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
  auto pf = tvm::ffi::Function::GetGlobal("target.llvm_get_system_triple");
  ICHECK(pf.has_value()) << "The target llvm_get_system_triple was not found, "
                            "please compile with USE_LLVM = ON";
  return (*pf)().cast<String>();
#endif
  return {};
}

TargetJSON ParseTarget(TargetJSON target) {
  String kind = Downcast<String>(target.Get("kind").value());
  Optional<String> mtriple = Downcast<Optional<String>>(target.Get("mtriple"));
  Optional<String> mcpu = Downcast<Optional<String>>(target.Get("mcpu"));

  // Try to fill in the blanks by detecting target information from the system
  if (kind == "llvm" && !mtriple.has_value() && !mcpu.has_value()) {
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

int extractVLENFromString(const std::string& input) {
  for (size_t i = 0; i + 4 <= input.size(); ++i) {
    // Look for the starting sequence "zvl"
    if (input[i] == 'z' && input[i + 1] == 'v' && input[i + 2] == 'l') {
      size_t j = i + 3;
      std::string number;

      // Collect digits
      while (j < input.size() && std::isdigit(input[j])) {
        number += input[j];
        ++j;
      }

      // Check if followed by 'b' after digits
      if (!number.empty() && j < input.size() && input[j] == 'b') {
        return std::stoi(number);  // Convert the number to int
      }
    }
  }

  throw std::runtime_error("No valid pattern found");
}

}  // namespace cpu
}  // namespace parsers
}  // namespace target
}  // namespace tvm
