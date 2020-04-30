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
 * \file micro_common.cc
 * \brief common utilties for uTVM
 */

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <cstdint>
#include "micro_session.h"
#include "micro_common.h"
#include "low_level_device.h"

namespace tvm {
namespace runtime {

const char* SectionToString(SectionKind section) {
  switch (section) {
    case SectionKind::kText: return "text";
    case SectionKind::kRodata: return "rodata";
    case SectionKind::kData: return "data";
    case SectionKind::kBss: return "bss";
    case SectionKind::kArgs: return "args";
    case SectionKind::kHeap: return "heap";
    case SectionKind::kWorkspace: return "workspace";
    case SectionKind::kStack: return "stack";
    default: return "";
  }
}

std::string RelocateBinarySections(
    const std::string& binary_path,
    TargetWordSize word_size,
    TargetPtr text_start,
    TargetPtr rodata_start,
    TargetPtr data_start,
    TargetPtr bss_start,
    TargetPtr stack_end,
    const std::string& toolchain_prefix) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   word_size.bytes(),
                                   text_start.cast_to<uint64_t>(),
                                   rodata_start.cast_to<uint64_t>(),
                                   data_start.cast_to<uint64_t>(),
                                   bss_start.cast_to<uint64_t>(),
                                   stack_end.cast_to<uint64_t>(),
                                   toolchain_prefix);
  return relocated_bin;
}

std::string ReadSection(const std::string& binary,
                        SectionKind section,
                        const std::string& toolchain_prefix) {
  CHECK(section == SectionKind::kText || section == SectionKind::kRodata ||
        section == SectionKind::kData || section == SectionKind::kBss)
      << "ReadSection requires section to be one of text, rodata, data, or bss.";
  const auto* f = Registry::Get("tvm_callback_read_binary_section");
  CHECK(f != nullptr)
    << "Require tvm_callback_read_binary_section to exist in registry";
  TVMByteArray arr;
  arr.data = &binary[0];
  arr.size = binary.length();
  std::string section_contents = (*f)(arr, SectionToString(section), toolchain_prefix);
  return section_contents;
}

size_t GetSectionSize(const std::string& binary_path,
                      SectionKind section,
                      const std::string& toolchain_prefix,
                      TargetWordSize word_size) {
  CHECK(section == SectionKind::kText || section == SectionKind::kRodata ||
        section == SectionKind::kData || section == SectionKind::kBss)
      << "GetSectionSize requires section to be one of text, rodata, data, or bss.";
  const auto* f = Registry::Get("tvm_callback_get_section_size");
  CHECK(f != nullptr)
    << "Require tvm_callback_get_section_size to exist in registry";
  int size = (*f)(binary_path, SectionToString(section), toolchain_prefix);
  return UpperAlignValue(size, word_size.bytes());
}

}  // namespace runtime
}  // namespace tvm
