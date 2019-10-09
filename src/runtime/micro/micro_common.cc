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
 *  Copyright (c) 2019 by Contributors
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

size_t GetDefaultSectionSize(SectionKind kind) {
  switch (kind) {
    case SectionKind::kText:
      return 0xF000;
    case SectionKind::kRodata:
      return 0xF000;
    case SectionKind::kData:
      return 0xF00;
    case SectionKind::kBss:
      return 0xF00;
    case SectionKind::kArgs:
      return 0xF0000;
    case SectionKind::kStack:
      return 0xF000;
    case SectionKind::kHeap:
      return 0xF00000;
    case SectionKind::kWorkspace:
      return 0xF0000;
    default:
      LOG(FATAL) << "invalid section " << static_cast<size_t>(kind);
      return 0;
  }
}

const char* SectionToString(SectionKind section) {
  switch (section) {
    case SectionKind::kText: return "text";
    case SectionKind::kRodata: return "rodata";
    case SectionKind::kData: return "data";
    case SectionKind::kBss: return "bss";
    case SectionKind::kArgs: return "args";
    case SectionKind::kStack: return "stack";
    case SectionKind::kHeap: return "heap";
    case SectionKind::kWorkspace: return "workspace";
    default: return "";
  }
}

static std::string AddrToString(void* addr) {
  std::stringstream stream;
  if (addr != nullptr)
    stream << addr;
  else
    stream << "0x0";
  std::string string_addr = stream.str();
  return string_addr;
}

std::string RelocateBinarySections(const std::string& binary_path,
                                   DevPtr text,
                                   DevPtr rodata,
                                   DevPtr data,
                                   DevPtr bss,
                                   const std::string& toolchain_prefix) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   AddrToString(text.cast_to<void*>()),
                                   AddrToString(rodata.cast_to<void*>()),
                                   AddrToString(data.cast_to<void*>()),
                                   AddrToString(bss.cast_to<void*>()),
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
                      size_t align) {
  CHECK(section == SectionKind::kText || section == SectionKind::kRodata ||
        section == SectionKind::kData || section == SectionKind::kBss)
      << "GetSectionSize requires section to be one of text, rodata, data, or bss.";
  const auto* f = Registry::Get("tvm_callback_get_section_size");
  CHECK(f != nullptr)
    << "Require tvm_callback_get_section_size to exist in registry";
  int size = (*f)(binary_path, SectionToString(section), toolchain_prefix);
  return UpperAlignValue(size, align);
}

}  // namespace runtime
}  // namespace tvm
