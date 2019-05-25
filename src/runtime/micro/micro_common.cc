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

namespace tvm {
namespace runtime {

DevBaseOffset DevAddr::operator-(DevBaseAddr base) {
  return DevBaseOffset(value_ - base.value());
}

DevAddr DevAddr::operator+(size_t n) {
  return DevAddr(value_ + n);
}

DevAddr DevBaseAddr::operator+(DevBaseOffset offset) {
  return DevAddr(value_ + offset.value());
}

DevAddr DevBaseOffset::operator+(DevBaseAddr base) {
  return DevAddr(value_ + base.value());
}

DevBaseOffset DevBaseOffset::operator+(size_t n) {
  return DevBaseOffset(value_ + n);
}

const char* SectionToString(SectionKind section) {
  switch (section) {
    case kText: return "text";
    case kRodata: return "rodata";
    case kData: return "data";
    case kBss: return "bss";
    case kArgs: return "args";
    case kStack: return "stack";
    case kHeap: return "heap";
    case kWorkspace: return "workspace";
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

std::string RelocateBinarySections(std::string binary_path,
                                   DevAddr text,
                                   DevAddr rodata,
                                   DevAddr data,
                                   DevAddr bss) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   AddrToString(text.cast_to<void*>()),
                                   AddrToString(rodata.cast_to<void*>()),
                                   AddrToString(data.cast_to<void*>()),
                                   AddrToString(bss.cast_to<void*>()));
  return relocated_bin;
}

std::string ReadSection(std::string binary, SectionKind section) {
  CHECK(section == kText || section == kRodata || section == kData || section == kBss)
    << "ReadSection requires section to be one of text, rodata, data, or bss.";
  const auto* f = Registry::Get("tvm_callback_read_binary_section");
  CHECK(f != nullptr)
    << "Require tvm_callback_read_binary_section to exist in registry";
  TVMByteArray arr;
  arr.data = &binary[0];
  arr.size = binary.length();
  std::string section_contents = (*f)(arr, SectionToString(section));
  return section_contents;
}

size_t GetSectionSize(std::string binary_path, SectionKind section, size_t align) {
  CHECK(section == kText || section == kRodata || section == kData || section == kBss)
    << "GetSectionSize requires section to be one of text, rodata, data, or bss.";
  const auto* f = Registry::Get("tvm_callback_get_section_size");
  CHECK(f != nullptr)
    << "Require tvm_callback_get_section_size to exist in registry";
  size_t size = (*f)(binary_path, SectionToString(section));
  size = UpperAlignValue(size, align);
  return size;
}
}  // namespace runtime
}  // namespace tvm
