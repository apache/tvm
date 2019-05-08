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
                                   void* text,
                                   void* rodata,
                                   void* data,
                                   void* bss) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   AddrToString(text),
                                   AddrToString(rodata),
                                   AddrToString(data),
                                   AddrToString(bss));
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
