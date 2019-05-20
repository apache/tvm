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

dev_base_offset dev_addr::operator-(dev_base_addr base) {
  return dev_base_offset(val_ - base.val());
}

dev_addr dev_addr::operator+(size_t n) {
  return dev_addr(val_ + n);
}

dev_addr dev_base_addr::operator+(dev_base_offset offset) {
  return dev_addr(val_ + offset.val());
}

dev_addr dev_base_offset::operator+(dev_base_addr base) {
  return dev_addr(val_ + base.val());
}

dev_base_offset dev_base_offset::operator+(size_t n) {
  return dev_base_offset(val_ + n);
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
                                   dev_addr text,
                                   dev_addr rodata,
                                   dev_addr data,
                                   dev_addr bss) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   AddrToString(text.as_ptr<void>()),
                                   AddrToString(rodata.as_ptr<void>()),
                                   AddrToString(data.as_ptr<void>()),
                                   AddrToString(bss.as_ptr<void>()));
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
