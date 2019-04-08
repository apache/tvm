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
    case kData: return "data";
    case kBss: return "bss";
    case kArgs: return "args";
    case kStack: return "stack";
    case kHeap: return "heap";
    case kWorkspace: return "workspace";
    default: return "";
  }
}

void* GetSymbol(std::unordered_map<std::string, void*> symbol_map,
                std::string name,
                void* base_addr) {
  void* symbol_addr = symbol_map[name];
  return (void*)((uint8_t*) symbol_addr - (uint8_t*) base_addr);
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
                                   void* data,
                                   void* bss) {
  const auto* f = Registry::Get("tvm_callback_relocate_binary");
  CHECK(f != nullptr)
    << "Require tvm_callback_relocate_binary to exist in registry";
  std::string relocated_bin = (*f)(binary_path,
                                   AddrToString(text),
                                   AddrToString(data),
                                   AddrToString(bss));
  return relocated_bin;
}

std::string ReadSection(std::string binary_name, SectionKind section) {
  CHECK(section == kText || section == kData || section == kBss)
    << "ReadSection requires section to be one of text, data or bss.";
  const auto* f = Registry::Get("tvm_callback_read_binary_section");
  CHECK(f != nullptr)
    << "Require tvm_callback_read_binary_section to exist in registry";
  TVMByteArray arr;
  arr.data = &binary[0];
  arr.size = binary.length();
  std::string section_contents = (*f)(arr, SectionToString(section));
  return section_contents;
}

size_t GetSectionSize(std::string binary_path, SectionKind section, int align) {
  CHECK(section == kText || section == kData || section == kBss)
    << "GetSectionSize requires section to be one of text, data or bss.";
  const auto* f = Registry::Get("tvm_callback_get_section_size");
  CHECK(f != nullptr)
    << "Require tvm_callback_get_section_size to exist in registry";
  size_t size = (*f)(binary_path, SectionToString(section));
  while (size % align) size++;
  return size;
}

std::unordered_map<std::string, void*> GetSymbolMap(std::string binary) {
  const auto* f = Registry::Get("tvm_callback_get_symbol_map");
  CHECK(f != nullptr) << "Require tvm_callback_get_symbol_map to exist in registry";
  TVMByteArray arr;
  arr.data = &binary[0];
  arr.size = binary.length();
  std::string map_str = (*f)(arr);
  // parse symbols and addresses from returned string
  std::unordered_map<std::string, void*> symbol_map;
  std::stringstream stream;
  stream << map_str;
  std::string name;
  void* addr;
  stream >> name;
  stream >> std::hex >> addr;
  while (stream) {
    symbol_map[name] = addr;
    stream >> name;
    stream >> std::hex >> addr;
  }
  return symbol_map;
}
}  // namespace runtime
}  // namespace tvm
