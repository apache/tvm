/*!
 *  Copyright (c) 2019 by Contributors
 * \file bin_util.cc
 * \brief binary modification utilities
 */

#include <stdio.h>
#include <string>
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

// TODO: implement these in Python using PackedFunc + Registry
void* GetSymbol(std::unordered_map<std::string, void*> symbol_map,
                std::string name,
                void* base_addr) {
  return nullptr;
}

std::string RelocateBinarySections(std::string binary_name,
                                   void* text,
                                   void* data,
                                   void* bss) {
  return "";
}

std::string ReadSection(std::string binary_name, SectionKind section) {
  return "";
}

size_t GetSectionSize(std::string binary_name, SectionKind section) {
  return 0;
}

std::unordered_map<std::string, void*> GetSymbolMap(std::string binary) {
  return std::unordered_map<std::string, void*>();
}
}  // namespace runtime
}  // namespace tvm
