/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_common.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_COMMON_H_
#define TVM_RUNTIME_MICRO_MICRO_COMMON_H_

#include <stdio.h>
#include <string>
#include "micro_session.h"

namespace tvm {
namespace runtime {
/*!
 * \brief enum of device memory region sections
 */
enum SectionKind : int {
  kText = 0,
  kData = 1,
  kBss = 2,
  kArgs = 3,
  kStack = 4,
  kHeap = 5,
  kWorkspace = 6,
};

/*!
 * \brief maps section enums to text
 * \param section section type
 * \return text form of the specified section
 */
const char* SectionToString(SectionKind section);

/*!
 * \brief get relative address of the symbol from the symbol map
 * \param map of symbols to addresses
 * \param name symbol name
 * \param base_addr base address to obtain offset from
 * \return address of the symbol relative to base_addr
 */
void* GetSymbol(std::unordered_map<std::string, void*> symbol_map,
                std::string name,
                void* base_addr);

/*!
 * \brief links binary by repositioning section addresses
 * \param binary_name input binary filename
 * \param text new text section address
 * \param data new data section address
 * \param bss new bss section address
 * \return relocated binary file contents
 */
std::string RelocateBinarySections(std::string binary_name,
                                   void* text,
                                   void* data,
                                   void* bss);

/*!
 * \brief reads section from binary file
 * \param binary_name input binary filename
 * \param section section type to be read
 * \return contents of the section
 */
std::string ReadSection(std::string binary_name, SectionKind section);

/*!
 * \brief finds size of the section in the binary
 * \param binary input binary contents
 * \param section section type
 * \return size of the section if it exists, 0 otherwise
 */
size_t GetSectionSize(std::string binary_name, SectionKind section);

/*!
 * \brief builds a map of symbol to address
 * \param binary contents of the binary file
 * \return map of symbols to their addresses
 */
std::unordered_map<std::string, void*> GetSymbolMap(std::string binary);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
