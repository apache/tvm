/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_common.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_COMMON_H_
#define TVM_RUNTIME_MICRO_MICRO_COMMON_H_

#include <stdio.h>
#include <string>
#include <unordered_map>

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

/*! \brief number of bytes in each page */
constexpr int kPageSize = 4096;

/*! \brief memory offset at which text section starts  */
constexpr int kTextStart = 64;

/*! \brief memory offset at which data section starts  */
constexpr int kDataStart = 50000;

/*! \brief memory offset at which bss section starts  */
constexpr int kBssStart = 100000;

/*! \brief memory offset at which args section starts  */
constexpr int kArgsStart = 150000;

/*! \brief memory offset at which stack section starts  */
constexpr int kStackStart = 250000;

/*! \brief memory offset at which heap section starts  */
constexpr int kHeapStart = 300000;

/*! \brief memory offset at which workspace section starts  */
constexpr int kWorkspaceStart = 350000;

/*! \brief total memory size */
constexpr int kMemorySize = 409600;

/*!
 * \brief converts actual address to offset from base_addr
 * \param addr address to be converted to offset
 * \param base_addr base address
 * \return offset from base_addr
 */
inline void* GetOffset(const void* addr, const void* base_addr) {
  return (void*) ((uint8_t*) addr - (uint8_t*) base_addr);
}

/*!
 * \brief converts offset to actual address
 * \param offset offset from base_addr
 * \param base_addr base address
 * \return address relative to base_addr
 */
inline void* GetAddr(const void* offset, const void* base_addr) {
  return (void*) ((uint8_t*) base_addr +
                  reinterpret_cast<std::uintptr_t>(offset));
}

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
                const void* base_addr);

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
 * \brief reads section from binary
 * \param binary input binary contents
 * \param section section type to be read
 * \return contents of the section
 */
std::string ReadSection(std::string binary, SectionKind section);

/*!
 * \brief finds size of the section in the binary
 * \param binary input binary contents
 * \param section section type
 * \param align alignment of the returned size
 * \return size of the section if it exists, 0 otherwise
 */
size_t GetSectionSize(std::string binary_name, SectionKind section, int align = 8);

/*!
 * \brief builds a map of symbol to address
 * \param binary contents of the binary file
 * \return map of symbols to their addresses
 */
std::unordered_map<std::string, void*> GetSymbolMap(std::string binary);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
