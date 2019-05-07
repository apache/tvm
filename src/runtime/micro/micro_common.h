/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_common.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_COMMON_H_
#define TVM_RUNTIME_MICRO_MICRO_COMMON_H_

#include <sstream>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <tvm/runtime/registry.h>

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

/*! \brief absolute device address */
struct dev_addr {
  std::uintptr_t val_;

  explicit dev_addr(std::uintptr_t val) : val_(val) {}
  dev_addr() : val_(0) {}
  explicit dev_addr(std::nullptr_t) : val_(0) {}
  ~dev_addr() {}
};

/*! \brief TODO */
struct dev_base_addr {
  std::uintptr_t val_;

  explicit dev_base_addr(std::uintptr_t val) : val_(val) {}
  dev_base_addr() : val_(0) {}
  explicit dev_base_addr(std::nullptr_t) : val_(0) {}
  ~dev_base_addr() {}
};

/*! \brief offset from device base address */
struct dev_base_offset {
  std::uintptr_t val_;

  explicit dev_base_offset(std::uintptr_t val) : val_(val) {}
  dev_base_offset() : val_(0) {}
  explicit dev_base_offset(std::nullptr_t) : val_(0) {}
  ~dev_base_offset() {}
};

class SymbolMap {
 public:
  SymbolMap() {}

  SymbolMap(std::string binary, dev_base_addr base_addr) {
    const auto* f = Registry::Get("tvm_callback_get_symbol_map");
    CHECK(f != nullptr) << "Require tvm_callback_get_symbol_map to exist in registry";
    TVMByteArray arr;
    arr.data = &binary[0];
    arr.size = binary.length();
    std::string map_str = (*f)(arr);
    // parse symbols and addresses from returned string
    std::stringstream stream;
    stream << map_str;
    std::string name;
    std::uintptr_t addr;
    stream >> name;
    stream >> std::hex >> addr;
    while (stream) {
      map_[name] = dev_base_offset(addr - base_addr.val_);
      stream >> name;
      stream >> std::hex >> addr;
    }
  }

  dev_base_offset operator[](std::string name) {
    auto result = map_.find(name);
    CHECK(result != map_.end()) << "\"" << name << "\" not in symbol map";
    return result->second;
  }

 private:
  std::unordered_map<std::string, dev_base_offset> map_;
};

/*! \brief number of bytes in each page */
constexpr int kPageSize = 4096;

/*! \brief memory offset at which text section starts  */
const dev_base_offset kTextStart = dev_base_offset(64);

/*! \brief memory offset at which data section starts  */
const dev_base_offset kDataStart = dev_base_offset(50000);

/*! \brief memory offset at which bss section starts  */
const dev_base_offset kBssStart = dev_base_offset(100000);

/*! \brief memory offset at which args section starts  */
const dev_base_offset kArgsStart = dev_base_offset(150000);

/*! \brief memory offset at which stack section starts  */
const dev_base_offset kStackStart = dev_base_offset(250000);

/*! \brief memory offset at which heap section starts  */
const dev_base_offset kHeapStart = dev_base_offset(300000);

/*! \brief memory offset at which workspace section starts  */
const dev_base_offset kWorkspaceStart = dev_base_offset(350000);

/*! \brief total memory size */
constexpr int kMemorySize = 450000;

/*! \brief default size alignment */
constexpr int kDefaultSizeAlignment = 8;


/*!
 * \brief converts actual address to offset from base_addr
 * \param addr address to be converted to offset
 * \param base_addr base address
 * \return offset from base_addr
 */
// inline void* GetOffset(const void* addr, const void* base_addr) {
//   return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(const_cast<void*>(addr)) -
//                                  reinterpret_cast<uint8_t*>(const_cast<void*>(base_addr)));
// }

/*!
 * \brief upper-aligns value according to specified alignment
 * \param value value to be aligned
 * \param align alignment
 * \return upper-aligned value
 */
inline size_t UpperAlignValue(size_t value, size_t align) {
  return value + (align - (value % align)) % align;
}

/*!
 * \brief converts offset to actual address
 * \param offset offset from base_addr
 * \param base base address
 * \return address relative to base_addr
 */
inline dev_addr GetAddr(const dev_base_offset offset, const dev_base_addr base) {
  // return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(const_cast<void*>(base_addr)) +
                                //  reinterpret_cast<std::uintptr_t>(offset));
  // TODO: replace with operator overloading
  return dev_addr(base.val_ + offset.val_);
}

/*!
 * \brief maps section enums to text
 * \param section section type
 * \return text form of the specified section
 */
const char* SectionToString(SectionKind section);

dev_addr GetSymbol(std::unordered_map<std::string, void*> symbol_map,
                   std::string name);

/*!
 * \brief get relative address of the symbol from the symbol map
 * \param map of symbols to addresses
 * \param name symbol name
 * \param base base address to obtain offset from
 * \return address of the symbol relative to base_addr
 */
dev_base_offset GetSymbolOffset(std::unordered_map<std::string, void*> symbol_map,
                std::string name,
                const dev_base_addr base);

/*!
 * \brief links binary by repositioning section addresses
 * \param binary_name input binary filename
 * \param text new text section address
 * \param data new data section address
 * \param bss new bss section address
 * \return relocated binary file contents
 */
// TODO: Convert to dev_base_offset or dev_addr arg types
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
 * \param align alignment of the returned size (default: 8)
 * \return size of the section if it exists, 0 otherwise
 */
size_t GetSectionSize(std::string binary_name, SectionKind section,
                      size_t align = kDefaultSizeAlignment);

/*!
 * \brief builds a map of symbol to address
 * \param binary contents of the binary file
 * \return map of symbols to their addresses
 */
//std::unordered_map<std::string, dev_base_offset> GetSymbolMap(std::string binary, dev_base_addr base_addr);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
