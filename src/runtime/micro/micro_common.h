/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_common.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_COMMON_H_
#define TVM_RUNTIME_MICRO_MICRO_COMMON_H_

#include <stdio.h>

#include <tvm/runtime/registry.h>

#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {
/*!
 * \brief enum of device memory region sections
 */
enum SectionKind : int {
  kText = 0,
  kRodata = 1,
  kData = 2,
  kBss = 3,
  kArgs = 4,
  kStack = 5,
  kHeap = 6,
  kWorkspace = 7,
};

// TODO(weberlo): There's a lot of duplication between these classes.  How can we consolidate?
class dev_addr;
class dev_base_addr;
class dev_base_offset;

/*! \brief absolute device address */
class dev_addr {
 public:
  explicit dev_addr(std::uintptr_t val) : val_(val) {}
  dev_addr() : val_(0) {}
  explicit dev_addr(std::nullptr_t) : val_(0) {}
  ~dev_addr() {}

  std::uintptr_t val() const { return val_; }
  template <typename T>
  T* as_ptr() const { return reinterpret_cast<T*>(val_); }
  bool is_null() const { return val_ == 0; }

  dev_base_offset operator-(dev_base_addr base);
  dev_addr operator+(size_t n);

 private:
  std::uintptr_t val_;
};

/*! \brief base address of the device */
class dev_base_addr {
 public:
  explicit dev_base_addr(std::uintptr_t val) : val_(val) {}
  dev_base_addr() : val_(0) {}
  explicit dev_base_addr(std::nullptr_t) : val_(0) {}
  ~dev_base_addr() {}

  std::uintptr_t val() const { return val_; }
  template <typename T>
  T* as_ptr() const { return reinterpret_cast<T*>(val_); }
  bool is_null() const { return val_ == 0; }

  dev_addr operator+(dev_base_offset offset);

 private:
  std::uintptr_t val_;
};

/*! \brief offset from device base address */
class dev_base_offset {
 public:
  explicit dev_base_offset(std::uintptr_t val) : val_(val) {}
  dev_base_offset() : val_(0) {}
  explicit dev_base_offset(std::nullptr_t) : val_(0) {}
  ~dev_base_offset() {}

  std::uintptr_t val() const { return val_; }
  template <typename T>
  T* as_ptr() const { return reinterpret_cast<T*>(val_); }
  bool is_null() const { return val_ == 0; }

  dev_addr operator+(dev_base_addr base);
  dev_base_offset operator+(size_t n);

 private:
  std::uintptr_t val_;
};

/*!
 * \brief map from symbols to their on-device offsets
 */
class SymbolMap {
 public:
  /*!
   * \brief default constructor
   */
  SymbolMap() {}

  /*!
   * \brief constructor that builds the mapping
   * \param binary contents of binary object file
   * \param base_addr base address of the target device
   */
  SymbolMap(std::string binary, dev_base_addr base_addr) {
    const auto* f = Registry::Get("tvm_callback_get_symbol_map");
    CHECK(f != nullptr) << "require tvm_callback_get_symbol_map to exist in registry";
    TVMByteArray arr;
    arr.data = &binary[0];
    arr.size = binary.length();
    std::string map_str = (*f)(arr);
    // Parse symbols and addresses from returned string.
    std::stringstream stream;
    stream << map_str;
    std::string name;
    std::uintptr_t addr;
    stream >> name;
    stream >> std::hex >> addr;
    while (stream) {
      map_[name] = dev_addr(addr) - base_addr;
      stream >> name;
      stream >> std::hex >> addr;
    }
  }

  /*!
   * \brief retrieve on-device offset for a symbol name
   * \param name name of the symbol
   * \return on-device offset of the symbol
   */
  dev_base_offset operator[](std::string name) {
    auto result = map_.find(name);
    CHECK(result != map_.end()) << "\"" << name << "\" not in symbol map";
    return result->second;
  }

 private:
  /*! \brief backing map */
  std::unordered_map<std::string, dev_base_offset> map_;
};

/*! \brief struct containing section location info */
struct SectionLocation {
  /*! \brief section start offset */
  dev_base_offset start;
  /*! \brief size of section */
  size_t size;
};

/*! \brief struct containing section locations and symbol mappings */
struct BinaryInfo {
  /*! \brief text section location */
  SectionLocation text;
  /*! \brief rodata section location */
  SectionLocation rodata;
  /*! \brief data section location */
  SectionLocation data;
  /*! \brief bss section location */
  SectionLocation bss;
  /*! \brief symbol map to offsets */
  SymbolMap symbol_map;
};

// TODO(weberlo): should this be here?
/*! \brief number of bytes in each page */
constexpr int kPageSize = 4096;

// TODO(weberlo): We need to allow configurable memory layouts by the user, and
// the constants below should be made into defaults.

/*! \brief memory offset at which text section starts  */
const dev_base_offset kTextStart = dev_base_offset(64);

/*! \brief memory offset at which rodata section starts  */
const dev_base_offset kRodataStart = dev_base_offset(500000000);

/*! \brief memory offset at which data section starts  */
const dev_base_offset kDataStart = dev_base_offset(1000000000);

/*! \brief memory offset at which bss section starts  */
const dev_base_offset kBssStart = dev_base_offset(1500000000);

/*! \brief memory offset at which args section starts  */
const dev_base_offset kArgsStart = dev_base_offset(2000000000);

/*! \brief memory offset at which stack section starts  */
const dev_base_offset kStackStart = dev_base_offset(3000000000);

/*! \brief memory offset at which heap section starts  */
const dev_base_offset kHeapStart = dev_base_offset(3500000000);

/*! \brief memory offset at which workspace section starts  */
const dev_base_offset kWorkspaceStart = dev_base_offset(4000000000);

/*! \brief total memory size */
constexpr uint64_t kMemorySize = 45000000000;

/*! \brief default size alignment */
constexpr int kDefaultSizeAlignment = 8;

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
 * \param rodata new rodata section address
 * \param data new data section address
 * \param bss new bss section address
 * \return relocated binary file contents
 */
std::string RelocateBinarySections(std::string binary_name,
                                   dev_addr text,
                                   dev_addr rodata,
                                   dev_addr data,
                                   dev_addr bss);

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
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
