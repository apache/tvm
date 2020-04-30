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
 * \file micro_common.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_COMMON_H_
#define TVM_RUNTIME_MICRO_MICRO_COMMON_H_

#include <stdio.h>

#include <tvm/runtime/registry.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace runtime {

/*!
 * \brief enum of device memory region sections
 *
 * The order in which the enum variants are defined also defines the order of
 * the sections in device memory.
 */
enum class SectionKind : size_t {
  kText = 0,
  kRodata,
  kData,
  kBss,
  kArgs,
  kHeap,
  kWorkspace,
  kStack,
  kNumKinds,
};

/*! \brief data type for word sizes */
class TargetWordSize {
 public:
  explicit TargetWordSize(size_t word_size_bits) : word_size_bits_{word_size_bits} {
    CHECK(word_size_bits == 32 || word_size_bits == 64)
      << "only 32-bit and 64-bit are supported now";
  }

  size_t bytes() const {
    return word_size_bits_ / 8;
  }

  size_t bits() const {
    return word_size_bits_;
  }

 private:
  size_t word_size_bits_;
};


/*! \brief class for storing values on varying target word sizes */
class TargetVal {
 private:
  size_t width_bits_;
  uint64_t value_;

 public:
  /*! \brief construct a TargetVal matching the size of the given integral argument */
  template<typename T, typename U = typename std::enable_if<std::is_integral<T>::value, T>::type>
  explicit constexpr TargetVal(T value) : TargetVal(sizeof(T) * 8, value) {}

  /*! \brief construct an uninitialized value */
  TargetVal() : width_bits_{0}, value_{0} {}

  /*! \brief construct a TargetVal with explicit size and value */
  TargetVal(size_t width_bits, uint64_t value) : width_bits_{width_bits} {
    CHECK(width_bits >= 8 &&
          width_bits <= 64 &&
          (width_bits & (width_bits - 1)) == 0)
      << "width_bits must be a power of 2 in [8, 64], got " << width_bits;
    value_ = value & Bitmask();
  }

  bool IsInitialized() const { return width_bits_ != 0; }

  size_t width_bits() const {
    CHECK(IsInitialized()) << "TargetVal is not initialized";
    return width_bits_;
  }

  uint64_t Bitmask() const {
    CHECK(IsInitialized()) << "TargetVal is not initialized";

    if (width_bits_ == 64) {
      return ~0UL;
    } else {
      return (1UL << width_bits_) - 1;
    }
  }

  uint32_t uint32() const {
    CHECK(IsInitialized()) << "TargetVal is not initialized";
    CHECK(width_bits_ <= 32) << "TargetVal: requested 32-bit value, actual width is "
                             << width_bits_;
    return uint32_t(value_ & Bitmask());
  }

  uint64_t uint64() const {
    CHECK(IsInitialized()) << "TargetVal is not initialized";
    return value_;
  }

  TargetVal& operator=(const TargetVal& other) {
    CHECK(other.IsInitialized()) << "Cannot assign an uninitialized TargetVal";

    if (!IsInitialized()) {
      width_bits_ = other.width_bits_;
    }

    CHECK(width_bits_ >= other.width_bits_)
      << "Cannot assign TargetVal with width " << other.width_bits_
      << "bits to TargetVal with width " << width_bits_ << "bits";

    value_ = other.value_ & Bitmask();
    return *this;
  }
};

// TODO(weberlo, areusch): just get rid of `TargetPtr`.
/*! \brief absolute device address */
class TargetPtr {
 public:
  /*! \brief construct a device address with variable-length value `value` */
  TargetPtr(TargetWordSize word_size, std::uint64_t value) :
      value_(TargetVal(word_size.bits(), value)) {}

  /*! \brief construct a null address */
  TargetPtr(TargetWordSize word_size, std::nullptr_t value) :
      value_{TargetVal(word_size.bits(), 0)} {}

  /*! \brief construct an uninitialized pointer whose word_size can be changed once */
  TargetPtr() = default;

  /*! \brief construct a device address using the given TargetVal */
  explicit TargetPtr(const TargetVal& value) : value_{value} {}

  /*! \brief destructor */
  ~TargetPtr() {}

  /*!
   * \brief get value of pointer
   * \return value of pointer
   */
  TargetVal value() const { return value_; }

  /*!
   * \brief cast location to type `T`
   * \return casted result
   */
  template <typename T>
  T cast_to() const { return reinterpret_cast<T>(value_.uint64()); }

  /*! \brief check if location is null */
  bool operator==(std::nullptr_t) const { return value_.uint64() == 0; }

  /*! \brief check if location is not null */
  bool operator!=(std::nullptr_t) const { return value_.uint64() != 0; }

  /*! \brief add an integer to this absolute address to get a larger absolute address */
  TargetPtr operator+(size_t n) const {
    return TargetPtr(TargetWordSize(value_.width_bits()), value_.uint64() + n);
  }

  /*! \brief mutably add an integer to this absolute address */
  TargetPtr& operator+=(size_t n) {
    value_ = TargetVal(value_.width_bits(), value_.uint64() + n);
    return *this;
  }

  /*! \brief subtract an integer from this absolute address to get a smaller absolute address */
  TargetPtr operator-(size_t n) const {
    return TargetPtr(TargetWordSize(value_.width_bits()), value_.uint64() - n);
  }

  /*! \brief mutably subtract an integer from this absolute address */
  TargetPtr& operator-=(size_t n) {
    value_ = TargetVal(value_.width_bits(), value_.uint64() - n);
    return *this;
  }

 private:
  /*! \brief raw value storing the pointer */
  TargetVal value_;
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
   * \param toolchain_prefix prefix of compiler toolchain to use
   */
  SymbolMap(const std::string& binary,
            const std::string& toolchain_prefix,
            TargetWordSize word_size) {
    const auto* f = Registry::Get("tvm_callback_get_symbol_map");
    CHECK(f != nullptr) << "require tvm_callback_get_symbol_map to exist in registry";
    TVMByteArray arr;
    arr.data = &binary[0];
    arr.size = binary.length();
    std::string map_str = (*f)(arr, toolchain_prefix);
    // Parse symbols and addresses from returned string.
    std::stringstream stream;
    stream << map_str;
    std::string name;
    std::uintptr_t addr;
    stream >> name;
    stream >> std::hex >> addr;
    while (stream) {
      map_.emplace(std::make_pair(name, TargetPtr(word_size, addr)));
      stream >> name;
      stream >> std::hex >> addr;
    }
  }

  /*!
   * \brief retrieve on-device offset for a symbol name
   * \param name name of the symbol
   * \return on-device offset of the symbol
   */
  TargetPtr operator[](const std::string& name) const {
    auto result = map_.find(name);
    CHECK(result != map_.end()) << "\"" << name << "\" not in symbol map";
    return result->second;
  }

  bool HasSymbol(const std::string& name) const {
    return map_.find(name) != map_.end();
  }

  void Dump(std::ostream& stream) const {
    for (auto e : map_) {
      stream << "Entry:" << e.first << std::endl;
    }
  }

 private:
  /*! \brief backing map */
  std::unordered_map<std::string, TargetPtr> map_;
};

/*! \brief struct containing start and size of a device memory region */
struct DevMemRegion {
  /*! \brief section start offset */
  TargetPtr start;
  /*! \brief size of section */
  size_t size;
};

/*! \brief struct containing section locations and symbol mappings */
struct BinaryInfo {
  /*! \brief text section region */
  DevMemRegion text_section;
  /*! \brief rodata section region */
  DevMemRegion rodata_section;
  /*! \brief data section region */
  DevMemRegion data_section;
  /*! \brief bss section region */
  DevMemRegion bss_section;
  /*! \brief symbol map to offsets */
  SymbolMap symbol_map;
};

struct BinaryContents {
  BinaryInfo binary_info;
  std::string text_contents;
  std::string rodata_contents;
  std::string data_contents;
  std::string bss_contents;
};

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

/*!
 * \brief links binary by repositioning section addresses
 * \param binary_name input binary filename
 * \param word_size word size on the target machine
 * \param text_start text section address
 * \param rodata_start rodata section address
 * \param data_start data section address
 * \param bss_start bss section address
 * \param stack_end stack section end address
 * \param toolchain_prefix prefix of compiler toolchain to use
 * \return relocated binary file contents
 */
std::string RelocateBinarySections(
    const std::string& binary_path,
    TargetWordSize word_size,
    TargetPtr text_start,
    TargetPtr rodata_start,
    TargetPtr data_start,
    TargetPtr bss_start,
    TargetPtr stack_end,
    const std::string& toolchain_prefix);

/*!
 * \brief reads section from binary
 * \param binary input binary contents
 * \param section section type to be read
 * \param toolchain_prefix prefix of compiler toolchain to use
 * \return contents of the section
 */
std::string ReadSection(const std::string& binary,
                        SectionKind section,
                        const std::string& toolchain_prefix);

/*!
 * \brief finds size of the section in the binary
 * \param binary input binary contents
 * \param section section type
 * \param toolchain_prefix prefix of compiler toolchain to use
 * \param word_size word size of the target, for alignment
 * \return size of the section if it exists, 0 otherwise
 */
size_t GetSectionSize(const std::string& binary_name,
                      SectionKind section,
                      const std::string& toolchain_prefix,
                      TargetWordSize word_size);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
