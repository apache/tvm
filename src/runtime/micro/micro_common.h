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
  kStack,
  kHeap,
  kWorkspace,
  kNumKinds,
};

/*! \brief default size alignment */
constexpr int kDefaultSizeAlignment = 8;

// TODO(weberlo): Do we only need a device location class? Think about pros/cons.
// It seems that offsets don't semantically fit in the class of device pointers.
// But the type safety guarantees from having all three subclasses is very
// helpful.  `DevBaseOffset` is the weirdest to have as a subclass, because it's
// not an address.

/*! \brief Base class for interfacing with device locations (pointers/offsets) */
class DeviceLocation {
 public:
  /*! \brief construct a location with value `value` */
  explicit DeviceLocation(std::uintptr_t value) : value_(value) {}

  /*! \brief default constructor */
  DeviceLocation() : value_(0) {}

  /*! \brief construct a null location */
  explicit DeviceLocation(std::nullptr_t value) : value_(0) {}

  /*! \brief destructor */
  virtual ~DeviceLocation() {}

  /*!
   * \brief get value of location
   * \return value of location
   */
  std::uintptr_t value() const { return value_; }

  /*!
   * \brief cast location to type `T`
   * \return casted result
   */
  template <typename T>
  T cast_to() const { return reinterpret_cast<T>(value_); }

  /*! \brief check if location is null */
  bool operator==(std::nullptr_t) const { return value_ == 0; }

  /*! \brief check if location is not null */
  bool operator!=(std::nullptr_t) const { return value_ != 0; }

 protected:
  /*! \brief raw value storing the location */
  std::uintptr_t value_;
};

class DevAddr;
class DevBaseAddr;
class DevBaseOffset;

/*! \brief absolute device address */
class DevAddr : public DeviceLocation {
 public:
  /*! \brief construct an absolute address with value `value` */
  explicit DevAddr(std::uintptr_t val) : DeviceLocation(val) {}

  /*! \brief default constructor */
  DevAddr() : DeviceLocation() {}

  /*! \brief construct a null absolute address */
  explicit DevAddr(std::nullptr_t val) : DeviceLocation(val) {}

  /*! \brief subtract a base address from this absolute address to get a base offset */
  DevBaseOffset operator-(DevBaseAddr base) const;

  /*! \brief add an integer to this absolute address to get a larger absolute address */
  DevAddr operator+(size_t n) const;

  /*! \brief mutably add an integer to this absolute address */
  DevAddr& operator+=(size_t n);

  /*! \brief subtract an integer from this absolute address to get a smaller absolute address */
  DevAddr operator-(size_t n) const;

  /*! \brief mutably subtract an integer from this absolute address */
  DevAddr& operator-=(size_t n);
};

/*! \brief base address of the device */
class DevBaseAddr : public DeviceLocation {
 public:
  /*! \brief construct a base address with value `value` */
  explicit DevBaseAddr(std::uintptr_t value) : DeviceLocation(value) {}

  /*! \brief default constructor */
  DevBaseAddr() : DeviceLocation() {}

  /*! \brief construct a null base address */
  explicit DevBaseAddr(std::nullptr_t value) : DeviceLocation(value) {}

  /*! \brief add a base offset to this base address to get an absolute address */
  DevAddr operator+(DevBaseOffset offset) const;
};

/*! \brief offset from device base address */
class DevBaseOffset : public DeviceLocation {
 public:
  /*! \brief construct a base offset with value `value` */
  explicit DevBaseOffset(std::uintptr_t value) : DeviceLocation(value) {}

  /*! \brief default constructor */
  DevBaseOffset() : DeviceLocation() {}

  /*! \brief construct a null base offset */
  explicit DevBaseOffset(std::nullptr_t value) : DeviceLocation(value) {}

  /*! \brief add this base offset to a base address to get an absolute address */
  DevAddr operator+(DevBaseAddr base) const;

  /*! \brief add an integer to this base offset to get a larger base offset */
  DevBaseOffset operator+(size_t n) const;

  /*! \brief mutably add an integer to this base offset */
  DevBaseOffset& operator+=(size_t n);

  /*! \brief subtract an integer from this base offset to get a smaller base offset */
  DevBaseOffset operator-(size_t n) const;

  /*! \brief mutably subtract an integer from this base offset */
  DevBaseOffset& operator-=(size_t n);
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
   * \param toolchain_prefix prefix of compiler toolchain to use
   */
  SymbolMap(const std::string& binary,
            DevBaseAddr base_addr,
            const std::string& toolchain_prefix) {
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
      map_[name] = DevAddr(addr) - base_addr;
      stream >> name;
      stream >> std::hex >> addr;
    }
  }

  /*!
   * \brief retrieve on-device offset for a symbol name
   * \param name name of the symbol
   * \return on-device offset of the symbol
   */
  DevBaseOffset operator[](const std::string& name) const {
    auto result = map_.find(name);
    CHECK(result != map_.end()) << "\"" << name << "\" not in symbol map";
    return result->second;
  }

 private:
  /*! \brief backing map */
  std::unordered_map<std::string, DevBaseOffset> map_;
};

/*! \brief struct containing start and size of a device memory region */
struct DevMemRegion {
  /*! \brief section start offset */
  DevBaseOffset start;
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

// TODO(weberlo): should this be here?
/*! \brief number of bytes in each page */
constexpr int kPageSize = 4096;

const DevBaseOffset kDeviceStart = DevBaseOffset(64);

/*!
 * \brief return default size of given section kind in bytes
 */
size_t GetDefaultSectionSize(SectionKind kind);

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
 * \param text new text section address
 * \param rodata new rodata section address
 * \param data new data section address
 * \param bss new bss section address
 * \param toolchain_prefix prefix of compiler toolchain to use
 * \return relocated binary file contents
 */
std::string RelocateBinarySections(const std::string& binary_name,
                                   DevAddr text,
                                   DevAddr rodata,
                                   DevAddr data,
                                   DevAddr bss,
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
 * \param align alignment of the returned size (default: 8)
 * \return size of the section if it exists, 0 otherwise
 */
size_t GetSectionSize(const std::string& binary_name,
                      SectionKind section,
                      const std::string& toolchain_prefix,
                      size_t align = kDefaultSizeAlignment);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
