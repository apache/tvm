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

  /*! \brief construct a null location */
  DeviceLocation() : value_(0) {}

  /*! \brief construct a null location */
  explicit DeviceLocation(std::nullptr_t value) : value_(0) {}

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

  bool operator==(std::nullptr_t) const { return value_ == 0; }
  bool operator!=(std::nullptr_t) const { return value_ != 0; }

 protected:
  std::uintptr_t value_;
};

// TODO(weberlo): Finish docs

class DevAddr;
class DevBaseAddr;
class DevBaseOffset;

/*! \brief absolute device address */
class DevAddr : public DeviceLocation {
 public:
  explicit DevAddr(std::uintptr_t val) : DeviceLocation(val) {}

  DevAddr() : DeviceLocation() {}

  explicit DevAddr(std::nullptr_t val) : DeviceLocation(val) {}

  DevBaseOffset operator-(DevBaseAddr base);
  DevAddr operator+(size_t n);
};

/*! \brief base address of the device */
class DevBaseAddr : public DeviceLocation {
 public:
  explicit DevBaseAddr(std::uintptr_t val) : DeviceLocation(val) {}

  DevBaseAddr() : DeviceLocation() {}

  explicit DevBaseAddr(std::nullptr_t val) : DeviceLocation(val) {}

  DevAddr operator+(DevBaseOffset offset);
};

/*! \brief offset from device base address */
class DevBaseOffset : public DeviceLocation {
 public:
  explicit DevBaseOffset(std::uintptr_t val) : DeviceLocation(val) {}

  DevBaseOffset() : DeviceLocation() {}

  explicit DevBaseOffset(std::nullptr_t val) : DeviceLocation(val) {}

  DevAddr operator+(DevBaseAddr base);
  DevBaseOffset operator+(size_t n);
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
  SymbolMap(std::string binary, DevBaseAddr base_addr) {
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
  DevBaseOffset operator[](std::string name) {
    auto result = map_.find(name);
    CHECK(result != map_.end()) << "\"" << name << "\" not in symbol map";
    return result->second;
  }

 private:
  /*! \brief backing map */
  std::unordered_map<std::string, DevBaseOffset> map_;
};

/*! \brief struct containing section location info */
struct SectionLocation {
  /*! \brief section start offset */
  DevBaseOffset start;
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
const DevBaseOffset kTextStart = DevBaseOffset(64);

/*! \brief memory offset at which rodata section starts  */
const DevBaseOffset kRodataStart = DevBaseOffset(500000000);

/*! \brief memory offset at which data section starts  */
const DevBaseOffset kDataStart = DevBaseOffset(1000000000);

/*! \brief memory offset at which bss section starts  */
const DevBaseOffset kBssStart = DevBaseOffset(1500000000);

/*! \brief memory offset at which args section starts  */
const DevBaseOffset kArgsStart = DevBaseOffset(2000000000);

/*! \brief memory offset at which stack section starts  */
const DevBaseOffset kStackStart = DevBaseOffset(3000000000);

/*! \brief memory offset at which heap section starts  */
const DevBaseOffset kHeapStart = DevBaseOffset(3500000000);

/*! \brief memory offset at which workspace section starts  */
const DevBaseOffset kWorkspaceStart = DevBaseOffset(4000000000);

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
                                   DevAddr text,
                                   DevAddr rodata,
                                   DevAddr data,
                                   DevAddr bss);

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
size_t GetSectionSize(std::string binary_name,
                      SectionKind section,
                      size_t align = kDefaultSizeAlignment);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_COMMON_H_
