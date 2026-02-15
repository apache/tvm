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
 * \file tvm/support/serializer.h
 * \brief Serializer<T> specializations for tvm::support::Stream.
 *
 * Built-in support for:
 *   - Arithmetic types (endian-aware)
 *   - Enum types (via underlying arithmetic type)
 *   - std::string, std::vector<T>, std::pair<A,B>, std::unordered_map<K,V>
 *   - DLDataType, DLDevice
 *
 * Custom types with Save(Stream*)/Load(Stream*) methods should define
 * Serializer<Type> specializations directly.
 */
#ifndef TVM_SUPPORT_SERIALIZER_H_
#define TVM_SUPPORT_SERIALIZER_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/endian.h>
#include <tvm/support/io.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace support {

// ---- Arithmetic types ----
template <typename T>
struct Serializer<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const T& data) {
    if constexpr (TVM_FFI_IO_NO_ENDIAN_SWAP) {
      strm->Write(&data, sizeof(T));
    } else {
      T copy = data;
      ffi::ByteSwap(&copy, sizeof(T), 1);
      strm->Write(&copy, sizeof(T));
    }
  }

  static bool Read(Stream* strm, T* data) {
    bool ok = strm->Read(data, sizeof(T)) == sizeof(T);
    if constexpr (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
      ffi::ByteSwap(data, sizeof(T), 1);
    }
    return ok;
  }
};

// ---- Enum types (delegate to underlying arithmetic type) ----
template <typename T>
struct Serializer<T, std::enable_if_t<std::is_enum_v<T>>> {
  static constexpr bool enabled = true;
  using U = std::underlying_type_t<T>;

  static void Write(Stream* strm, const T& data) {
    Serializer<U>::Write(strm, static_cast<U>(data));
  }

  static bool Read(Stream* strm, T* data) {
    U val;
    if (!Serializer<U>::Read(strm, &val)) return false;
    *data = static_cast<T>(val);
    return true;
  }
};

// ---- std::string ----
template <>
struct Serializer<std::string> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const std::string& data) {
    uint64_t sz = static_cast<uint64_t>(data.size());
    Serializer<uint64_t>::Write(strm, sz);
    if (sz != 0) {
      strm->Write(data.data(), data.size());
    }
  }

  static bool Read(Stream* strm, std::string* data) {
    uint64_t sz;
    if (!Serializer<uint64_t>::Read(strm, &sz)) return false;
    data->resize(static_cast<size_t>(sz));
    if (sz != 0) {
      size_t nbytes = static_cast<size_t>(sz);
      return strm->Read(&(*data)[0], nbytes) == nbytes;
    }
    return true;
  }
};

// ---- std::vector<T> ----
template <typename T>
struct Serializer<std::vector<T>> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const std::vector<T>& vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    Serializer<uint64_t>::Write(strm, sz);
    if constexpr (std::is_arithmetic_v<T> && TVM_FFI_IO_NO_ENDIAN_SWAP) {
      if (sz != 0) {
        strm->Write(vec.data(), sizeof(T) * vec.size());
      }
    } else {
      for (const auto& v : vec) {
        Serializer<T>::Write(strm, v);
      }
    }
  }

  static bool Read(Stream* strm, std::vector<T>* vec) {
    uint64_t sz;
    if (!Serializer<uint64_t>::Read(strm, &sz)) return false;
    vec->resize(static_cast<size_t>(sz));
    if constexpr (std::is_arithmetic_v<T> && TVM_FFI_IO_NO_ENDIAN_SWAP) {
      if (sz != 0) {
        size_t nbytes = sizeof(T) * static_cast<size_t>(sz);
        return strm->Read(vec->data(), nbytes) == nbytes;
      }
      return true;
    } else {
      for (size_t i = 0; i < static_cast<size_t>(sz); ++i) {
        if (!Serializer<T>::Read(strm, &(*vec)[i])) return false;
      }
      return true;
    }
  }
};

// ---- std::pair<A, B> ----
template <typename A, typename B>
struct Serializer<std::pair<A, B>> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const std::pair<A, B>& data) {
    Serializer<A>::Write(strm, data.first);
    Serializer<B>::Write(strm, data.second);
  }

  static bool Read(Stream* strm, std::pair<A, B>* data) {
    return Serializer<A>::Read(strm, &data->first) && Serializer<B>::Read(strm, &data->second);
  }
};

// ---- std::unordered_map<K, V> ----
template <typename K, typename V>
struct Serializer<std::unordered_map<K, V>> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const std::unordered_map<K, V>& data) {
    std::vector<std::pair<K, V>> vec(data.begin(), data.end());
    Serializer<std::vector<std::pair<K, V>>>::Write(strm, vec);
  }

  static bool Read(Stream* strm, std::unordered_map<K, V>* data) {
    std::vector<std::pair<K, V>> vec;
    if (!Serializer<std::vector<std::pair<K, V>>>::Read(strm, &vec)) return false;
    data->clear();
    data->insert(vec.begin(), vec.end());
    return true;
  }
};

// ---- DLDataType ----
template <>
struct Serializer<DLDataType> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const DLDataType& dtype) {
    Serializer<uint8_t>::Write(strm, dtype.code);
    Serializer<uint8_t>::Write(strm, dtype.bits);
    Serializer<uint16_t>::Write(strm, dtype.lanes);
  }

  static bool Read(Stream* strm, DLDataType* dtype) {
    if (!Serializer<uint8_t>::Read(strm, &dtype->code)) return false;
    if (!Serializer<uint8_t>::Read(strm, &dtype->bits)) return false;
    if (!Serializer<uint16_t>::Read(strm, &dtype->lanes)) return false;
    return true;
  }
};

// ---- DLDevice ----
template <>
struct Serializer<DLDevice> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const DLDevice& dev) {
    int32_t device_type = static_cast<int32_t>(dev.device_type);
    Serializer<int32_t>::Write(strm, device_type);
    Serializer<int32_t>::Write(strm, dev.device_id);
  }

  static bool Read(Stream* strm, DLDevice* dev) {
    int32_t device_type = 0;
    if (!Serializer<int32_t>::Read(strm, &device_type)) return false;
    dev->device_type = static_cast<DLDeviceType>(device_type);
    if (!Serializer<int32_t>::Read(strm, &dev->device_id)) return false;
    return true;
  }
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_SERIALIZER_H_
