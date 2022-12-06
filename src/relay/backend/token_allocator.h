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
 * \file relay/backend/token_allocator.h
 * \brief Token allocation classes for backend
 */
#ifndef TVM_RELAY_BACKEND_TOKEN_ALLOCATOR_H_
#define TVM_RELAY_BACKEND_TOKEN_ALLOCATOR_H_

#include <tvm/relay/type.h>
#include <tvm/target/virtual_device.h>

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../runtime/texture.h"

namespace tvm {
namespace relay {

/*! A representation of a block of memory required at runtime on some device. */
struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type. */
  TensorType ttype{nullptr};
  /*! \brief VirtualDevice on which the memory will reside. */
  VirtualDevice virtual_device = VirtualDevice::FullyUnconstrained();
  /*! \brief The storage id */
  int64_t storage_id{-1};

  bool is_valid() const { return !virtual_device->IsFullyUnconstrained(); }

  bool is_compatible(const StorageToken& that) const {
    return virtual_device == that.virtual_device;
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "{storage_id: " << storage_id << ", max_bytes: " << max_bytes
       << ", ttype: " << PrettyPrint(ttype) << ", virtual_device: " << virtual_device << "}";
    return os.str();
  }
};

/**
 * @brief Memory manager for flattened 1d memory (buffers)
 */
class TokenAllocator1D {
 public:
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }

  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   *
   * TODO(mbs): Gf GetMemorySizeBytes in aot_executor_codegen.cc,
   * CalculateRelayExprSizeBytes in utils.cc
   */
  size_t GetMemorySize(StorageToken* prototype);
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype);
  /*!
   * \brief Alloacte a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, int64_t storage_id);
  /*!
   * \brief Check if we can release token.
   * \param tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok);

 private:
  // scale used for rough match
  const size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;
};

/**
 * @brief Memory manager for 2d memory (textures)
 */
class TokenAllocator2D {
 public:
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype);
  /*!
   * \brief Alloacte a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, int64_t storage_id);
  /*!
   * \brief Check if we can release token.
   * \param tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok);
  /*!
   * \brief Get the texture 2d size requirement
   * \param prototype The prototype token.
   * \return The required texture 2d memory size in (width, height, channel).
   */
  runtime::Texture2DShape<int64_t> GetSize2D(StorageToken* prototype);

 protected:
  struct MemBlock {
    StorageToken* token_;
    int64_t x_;
    int64_t y_;
  };

  std::unordered_map<int64_t, MemBlock> blocks_;
  std::unordered_set<int64_t> free_list_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TOKEN_ALLOCATOR_H_
