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
 * \file relay/backend/token_allocator.cc
 * \brief Token allocation classes for backend
 */

#include "token_allocator.h"

#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>

namespace tvm {
namespace relay {
constexpr auto Is2DStorage = runtime::IsTextureStorage;

/*
 * Mixed mode memory allocator
 */
size_t TokenAllocatorMixed::GetMemorySize(StorageToken* prototype) {
  TensorType ttype = prototype->ttype;
  ICHECK(ttype.defined());
  size_t size = 1;
  if (relay::Is2DStorage(prototype->virtual_device->memory_scope)) {
    size = GetSize2D(prototype);
  } else {
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ICHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
      ICHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
      size *= static_cast<size_t>(pval[0]);
    }
    size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
  }
  return size;
}

bool IsTargetContainsKey(StorageToken* tok, String key) {
  Target null_tgt{nullptr};
  if (null_tgt == tok->virtual_device->target) {
    return false;
  }
  auto prototype_keys = tok->virtual_device->target->GetKeys();
  return std::find(prototype_keys.begin(), prototype_keys.end(), key) != prototype_keys.end();
}

StorageToken* TokenAllocatorMixed::Request(StorageToken* prototype) {
  // calculate the size;
  size_t size = GetMemorySize(prototype);
  // search memory block in [size / match_range_, size * match_range_)
  if (match_range_ == 0) {
    return nullptr;
  }
  auto begin = free_.lower_bound(size / match_range_);
  auto mid = free_.lower_bound(size);
  auto end = free_.upper_bound(size * match_range_);
  // search for memory blocks larger than requested
  bool is_prototype_adreno = IsTargetContainsKey(prototype, "adreno");
  for (auto it = mid; it != end; ++it) {
    StorageToken* tok = it->second;
    // TODO(Siva): We need a additional ways of comparing VirtualDevice
    bool is_tok_adreno = IsTargetContainsKey(tok, "adreno");

    if (tok->is_compatible(*prototype) || (is_prototype_adreno && is_tok_adreno)) {
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      if (size > tok->max_bytes) {
        tok->max_bytes = size;
        tok->ttype = prototype->ttype;
      }
      tok->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      free_.erase(it);
      return tok;
    }
  }
  // then search for memory blocks smaller than requested space
  for (auto it = mid; it != begin;) {
    --it;
    StorageToken* tok = it->second;
    bool is_tok_adreno = IsTargetContainsKey(tok, "adreno");
    if (tok->is_compatible(*prototype) || (is_prototype_adreno && is_tok_adreno)) {
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      if (size > tok->max_bytes) {
        tok->max_bytes = size;
        tok->ttype = prototype->ttype;
      }
      tok->ref_counter = prototype->ref_counter;
      // erase from map and return
      free_.erase(it);
      return tok;
    }
  }
  return nullptr;
}

StorageToken* TokenAllocatorMixed::Alloc(StorageToken* prototype, int64_t storage_id) {
  size_t size = GetMemorySize(prototype);
  prototype->max_bytes = size;
  prototype->storage_id = storage_id;
  data_.push_back(prototype);
  return prototype;
}

void TokenAllocatorMixed::CheckForRelease(StorageToken* tok) {
  ICHECK_GE(tok->storage_id, 0);
  ICHECK_GE(tok->ref_counter, 0);
  if (tok->ref_counter == 0) {
    free_.insert({tok->max_bytes, tok});
  }
}

size_t TokenAllocatorMixed::GetSize2D(StorageToken* prototype) {
  TensorType ttype = prototype->ttype;
  ICHECK(ttype.defined());
  struct Shape {
    const Array<PrimExpr>& shape;
    int64_t operator[](size_t i) const { return *tir::as_const_int(shape[i]); }
    int size() { return this->shape.size(); }
  };
  auto shape = Shape{ttype->shape};
  int image_row_align =
      prototype->virtual_device->target->GetAttr<Integer>("image_base_address_alignment")
          .value_or(Integer(64))
          ->value;
  return runtime::GetTextureMemorySize<Shape>(shape, ttype->dtype.bits(), ttype->dtype.lanes(),
                                              prototype->virtual_device->memory_scope,
                                              image_row_align);
}

}  // namespace relay
}  // namespace tvm
