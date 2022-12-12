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

size_t TokenAllocator1D::GetMemorySize(StorageToken* prototype) {
  TensorType ttype = prototype->ttype;
  ICHECK(ttype.defined());
  size_t size = 1;
  for (IndexExpr dim : ttype->shape) {
    const int64_t* pval = tir::as_const_int(dim);
    ICHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
    ICHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
    size *= static_cast<size_t>(pval[0]);
  }
  size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
  return size;
}

StorageToken* TokenAllocator1D::Request(StorageToken* prototype) {
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
  for (auto it = mid; it != end; ++it) {
    StorageToken* tok = it->second;
    if (!tok->is_compatible(*prototype)) continue;
    ICHECK_EQ(tok->ref_counter, 0);
    // Use exect matching strategy
    tok->max_bytes = std::max(size, tok->max_bytes);
    tok->ref_counter = prototype->ref_counter;
    // find a exact match, erase from map and return
    free_.erase(it);
    return tok;
  }
  // then search for memory blocks smaller than requested space
  for (auto it = mid; it != begin;) {
    --it;
    StorageToken* tok = it->second;
    if (!tok->is_compatible(*prototype)) continue;
    ICHECK_EQ(tok->ref_counter, 0);
    // Use exect matching strategy
    tok->max_bytes = std::max(size, tok->max_bytes);
    tok->ref_counter = prototype->ref_counter;
    // erase from map and return
    free_.erase(it);
    return tok;
  }
  return nullptr;
}

StorageToken* TokenAllocator1D::Alloc(StorageToken* prototype, int64_t storage_id) {
  size_t size = GetMemorySize(prototype);
  prototype->max_bytes = size;
  prototype->storage_id = storage_id;
  data_.push_back(prototype);
  return prototype;
}

void TokenAllocator1D::CheckForRelease(StorageToken* tok) {
  ICHECK_GE(tok->storage_id, 0);
  ICHECK_GE(tok->ref_counter, 0);
  if (tok->ref_counter == 0) {
    free_.insert({tok->max_bytes, tok});
  }
}

StorageToken* TokenAllocator2D::Request(StorageToken* prototype) {
  auto shape = GetSize2D(prototype);
  const int64_t max_ratio = 5;
  int64_t min_added_size_x = std::numeric_limits<int64_t>::max();
  int64_t min_added_size_y = std::numeric_limits<int64_t>::max();
  int64_t min_wasted_size_x = std::numeric_limits<int64_t>::max();
  int64_t min_wasted_size_y = std::numeric_limits<int64_t>::max();
  int64_t best_storage_id = -1;
  MemBlock new_mem;
  for (int64_t free_id : free_list_) {
    MemBlock& cached = blocks_[free_id];
    // Can only reuse texture 2d blocks of the same type
    if (cached.token_->ttype->dtype != prototype->ttype->dtype) {
      continue;
    }
    // Can only reuse texture 2d blocks of the same scope
    // Because reusing textures with different memory scope may lead to
    // accuracy issues, because the data will be packed in a different way for
    // different memory scopes.
    if (cached.token_->virtual_device->memory_scope != prototype->virtual_device->memory_scope) {
      continue;
    }
    // avoid reusing too small and too big textures
    if (shape.width / cached.x_ > max_ratio || cached.x_ / shape.width > max_ratio ||
        shape.height / cached.y_ > max_ratio || cached.y_ / shape.height > max_ratio) {
      continue;
    }
    int64_t new_width = std::max(cached.x_, shape.width);
    int64_t new_height = std::max(cached.y_, shape.height);
    int64_t added_size_x = new_width - cached.x_;
    int64_t added_size_y = new_height - cached.y_;
    int64_t wasted_size_x = new_width - shape.width;
    int64_t wasted_size_y = new_height - shape.height;
    // Prioritize minimization of added size first, then minimize
    // wasted size among blocks which would not require expansion
    if ((min_added_size_x > 0 && added_size_x < min_added_size_x) ||
        (min_added_size_y > 0 && added_size_y < min_added_size_y) ||
        (min_added_size_x == added_size_x && wasted_size_x < min_wasted_size_x) ||
        (min_added_size_y == added_size_y && wasted_size_y < min_wasted_size_y)) {
      min_added_size_x = added_size_x;
      min_added_size_y = added_size_y;
      min_wasted_size_x = wasted_size_x;
      min_wasted_size_y = wasted_size_y;
      best_storage_id = free_id;
      new_mem.x_ = new_width;
      new_mem.y_ = new_height;
    }
  }

  if (min_added_size_x == 0 && min_added_size_y == 0) {
    // use existing block
    free_list_.erase(best_storage_id);
    blocks_[best_storage_id].token_->ref_counter += prototype->ref_counter;
    return blocks_[best_storage_id].token_;
  } else if (min_added_size_x <= shape.width || min_added_size_y <= shape.height) {
    // Reset the reference counter of the now live token
    free_list_.erase(best_storage_id);
    new_mem.token_ = prototype;
    new_mem.token_->ref_counter += 1;
    new_mem.token_->storage_id = best_storage_id;
    blocks_[best_storage_id] = new_mem;
    return new_mem.token_;
  }
  return nullptr;
}

StorageToken* TokenAllocator2D::Alloc(StorageToken* prototype, int64_t storage_id) {
  auto shape = GetSize2D(prototype);
  MemBlock block;
  block.x_ = shape.width;
  block.y_ = shape.height;
  prototype->storage_id = storage_id;
  block.token_ = prototype;
  blocks_[prototype->storage_id] = block;
  return prototype;
}

void TokenAllocator2D::CheckForRelease(StorageToken* tok) {
  ICHECK_GE(tok->storage_id, 0);
  ICHECK_GE(tok->ref_counter, 0);
  if (tok->ref_counter == 0) {
    free_list_.insert(tok->storage_id);
  }
}

runtime::Texture2DShape<int64_t> TokenAllocator2D::GetSize2D(StorageToken* prototype) {
  TensorType ttype = prototype->ttype;
  ICHECK(ttype.defined());
  size_t axis = runtime::DefaultTextureLayoutSeparator(ttype->shape.size(),
                                                       prototype->virtual_device->memory_scope);
  struct Shape {
    const Array<PrimExpr>& shape;
    int64_t operator[](size_t i) const { return *tir::as_const_int(shape[i]); }
  };
  return runtime::ApplyTexture2DFlattening<int64_t>(Shape{ttype->shape}, ttype->shape.size(), axis);
}

}  // namespace relay
}  // namespace tvm
