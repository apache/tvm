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
 * \file texture_pool.h
 * \brief Texture pool utility.
 */
#include <limits>
#include <memory>

#include "../texture.h"

namespace tvm {
namespace runtime {

void* Pool2D::Alloc(Device dev, DeviceAPI* device, size_t width, size_t height,
                    DLDataType type_hint) {
  Entry e;
  Entry new_mem;
  // Processed several experiments and found that when we are trying to fit
  // small texture to too big texture then it may lead to the performance
  // degradation.
  // Coefficient at 5 looks like robust variant for reusing textures.
  const int64_t max_ratio = 5;
  e.data = nullptr;
  std::vector<Entry>::iterator best_mem;
  if (free_list_.size() != 0) {
    int64_t min_added_size_x = std::numeric_limits<int64_t>::max();
    int64_t min_added_size_y = std::numeric_limits<int64_t>::max();
    int64_t min_wasted_size_x = std::numeric_limits<int64_t>::max();
    int64_t min_wasted_size_y = std::numeric_limits<int64_t>::max();
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
      if (it->type.code != type_hint.code) {
        continue;
      }
      // avoid reusing too small and too big textures
      if (width / it->x > max_ratio || it->x / width > max_ratio || height / it->y > max_ratio ||
          it->y / height > max_ratio) {
        continue;
      }
      int64_t new_width = std::max(it->x, width);
      int64_t new_height = std::max(it->y, height);
      int64_t added_size_x = new_width - it->x;
      int64_t added_size_y = new_height - it->y;
      int64_t wasted_size_x = new_width - width;
      int64_t wasted_size_y = new_height - height;
      // Minimize added size first and wasted size thereafter
      if ((min_added_size_x > 0 && added_size_x < min_added_size_x) ||
          (min_added_size_y > 0 && added_size_y < min_added_size_y) ||
          (min_added_size_x == added_size_x && wasted_size_x < min_wasted_size_x) ||
          (min_added_size_y == added_size_y && wasted_size_y < min_wasted_size_y)) {
        min_added_size_x = added_size_x;
        min_added_size_y = added_size_y;
        min_wasted_size_x = wasted_size_x;
        min_wasted_size_y = wasted_size_y;
        best_mem = it;
        new_mem.x = new_width;
        new_mem.y = new_height;
      }
    }

    if (min_added_size_x == 0 && min_added_size_y == 0) {
      // use existing block
      e = *best_mem;
      free_list_.erase(best_mem);
    } else if (static_cast<size_t>(min_added_size_x) <= width ||
               static_cast<size_t>(min_added_size_y) <= height) {
      // if added size is less or equal to
      // what is needed by alloc, then grow entry
      device->FreeDataSpace(dev, best_mem->data);
      free_list_.erase(best_mem);
      new_mem.type = type_hint;
      std::vector<int64_t> shape{int64_t(new_mem.y), int64_t(new_mem.x), 4};
      new_mem.data = device->AllocDataSpace(dev, shape.size(), shape.data(), new_mem.type,
                                            Optional<String>("global.texture"));
      e = new_mem;
    }
  }

  if (e.data == nullptr) {
    // create new block
    std::vector<int64_t> shape{int64_t(height), int64_t(width), 4};
    e.data = device->AllocDataSpace(dev, shape.size(), shape.data(), type_hint,
                                    Optional<String>("global.texture"));
    e.x = width;
    e.y = height;
    e.type = type_hint;
  }

  allocated_.push_back(e);
  return e.data;
}

void Pool2D::Free(void* data) {
  Entry e;
  if (allocated_.back().data == data) {
    // quick path, last allocated.
    e = allocated_.back();
    allocated_.pop_back();
  } else {
    int index = static_cast<int>(allocated_.size()) - 2;
    for (; index >= 0 && allocated_[index].data != data; --index) {
    }
    ICHECK_GE(index, 0) << "Attempt to free texture that has not been allocated";
    e = allocated_[index];
    allocated_.erase(allocated_.begin() + index);
  }
  free_list_.push_back(e);
}

// Release all resources immediately
void Pool2D::Release(Device dev, DeviceAPI* device) {
  for (auto& e : allocated_) {
    device->FreeDataSpace(dev, e.data);
  }
  for (auto& e : free_list_) {
    device->FreeDataSpace(dev, e.data);
  }
  allocated_.clear();
  free_list_.clear();
}

TexturePool::TexturePool(DLDeviceType device_type, DeviceAPI* device)
    : device_type_(device_type), device_(device) {}

TexturePool::~TexturePool() {
  for (size_t i = 0; i < array_.size(); ++i) {
    if (array_[i] != nullptr) {
      Device dev;
      dev.device_type = device_type_;
      dev.device_id = static_cast<int>(i);
      array_[i]->Release(dev, device_);
      delete array_[i];
    }
  }
}

void* TexturePool::AllocTexture(Device dev, size_t width, size_t height, DLDataType type_hint) {
  if (static_cast<size_t>(dev.device_id) >= array_.size()) {
    array_.resize(dev.device_id + 1, nullptr);
  }
  if (array_[dev.device_id] == nullptr) {
    array_[dev.device_id] = new Pool2D();
  }
  return array_[dev.device_id]->Alloc(dev, device_, width, height, type_hint);
}

void TexturePool::FreeTexture(Device dev, void* ptr) {
  ICHECK(static_cast<size_t>(dev.device_id) < array_.size() && array_[dev.device_id] != nullptr)
      << "Attempt to free texture from null texture pool";
  array_[dev.device_id]->Free(ptr);
}

}  // namespace runtime
}  // namespace tvm
