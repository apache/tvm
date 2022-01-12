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

class TexturePool::Pool {
 public:
  Pool() = default;
  void* Alloc(Device dev, DeviceAPI* device, size_t width, size_t height, DLDataType type_hint) {
    Entry e;
    e.data = nullptr;
    if (free_list_.size() != 0) {
      int64_t req_size = height * width;
      Entry new_mem;
      int64_t min_added_size = std::numeric_limits<int64_t>::max();
      int64_t min_wasted_size = std::numeric_limits<int64_t>::max();
      std::vector<Entry>::iterator best_mem;
      for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        if (it->type.code != type_hint.code) {
          continue;
        }
        int64_t old_size = it->x * it->y;
        new_mem.x = std::max(it->x, width);
        new_mem.y = std::max(it->y, height);
        int64_t new_size = new_mem.x * new_mem.y;
        int64_t added_size = new_size - old_size;
        int64_t wasted_size = new_size - req_size;
        // Minimize added size first and wasted size thereafter
        if ((min_added_size > 0 && added_size < min_added_size) ||
            (min_added_size == 0 && wasted_size < min_wasted_size)) {
          min_added_size = added_size;
          min_wasted_size = wasted_size;
          best_mem = it;
        }
      }

      if (min_added_size == 0) {
        // use existing block
        e = *best_mem;
        free_list_.erase(best_mem);
      } else if (min_added_size <= req_size) {
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

  void Free(void* data) {
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
  void Release(Device dev, DeviceAPI* device) {
    for (auto& e : allocated_) {
      device->FreeDataSpace(dev, e.data);
    }
    for (auto& e : free_list_) {
      device->FreeDataSpace(dev, e.data);
    }
    allocated_.clear();
    free_list_.clear();
  }

 private:
  struct Entry {
    void* data;
    size_t x;
    size_t y;
    DLDataType type;
  };
  std::vector<Entry> free_list_;
  std::vector<Entry> allocated_;
};

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
    array_[dev.device_id] = new Pool();
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
