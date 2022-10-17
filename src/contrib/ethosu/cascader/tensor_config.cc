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
#include "tensor_config.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void MemoryRegionNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("name", &name);
  v->Visit("size", &size);
  v->Visit("read_bandwidth", &read_bandwidth);
  v->Visit("write_bandwidth", &write_bandwidth);
  v->Visit("read_latency", &read_latency);
  v->Visit("write_latency", &write_latency);
  v->Visit("burst_length", &burst_length);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.MemoryRegion")
    .set_body_typed([](String name, int size, int read_bandwidth, int write_bandwidth,
                       int read_latency, int write_latency, int burst_length) {
      return MemoryRegion(name, size, read_bandwidth, write_bandwidth, read_latency, write_latency,
                          burst_length);
    });

TVM_REGISTER_NODE_TYPE(MemoryRegionNode);

void TensorConfigNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("_tensor", &tensor_);
  v->Visit("_home_region", &home_region_);
  int state = static_cast<int>(state_);
  v->Visit("_state", &state);
  int buffer_mode = static_cast<int>(buffer_mode_);
  v->Visit("_buffer_mode", &buffer_mode);
  Array<StripeConfig> tmp_arr(stripe_configs_);
  v->Visit("_stripe_configs", &tmp_arr);
  v->Visit("_copy_tensor", &copy_tensor_);
  v->Visit("_copy_region", &copy_region_);
  int64_t tmp_hash = static_cast<int64_t>(hash_);
  v->Visit("_hash", &tmp_hash);
}

int TensorConfigNode::GetBufferSize() const {
  if (buffer_mode_ == BufferMode::RECOMPUTE) {
    return GetRecomputeBufferSize_();
  } else {
    return GetRollingBufferSize_();
  }
}

void TensorConfigNode::ComputeHash_() {
  hash_ = ObjectHash()(tensor_);
  hash_combine(&hash_, std::hash<std::string>()(home_region_->name));
  hash_combine(&hash_, std::hash<int>()(static_cast<int>(state_)));
  hash_combine(&hash_, std::hash<int>()(static_cast<int>(buffer_mode_)));
  hash_combine(&hash_, hash_vector(stripe_configs_));
  hash_combine(&hash_, std::hash<bool>()(copy_tensor_));
  hash_combine(&hash_, std::hash<std::string>()(copy_region_->name));
}

int TensorConfigNode::GetRecomputeBufferSize_() const {
  size_t buffer_size = 0;
  for (const auto& stripe_config : stripe_configs_) {
    buffer_size += mul_reduce(stripe_config->GetShape());
  }
  return buffer_size * tensor_->GetDataType().bytes() * tensor_->GetCompressionRatio();
}

int TensorConfigNode::GetRollingBufferSize_() const {
  int buffer_size = 0;
  for (const auto& stripe_config : stripe_configs_) {
    int rolling_axis = -1;
    for (size_t i = 0; i < stripe_config->GetOrder().size(); i++) {
      // The axis must be striped (> 1 stripes) and ordered (order != 0)
      if (stripe_config->GetStripes()[i] > 1 && stripe_config->GetOrder()[i] != 0) {
        // If we've yet to find a possible rolling axis, use this one
        if (rolling_axis == -1) {
          rolling_axis = i;
          continue;
        }
        // Otherwise, replace the rolling axis if the current axis has an earlier order
        if (stripe_config->GetOrder()[i] < stripe_config->GetOrder()[rolling_axis]) {
          rolling_axis = i;
        }
      }
    }
    // If we didn't find a rolling axis, just use axis 0
    if (rolling_axis == -1) {
      rolling_axis = 0;
    }
    int rolling_size = 1;
    for (size_t i = 0; i < tensor_->GetShape().size(); i++) {
      if (static_cast<int>(i) == rolling_axis) {
        rolling_size *= stripe_config->GetShape()[i];
      } else {
        rolling_size *= tensor_->GetShape()[i];
      }
    }
    buffer_size += rolling_size;
  }
  return buffer_size * tensor_->GetDataType().bytes() * tensor_->GetCompressionRatio();
}

TensorConfig::TensorConfig(const Tensor& tensor, const MemoryRegion& home_region,
                           TensorConfigState state, BufferMode buffer_mode,
                           const std::vector<StripeConfig>& stripe_configs, bool copy_tensor,
                           const MemoryRegion& copy_region) {
  auto n = make_object<TensorConfigNode>();
  n->tensor_ = std::move(tensor);
  n->home_region_ = std::move(home_region);
  n->state_ = state;
  n->buffer_mode_ = buffer_mode;
  n->stripe_configs_ = std::move(stripe_configs);
  n->copy_tensor_ = copy_tensor;
  n->copy_region_ = std::move(copy_region);
  n->ComputeHash_();
  data_ = std::move(n);
}

inline bool TensorConfig::operator==(const TensorConfig& other) const {
  if (get() == other.get()) return true;
  if (get() == nullptr || other.get() == nullptr) return false;
  if ((*this)->tensor_ == other->tensor_ && (*this)->home_region_ == other->home_region_ &&
      (*this)->state_ == other->state_ && (*this)->buffer_mode_ == other->buffer_mode_ &&
      (*this)->stripe_configs_ == other->stripe_configs_ &&
      (*this)->copy_tensor_ == other->copy_tensor_ &&
      (*this)->copy_region_ == other->copy_region_) {
    return true;
  }
  return false;
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.TensorConfig")
    .set_body_typed([](Tensor tensor, MemoryRegion home_region, int state, int buffer_mode,
                       Array<StripeConfig> stripe_configs, bool copy_tensor,
                       MemoryRegion copy_region) {
      TensorConfigState estate = static_cast<TensorConfigState>(state);
      BufferMode ebuffer_mode = static_cast<BufferMode>(buffer_mode);
      std::vector<StripeConfig> vstripe_configs(stripe_configs.begin(), stripe_configs.end());
      return TensorConfig(tensor, home_region, estate, ebuffer_mode, vstripe_configs, copy_tensor,
                          copy_region);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.TensorConfigEqual")
    .set_body_method(&TensorConfig::operator==);

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.TensorConfigGetBufferSize")
    .set_body_method<TensorConfig>(&TensorConfigNode::GetBufferSize);

TVM_REGISTER_NODE_TYPE(TensorConfigNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
