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
#include "stripe_config.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "common.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

template <class T>
std::map<std::vector<T>, int> MultiplyCombinations(std::vector<std::map<T, int>> values) {
  if (values.size() == 1) {
    std::map<std::vector<T>, int> combs;
    for (const auto& it : values[0]) {
      combs[std::vector<T>(1, it.first)] = it.second;
    }
    return combs;
  }
  auto combs =
      MultiplyCombinations(std::vector<std::map<T, int>>(values.begin(), values.end() - 1));
  std::map<std::vector<T>, int> new_combs;
  for (const auto& val_it : values.back()) {
    for (const auto& comb_it : combs) {
      auto new_comb = std::vector<T>(comb_it.first);
      new_comb.push_back(val_it.first);
      new_combs[new_comb] = val_it.second * comb_it.second;
    }
  }
  return new_combs;
}

std::map<std::vector<int>, int> CountStripes(const StripeConfig& stripe_config,
                                             bool enable_sliding_window = false) {
  std::vector<std::map<int, int>> per_axis_sizes(stripe_config->GetOrder().size());
  for (size_t axis = 0; axis < stripe_config->GetOrder().size(); axis++) {
    int start = stripe_config->GetOffset()[axis];
    size_t stripe_count = static_cast<size_t>(stripe_config->GetStripes()[axis]);
    int stride = stripe_config->GetStrides()[axis];
    int shape = stripe_config->GetShape()[axis];
    int extent = stripe_config->GetExtent()[axis];
    int low;
    int high = std::numeric_limits<int>::min();
    for (size_t i = 0; i < stripe_count; i++) {
      // Calculate the 'non-edge case' sizes in one go to save effort
      if (!enable_sliding_window || i > 0) {
        if (start >= 0 && extent - shape - start >= 0 && stride > 0) {
          int whole_stripes =
              std::min(static_cast<int>(stripe_count - i), (extent - shape - start) / stride + 1);
          if (enable_sliding_window) {
            per_axis_sizes[axis][stride] += whole_stripes;
          } else {
            per_axis_sizes[axis][shape] += whole_stripes;
          }
          i += whole_stripes - 1;
          start += whole_stripes * stride;
          high = std::min(start - stride + shape, extent);
          continue;
        }
      }
      low = std::max(start, 0);
      if (enable_sliding_window) {
        low = std::max(low, high);
      }
      high = std::min(start + shape, extent);
      int size = high - low;
      if (size > 0) {
        per_axis_sizes[axis][size]++;
      }
      start += stride;
    }
  }
  return MultiplyCombinations(per_axis_sizes);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.CountStripes")
    .set_body_typed([](StripeConfig stripe_config, bool enable_sliding_window) {
      Map<Array<Integer>, Integer> ret;
      auto stripe_counts = CountStripes(stripe_config, enable_sliding_window);
      for (const auto& it : stripe_counts) {
        ret.Set(make_array(it.first), it.second);
      }
      return ret;
    });

void StripeConfigNode::VisitAttrs(AttrVisitor* v) {
  Array<Integer> tmp_arr = make_array(shape_);
  v->Visit("_shape", &tmp_arr);
  tmp_arr = make_array(extent_);
  v->Visit("_extent", &tmp_arr);
  tmp_arr = make_array(order_);
  v->Visit("_order", &tmp_arr);
  tmp_arr = make_array(stripes_);
  v->Visit("_stripes", &tmp_arr);
  tmp_arr = make_array(offset_);
  v->Visit("_offset", &tmp_arr);
  Array<FloatImm> tmp_float_arr = make_array(strides_);
  v->Visit("_strides", &tmp_float_arr);
  int64_t tmp_hash = static_cast<int64_t>(hash_);
  v->Visit("_hash", &tmp_hash);
}

void StripeConfigNode::ComputeHash_() {
  hash_ = hash_vector(shape_);
  hash_combine(&hash_, hash_vector(extent_));
  hash_combine(&hash_, hash_vector(strides_));
  hash_combine(&hash_, hash_vector(order_));
  hash_combine(&hash_, hash_vector(stripes_));
  hash_combine(&hash_, hash_vector(offset_));
}

StripeConfig::StripeConfig(const std::vector<int>& shape, const std::vector<int>& extent,
                           const std::vector<float>& strides, const std::vector<int>& order,
                           const std::vector<int>& stripes, const std::vector<int>& offset) {
  auto n = make_object<StripeConfigNode>();
  n->shape_ = std::move(shape);
  n->extent_ = std::move(extent);
  n->strides_ = std::move(strides);
  n->order_ = std::move(order);
  n->stripes_ = std::move(stripes);
  n->offset_ = std::move(offset);
  n->ComputeHash_();
  data_ = std::move(n);
}

inline bool StripeConfig::operator==(const StripeConfig& other) const {
  if (get() == other.get()) return true;
  if (get() == nullptr || other.get() == nullptr) return false;
  return ((*this)->shape_ == other->shape_ && (*this)->extent_ == other->extent_ &&
          (*this)->strides_ == other->strides_ && (*this)->order_ == other->order_ &&
          (*this)->stripes_ == other->stripes_ && (*this)->offset_ == other->offset_);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.StripeConfig")
    .set_body_typed([](Array<Integer> shape, Array<Integer> extent, Array<FloatImm> strides,
                       Array<Integer> order, Array<Integer> stripes, Array<Integer> offset) {
      std::vector<int> vshape = make_vector<int, Integer>(shape);
      std::vector<int> vextent = make_vector<int, Integer>(extent);
      std::vector<float> vstrides = make_vector<float, FloatImm>(strides);
      std::vector<int> vorder = make_vector<int, Integer>(order);
      std::vector<int> vstripes = make_vector<int, Integer>(stripes);
      std::vector<int> voffset = make_vector<int, Integer>(offset);
      return StripeConfig(vshape, vextent, vstrides, vorder, vstripes, voffset);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.StripeConfigEqual")
    .set_body_method(&StripeConfig::operator==);

TVM_REGISTER_NODE_TYPE(StripeConfigNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
