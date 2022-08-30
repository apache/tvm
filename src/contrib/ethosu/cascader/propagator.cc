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
#include "propagator.h"

#include <tvm/relay/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>

#include <utility>
#include <vector>

#include "common.h"
#include "stripe_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void PropagatorNode::VisitAttrs(AttrVisitor* v) {
  Array<Array<FloatImm>> tmp_transform;
  for (const auto& vec : transform_) {
    tmp_transform.push_back(make_array(vec));
  }
  v->Visit("_transform", &tmp_transform);
  Array<Integer> tmp_arr = make_array(offset_);
  v->Visit("_offset", &tmp_arr);
}

Propagator::Propagator(const std::vector<std::vector<float>>& transform,
                       const std::vector<int>& offset) {
  auto n = make_object<PropagatorNode>();
  size_t rows = transform.size();
  ICHECK_GT(rows, 0) << "The transform matrix must have at least 1 row.";
  size_t columns = transform[0].size();
  for (const auto& row : transform) {
    ICHECK_EQ(row.size(), columns)
        << "All rows of the transform matrix must be of the same length.";
  }
  ICHECK_EQ(offset.size(), rows - 1)
      << "The offset vector length must be equal to the transform matrix rows - 1.";
  n->transform_ = std::move(transform);
  n->offset_ = std::move(offset);
  data_ = std::move(n);
}

StripeConfig PropagatorNode::propagate(const StripeConfig& stripe_config) const {
  size_t input_dimensions = transform_[0].size() - 1;
  size_t output_dimensions = transform_.size() - 1;
  auto n = make_object<StripeConfigNode>();
  n->shape_.resize(output_dimensions);
  n->extent_.resize(output_dimensions);
  n->strides_.resize(output_dimensions);
  n->order_.resize(output_dimensions);
  n->stripes_.resize(output_dimensions);
  n->offset_.resize(output_dimensions);
  for (size_t i = 0; i < output_dimensions; i++) {
    float new_shape_acc{};
    float new_extent_acc{};
    const float* row = &transform_[i][0];
    for (size_t j = 0; j < input_dimensions; j++) {
      new_shape_acc += row[j] * stripe_config->shape_[j];
      new_extent_acc += row[j] * stripe_config->extent_[j];
      n->strides_[i] += row[j] * stripe_config->strides_[j];
      // Order, stripes and offset should only get re-ordered, so we only
      // care about whether or not transform elements are non-zero.
      int non_zero = row[j] != 0;
      n->order_[i] += non_zero * stripe_config->order_[j];
      n->stripes_[i] += non_zero * stripe_config->stripes_[j];
      n->offset_[i] += non_zero * stripe_config->offset_[j];
    }
    // Shape and extent gain an additional constant term
    new_shape_acc += row[input_dimensions];
    new_extent_acc += row[input_dimensions];
    // Shape and extent are ceil-rounded back to integers
    n->shape_[i] = std::ceil(new_shape_acc);
    n->extent_[i] += std::ceil(new_extent_acc);
    // Apply the offset
    n->offset_[i] += offset_[i];
    // No axis can have '0 stripes', so change all 0 elements to 1
    n->stripes_[i] = n->stripes_[i] == 0 ? 1 : n->stripes_[i];
  }
  // Remember to compute the hash
  n->ComputeHash_();
  return StripeConfig(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.Propagator")
    .set_body_typed([](Array<Array<FloatImm>> transform, Array<Integer> offset) {
      std::vector<std::vector<float>> vtransform;
      for (const auto& vec : transform) {
        vtransform.push_back(make_vector<float, FloatImm>(vec));
      }
      std::vector<int> voffset = make_vector<int, Integer>(offset);
      return Propagator(vtransform, voffset);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PropagatorPropagate")
    .set_body_typed([](Propagator propagator, StripeConfig stripe_config) {
      return propagator->propagate(stripe_config);
    });

TVM_REGISTER_NODE_TYPE(PropagatorNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
