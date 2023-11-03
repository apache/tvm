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
 * \file src/relay/backend/contrib/cmsisnn/compute_luts.cc
 * \brief Creates LUTs for operators in different bit formats for accelerating computations.
 */

#include "compute_luts.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

void CalculateLUTInt16(int key_zero_point, float key_scale, int value_zero_point, float value_scale,
                       float (*func)(float), const int steps, int16_t* lut) {
  const float value_min = static_cast<float>(std::numeric_limits<int16_t>::min());
  const float value_max = static_cast<float>(std::numeric_limits<int16_t>::max());
  const float key_min_deq = key_scale * (std::numeric_limits<int16_t>::min() - key_zero_point);
  const float key_max_deq = key_scale * (std::numeric_limits<int16_t>::max() - key_zero_point);
  const float value_min_deq =
      value_scale * (std::numeric_limits<int16_t>::min() - value_zero_point);
  const float value_max_deq =
      value_scale * (std::numeric_limits<int16_t>::max() - value_zero_point);

  const float step_size_deq = (key_max_deq - key_min_deq) / (steps - 1);
  const float half_step_size_deq = step_size_deq / 2;

  const float value_inv_quantizing =
      (std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min() + 1) /
      (value_max_deq - value_min_deq);

  for (int i = 0; i < steps - 1; i++) {
    float value_deq = func(key_min_deq + i * step_size_deq);
    float mid_value_deq = func(key_min_deq + i * step_size_deq + half_step_size_deq);
    float next_value_deq = func(key_min_deq + (i + 1) * step_size_deq);

    float value = std::round(value_deq * value_inv_quantizing);
    float mid_value = std::round(mid_value_deq * value_inv_quantizing);
    float next_value = std::round(next_value_deq * value_inv_quantizing);
    float mid_iterp_value = std::round((value + next_value) / 2);

    float mid_err = mid_iterp_value - mid_value;
    float bias = std::round(mid_err / 2);

    lut[i] = static_cast<int16_t>(std::max(std::min(value - bias, value_max), value_min));
  }

  lut[steps - 1] = static_cast<int16_t>(
      std::max(std::min(func(value_max_deq) * value_inv_quantizing, value_max), value_min));
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
