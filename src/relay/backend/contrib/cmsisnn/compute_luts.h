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
 * \file src/relay/backend/contrib/cmsisnn/compute_luts.h
 * \brief CMSIS-NN LUTs calculation functions
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPUTE_LUTS_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPUTE_LUTS_H_

#include <cstdint>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief Populates an int16 LUT based on the quantization parameters of its keys, values and
 * respective transformation function
 *
 * \param key_zero_point - zero point of table's keys
 * \param key_scale - scale of the table's keys
 * \param value_zero_point - zero point of table's values
 * \param value_scale - scale of the table's values
 * \param func - function pointer of the transformation performed by the LUT
 * \param steps - number of total values inside the table
 * \param lut - int16_t array storing the values of the LUT
 */
void CalculateLUTInt16(int key_zero_point, float key_scale, int value_zero_point, float value_scale,
                       float (*func)(float), const int steps, int16_t* lut);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPUTE_LUTS_H_
