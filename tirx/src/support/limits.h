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
 * \file src/support/limits.h
 * \brief Numeric range limits for TVM low-precision floating-point dtypes.
 */
#ifndef TVM_SUPPORT_LIMITS_H_
#define TVM_SUPPORT_LIMITS_H_

namespace tvm {
namespace support {

// 2^15 * (1 + 1023/1024)
// See https://en.wikipedia.org/wiki/Half-precision_floating-point_format
constexpr double kMaxFloat16 = 65504.0;

// 2^127 * (1 + 127/128)
// See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
constexpr double kMaxBFloat16 = 3.895313892515354759047080037148786688e38;

// 2^15 * (1 + 3/4)
// See https://arxiv.org/pdf/2209.05433.pdf
constexpr double kMaxE5M2 = 57344;

// 2^15 * (1 + 3/4)
constexpr double kMaxE5M2FNUZ = 57344;

// 2^8 * (1 + 6/8)
// See https://arxiv.org/pdf/2209.05433.pdf
constexpr double kMaxE4M3FN = 448;

// 2^8 * (1 + 6/8)
constexpr double kMaxE4M3 = 448;

// 2^8 * (1 + 6/8)
constexpr double kMaxE4M3FNUZ = 448;

// 2^4 * 1.875
constexpr double kMaxE4M3B11FNUZ = 30;

// 2^4 * 1.9375
constexpr double kMaxE3M4 = 31;

// 2^(255 - 127)
constexpr double kMaxE8M0FNU = 3.4028236692093846e38;

// 2^2 * (1 + 7/8)
constexpr double kMaxE2M3FN = 7.5;

// 2^4 * (1 + 3/4)
constexpr double kMaxE3M2FN = 28.0;

// 2^2 * (1 + 1/2)
constexpr double kMaxE2M1FN = 6.0;

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_LIMITS_H_
