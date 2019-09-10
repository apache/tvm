/* * Licensed to the Apache Software Foundation (ASF) under one
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

#include "libs.h"

#include <cstdint>
#include <iostream>

#define GCC_BINARY_OP(OP, SYMB)                                            \
  extern "C" void OP(ExternalTensor a, ExternalTensor b,                   \
                     ExternalTensor* out) {                                \
    if (a.ndim > 2 || a.ndim != b.ndim || a.ndim != out->ndim) {           \
      std::cerr << "Array sizes are not consistent, a.ndim = " << a.ndim   \
                << ", b.ndim = " << b.ndim << ", out ndim = " << out->ndim \
                << std::endl;                                              \
    }                                                                      \
    for (int i = 0; i < a.ndim; i++) {                                     \
      if (a.shape[i] != b.shape[i]) {                                      \
        std::cerr << "shape[" << i << "]: a = " << a.shape[i]              \
                  << ", b = " << b.shape[i] << std::endl;                  \
      }                                                                    \
    }                                                                      \
    std::cout << "dim: " << a.ndim << " shape: " << std::endl;             \
    for (int i = 0; i < a.ndim; i++) {                                     \
      std::cout << a.shape[i] << " " << b.shape[i] << std::endl;           \
    }                                                                      \
    float* a_ptr = static_cast<float*>(a.data);                            \
    float* b_ptr = static_cast<float*>(b.data);                            \
    float* out_ptr = static_cast<float*>(out->data);                       \
    if (a.ndim == 1) {                                                     \
      for (int64_t i = 0; i < a.shape[0]; i++) {                           \
        out_ptr[i] = a_ptr[i] SYMB b_ptr[i];                               \
      }                                                                    \
    } else {                                                               \
      for (int64_t i = 0; i < a.shape[0]; i++) {                           \
        for (int64_t j = 0; j < a.shape[1]; j++) {                         \
          int64_t k = i * a.shape[1] + j;                                  \
          out_ptr[k] = a_ptr[k] SYMB b_ptr[k];                             \
        }                                                                  \
      }                                                                    \
    }                                                                      \
  }

GCC_BINARY_OP(subtract, -);
GCC_BINARY_OP(add, +);
GCC_BINARY_OP(multiply, *);

// extern "C" void Subtract(ExternalTensor a, ExternalTensor b, ExternalTensor* out) {
//   if (a.ndim > 2 || a.ndim != b.ndim || a.ndim  != out->ndim) {
//     std::cerr << "Array sizes are not consistent, a.ndim = " << a.ndim
//               << ", b.ndim = " << b.ndim
//               << ", out ndim = " << out->ndim << std::endl;
//   }
//   for (int i = 0; i < a.ndim; i++) {
//     if (a.shape[i] != b.shape[i]) {
//       std::cerr << "shape[" << i << "]: a = " << a.shape[i] << ", b = " << b.shape[i] << std::endl;
//     }
//   }
//   std::cout << "dim: " << a.ndim << " shape: " << std::endl;
//   for (int i = 0; i < a.ndim; i++) {
//     std::cout << a.shape[i] << " " << b.shape[i] << std::endl;
//   }
//   float* a_ptr = static_cast<float*>(a.data);
//   float* b_ptr = static_cast<float*>(b.data);
//   float* out_ptr = static_cast<float*>(out->data);
//   if (a.ndim == 1) {
//     for (int64_t i = 0; i < a.shape[0]; i++) {
//       out_ptr[i] = a_ptr[i] - b_ptr[i];
//     }
//   } else {
//     for (int64_t i = 0; i < a.shape[0]; i++) {
//       for (int64_t j = 0; j < a.shape[1]; j++) {
//         int64_t k = i * a.shape[1] + j;
//         out_ptr[k] = a_ptr[k] - b_ptr[k];
//       }
//     }
//   }
// }
