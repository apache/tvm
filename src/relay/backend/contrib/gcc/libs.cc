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
#include <cstring>
#include <iostream>

#define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      out[i] = a[i] p_OP_ b[i];                           \
    }                                                     \
  }

#define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      for (int64_t j = 0; j < p_DIM2_; ++j) {             \
        int64_t k = i * p_DIM2_ + j;                      \
        out[k] = a[k] p_OP_ b[k];                         \
      }                                                   \
    }                                                     \
  }
