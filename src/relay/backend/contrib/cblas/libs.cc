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

#include <mkl_cblas.h>
#include <stdio.h>

#define DENSE_FP32(p_ID_, p_M_, p_N_, p_K_)                                                       \
  extern "C" void p_ID_(float* A, float* B, float* C) {                                           \
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p_M_, p_N_, p_K_, 1.0, A, p_K_, B, p_N_, \
                0.0, C, p_N_);                                                                    \
  }

#define DENSE_FP64(p_ID_, p_M_, p_N_, p_K_)                                                       \
  extern "C" void p_ID_(double* A, double* B, double* C) {                                        \
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, p_M_, p_N_, p_K_, 1.0, A, p_K_, B, p_N_, \
                0.0, C, p_N_);                                                                    \
  }
