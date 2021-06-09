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
 * \file Use external mlas library call.
 */
#include <mlas.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.gemm_packb_size")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      /*
        function to get packed B matrix size
      */
      int N = args[0];
      int K = args[1];
      size_t packb_size = MlasGemmPackBSize(N, K);
      *ret = (int64_t)packb_size;
    });

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.gemm_packb").set_body([](TVMArgs args, TVMRetValue* ret) {
  /*
    function to pre-pack B matrix of matmul using mlas packb API
  */
  int N = args[0];
  int K = args[1];
  int ldb = args[2];
  bool transb = args[3];
  DLTensor* B = args[4];
  DLTensor* PackedB = args[5];
  if (transb) {
    MlasGemmPackB(CblasTrans, N, K, static_cast<float*>(B->data), ldb,
                  static_cast<void*>(PackedB->data));
  } else {
    MlasGemmPackB(CblasNoTrans, N, K, static_cast<float*>(B->data), ldb,
                  static_cast<void*>(PackedB->data));
  }
});

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.batch_sgemm").set_body([](TVMArgs args, TVMRetValue* ret) {
  /*
    batch_matmul using MLAS sgemm routine
  */
  int batch_size_A = args[0];
  int batch_size_B = args[1];
  int M = args[2];
  int N = args[3];
  int K = args[4];
  bool packb = args[5];

  const float alpha = 1.0;
  const float beta = 0.0;

  DLTensor* A = args[6];
  DLTensor* B = args[7];
  DLTensor* C = args[8];
  const float* A_ptr = static_cast<const float*>(A->data);
  const float* B_ptr = static_cast<const float*>(B->data);
  float* C_ptr = static_cast<float*>(C->data);

  if (packb != 0) {  // use mlas packed API, batch_size_B must be 1
    ICHECK_EQ(batch_size_B, 1) << "batch_size_B must be 1 when B is pre-packed";
    for (int batch_id = 0; batch_id < batch_size_A; batch_id++) {
      MlasGemm(CblasNoTrans, static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K),
               alpha, A_ptr + batch_id * M * K, static_cast<size_t>(K), B_ptr, beta,
               C_ptr + batch_id * M * N, static_cast<size_t>(N), nullptr);
    }

  } else {  // normal gemm
    ICHECK((batch_size_A == batch_size_B) || (batch_size_B == 1))
        << "Unsupported batch_size where batch_size_A = " << batch_size_A
        << ", batch_size_B = " << batch_size_B;
    if (batch_size_A == batch_size_B) {
      for (int batch_id = 0; batch_id < batch_size_A; batch_id++) {
        MlasGemm(CblasNoTrans, CblasTrans, static_cast<size_t>(M), static_cast<size_t>(N),
                 static_cast<size_t>(K), alpha, A_ptr + batch_id * M * K, static_cast<size_t>(K),
                 B_ptr + batch_id * K * N, static_cast<size_t>(K), beta, C_ptr + batch_id * M * N,
                 static_cast<size_t>(N), nullptr);
      }
    } else if (batch_size_B == 1) {
      for (int batch_id = 0; batch_id < batch_size_A; batch_id++) {
        MlasGemm(CblasNoTrans, CblasTrans, static_cast<size_t>(M), static_cast<size_t>(N),
                 static_cast<size_t>(K), alpha, A_ptr + batch_id * M * K, static_cast<size_t>(K),
                 B_ptr, static_cast<size_t>(K), beta, C_ptr + batch_id * M * N,
                 static_cast<size_t>(N), nullptr);
      }
    }
  }
});

}  // namespace contrib
}  // namespace tvm
