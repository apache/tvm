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

#ifdef __cplusplus
extern "C"
{
#include <mkl_cblas.h>
#endif  // extern "C"

// TODO(@zhiics) Generate the signature that is consistent to cblas_sgemm
// directly. We can process the other parameters from attribute of a Relay call
// node.
void dense(float* A, float* B, float* C, int M, int N, int K) {
 cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, 1, B, 1, 0.0, C, 1);
}

#ifdef __cplusplus
}
#endif
