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

#include "mps_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mps.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  // call gemm for simple compact code.
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);
  ICHECK(C->strides == nullptr);
  ICHECK(B->strides == nullptr);
  ICHECK(A->strides == nullptr);
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32));
  ICHECK(TypeMatch(B->dtype, kDLFloat, 32));
  ICHECK(TypeMatch(C->dtype, kDLFloat, 32));
  // Get Metal device API
  MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
  // ICHECK_EQ(A->device, B->device);
  // ICHECK_EQ(A->device, C->device);
  id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(A->device);
  id<MTLCommandQueue> queue = entry_ptr->metal_api->GetCommandQueue(A->device);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  NSUInteger M = A->shape[0 + (transa ? 1 : 0)];
  NSUInteger N = B->shape[1 - (transb ? 1 : 0)];
  NSUInteger K = B->shape[0 + (transb ? 1 : 0)];

  ICHECK_EQ(A->shape[1 - (transa ? 1 : 0)], K);
  // mps a
  MPSDataType dtype = MPSType::DLTypeToMPSType(A->dtype);
  MPSMatrixDescriptor* descA =
      [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                  columns:K
                                                 rowBytes:K * sizeof(MPSDataTypeFloat32)
                                                 dataType:MPSDataTypeFloat32];
  id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)(A->data);
  MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
  // mps b
  MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:K
                                                                           columns:N
                                                                          rowBytes:N * sizeof(dtype)
                                                                          dataType:dtype];
  id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(B->data);
  MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
  // mps c
  MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(dtype)
                                                                          dataType:dtype];
  id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)(C->data);
  MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];
  // kernel

  MPSMatrixMultiplication* mul_obj = [[MPSMatrixMultiplication alloc] init];
  MPSMatrixMultiplication* sgemm = [mul_obj initWithDevice:dev
                                             transposeLeft:transa
                                            transposeRight:transb
                                                resultRows:M
                                             resultColumns:N
                                           interiorColumns:K
                                                     alpha:1.0f
                                                      beta:0.0f];
  ICHECK(sgemm != nil);
  [sgemm encodeToCommandBuffer:cb leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
  [cb commit];
});

}  // namespace contrib
}  // namespace tvm
