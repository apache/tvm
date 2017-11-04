#include "../../runtime/metal/metal_common.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mps.matmul")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      DLTensor *A = args[0];
      DLTensor *B = args[1];
      DLTensor *C = args[2];
      bool transa = args[3];
      bool transb = args[4];
      // call gemm for simple compact code.
      CHECK_EQ(A->ndim, 2);
      CHECK_EQ(B->ndim, 2);
      CHECK_EQ(C->ndim, 2);
      CHECK(C->strides == nullptr);
      CHECK(B->strides == nullptr);
      CHECK(A->strides == nullptr);
      CHECK(TypeMatch(A->dtype, kDLFloat, 32));
      CHECK(TypeMatch(B->dtype, kDLFloat, 32));
      CHECK(TypeMatch(C->dtype, kDLFloat, 32));
      // Get Metal device API
      auto func = runtime::Registry::Get("device_api.metal");
      void *dev_handle = (*func)();
      runtime::metal::MetalWorkspace *metal_api =
          static_cast<runtime::metal::MetalWorkspace *>(dev_handle);
      // TODO(Check same device)
      id<MTLDevice> dev = metal_api->GetDevice(A->ctx);
      id<MTLCommandQueue> queue = metal_api->GetCommandQueue(A->ctx);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      // TODO(Check shape)
      NSUInteger M = A->shape[0];
      NSUInteger N = B->shape[1];
      NSUInteger K = B->shape[0];
      // mps a
      MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:M
                                 columns:K
                                rowBytes:M * sizeof(float)
                                dataType:MPSDataTypeFloat32];
      id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)(A->data);
      MPSMatrix *matrixA =
          [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
      // mps b
      MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:K
                                 columns:N
                                rowBytes:K * sizeof(float)
                                dataType:MPSDataTypeFloat32];
      id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(B->data);
      MPSMatrix *matrixB =
          [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
      // mps c
      MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:M
                                 columns:N
                                rowBytes:M * sizeof(float)
                                dataType:MPSDataTypeFloat32];
      id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)(C->data);
      MPSMatrix *matrixC =
          [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];
      // kernel

      MPSMatrixMultiplication *mul_obj = [[MPSMatrixMultiplication alloc] init];
      MPSMatrixMultiplication *sgemm = [mul_obj initWithDevice:dev
                                                 transposeLeft:transa
                                                transposeRight:transb
                                                    resultRows:M
                                                 resultColumns:N
                                               interiorColumns:K
                                                         alpha:1.0f
                                                          beta:0.0f];
      CHECK(sgemm != nil);
      [sgemm encodeToCommandBuffer:cb
                        leftMatrix:matrixA
                       rightMatrix:matrixB
                      resultMatrix:matrixC];
      [cb commit];
      [cb waitUntilCompleted];
      [mul_obj dealloc];
      [matrixA dealloc];
      [matrixB dealloc];
      [matrixC dealloc];
    });

} // namespace contrib
} // namespace tvm
