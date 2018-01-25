#include "mps_utils.h"
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mps.conv2d")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      // MPS-NHWC
      DLTensor *data = args[0];
      DLTensor *weight = args[1];
      DLTensor *output = args[2];
      // int stride = args[3]; TODO: stride != 1

      CHECK_EQ(data->ndim, 4);
      CHECK_EQ(weight->ndim, 4);
      CHECK_EQ(output->ndim, 4);
      CHECK(output->strides == nullptr);
      CHECK(weight->strides == nullptr);
      CHECK(data->strides == nullptr);

      int oCh = weight->shape[0];
      int iCh = weight->shape[1];
      int kH = weight->shape[2];
      int kW = weight->shape[3];
      // Get Metal device API
      MetalThreadEntry *entry_ptr = MetalThreadEntry::ThreadLocal();

      id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(data->ctx);
      id<MTLCommandQueue> queue =
          entry_ptr->metal_api->GetCommandQueue(data->ctx);
      id<MTLCommandBuffer> cb = [queue commandBuffer];


      MPSDataType dtype = MPSType::DLTypeToMPSType(data->dtype);
      id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)(data->data);
      id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(weight->data);
      id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)(output->data);
      

      // copy gpu mem to cpu
      id<MTLBuffer> tempA = [dev newBufferWithLength:[bufA length]
            options:MTLStorageModeShared];
      id<MTLBuffer> tempB = [dev newBufferWithLength:[bufB length]
            options:MTLStorageModeShared];
      id<MTLBuffer> tempC = [dev newBufferWithLength:[bufC length]
            options:MTLStorageModeShared];

      id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
      [encoder copyFromBuffer:bufA
               sourceOffset:0
               toBuffer:tempA
               destinationOffset:0
               size:[bufA length]];
      
       [encoder copyFromBuffer:bufB
               sourceOffset:0
               toBuffer:tempB
               destinationOffset:0
               size:[bufB length]];

      [encoder endEncoding];
      [cb commit];
      [cb waitUntilCompleted];

      MPSImageDescriptor *descA = [MPSImageDescriptor
          imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                     width:data->shape[2]
                                    height:data->shape[1]
                           featureChannels:data->shape[3]];

      // MPSTemporaryImage * imgA = [MPSTemporaryImage temporaryImageWithCommandBuffer:cb imageDescriptor:descA];
      MPSImage *imgA =
           [[MPSImage alloc] initWithDevice:dev imageDescriptor:descA];

      [imgA readBytes:[tempA contents]
           dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
           imageIndex:0];
    

      const float *ptr_w = (float *)[tempB contents];

      // mps output
      
      MPSImageDescriptor *descC = [MPSImageDescriptor
          imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                     width:output->shape[2]
                                    height:output->shape[1]
                           featureChannels:output->shape[3]];
      // MPSTemporaryImage * imgC = [MPSTemporaryImage temporaryImageWithCommandBuffer:cb imageDescriptor:descC];
      MPSImage *imgC =
          [[MPSImage alloc] initWithDevice:dev imageDescriptor:descC];
      // kernel

      // Set up the convolution operator with description
      MPSCNNConvolutionDescriptor *conv_desc =
          [MPSCNNConvolutionDescriptor new];
      [conv_desc setKernelWidth:kW];
      [conv_desc setKernelHeight:kH];
      [conv_desc setInputFeatureChannels:iCh];
      [conv_desc setOutputFeatureChannels:oCh];
      // [conv_desc setStrideInPixelsX:stride];
      // [conv_desc setStrideInPixelsY:stride];

      MPSCNNConvolution *conv =
          [[MPSCNNConvolution alloc] initWithDevice:dev
                              convolutionDescriptor:conv_desc
                                      kernelWeights:ptr_w
                                          biasTerms:nil
                                              flags:MPSCNNConvolutionFlagsNone];

      
      [conv encodeToCommandBuffer:cb
                      sourceImage:imgA
                 destinationImage:imgC];
      
      #if TARGET_OS_OSX
      id<MTLBlitCommandEncoder> bilt = cb.blitCommandEncoder;
      [bilt synchronizeResource:imgC.texture];
      [bilt endEncoding];
      #endif
      
      [cb commit];
      [cb waitUntilCompleted];
      
      [imgC writeBytes:[tempC contents]
            dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
            imageIndex:0];

      encoder = [cb blitCommandEncoder];
 
      [encoder copyFromBuffer:tempC
               sourceOffset:0
               toBuffer:bufC
               destinationOffset:0
               size:[bufC length]];
      [encoder endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      
      [tempA release];
      [tempB release];
      [tempC release];
    });

} // namespace contrib
} // namespace tvm
