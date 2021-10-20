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

TVM_REGISTER_GLOBAL("tvm.contrib.mps.buffer2img").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* buf = args[0];
  DLTensor* img = args[1];
  // copy to temp
  id<MTLBuffer> mtlbuf = (__bridge id<MTLBuffer>)(buf->data);
  MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
  runtime::metal::MetalThreadEntry* rt = runtime::metal::MetalThreadEntry::ThreadLocal();
  id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(buf->device);
  id<MTLBuffer> temp = rt->GetTempBuffer(buf->device, [mtlbuf length]);
  entry_ptr->metal_api->CopyDataFromTo((__bridge void*)mtlbuf, 0, (__bridge void*)temp, 0,
                                       [mtlbuf length], buf -> device, buf -> device, buf -> dtype,
                                       nullptr);

  MPSImageDescriptor* desc =
      [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                     width:buf->shape[2]
                                                    height:buf->shape[1]
                                           featureChannels:buf->shape[3]];

  MPSImage* mpsimg = entry_ptr->AllocMPSImage(dev, desc);

  [mpsimg writeBytes:[temp contents]
          dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
          imageIndex:0];

  img->data = (__bridge void*)mpsimg;

  [mpsimg readBytes:[temp contents]
         dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
         imageIndex:0];
});

TVM_REGISTER_GLOBAL("tvm.contrib.mps.img2buffer").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* img = args[0];
  DLTensor* buf = args[1];
  id<MTLBuffer> mtlbuf = (__bridge id<MTLBuffer>)(buf->data);
  MPSImage* mpsimg = (__bridge MPSImage*)(img->data);
  MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
  runtime::metal::MetalThreadEntry* rt = runtime::metal::MetalThreadEntry::ThreadLocal();
  id<MTLBuffer> temp = rt->GetTempBuffer(buf->device, [mtlbuf length]);

  [mpsimg readBytes:[temp contents]
         dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
         imageIndex:0];

  entry_ptr->metal_api->CopyDataFromTo((__bridge void*)temp, 0, (__bridge void*)mtlbuf, 0,
                                       [mtlbuf length], buf -> device, buf -> device, buf -> dtype,
                                       nullptr);
});

TVM_REGISTER_GLOBAL("tvm.contrib.mps.conv2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  // MPS-NHWC
  DLTensor* data = args[0];
  DLTensor* weight = args[1];
  DLTensor* output = args[2];
  int pad = args[3];
  int stride = args[4];

  ICHECK_EQ(data->ndim, 4);
  ICHECK_EQ(weight->ndim, 4);
  ICHECK_EQ(output->ndim, 4);
  ICHECK(output->strides == nullptr);
  ICHECK(weight->strides == nullptr);
  ICHECK(data->strides == nullptr);

  ICHECK_EQ(data->shape[0], 1);
  ICHECK_EQ(output->shape[0], 1);

  int oCh = weight->shape[0];
  int kH = weight->shape[1];
  int kW = weight->shape[2];
  int iCh = weight->shape[3];

  auto f_buf2img = runtime::Registry::Get("tvm.contrib.mps.buffer2img");
  auto f_img2buf = runtime::Registry::Get("tvm.contrib.mps.img2buffer");
  // Get Metal device API
  MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
  runtime::metal::MetalThreadEntry* rt = runtime::metal::MetalThreadEntry::ThreadLocal();
  id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(data->device);
  id<MTLCommandQueue> queue = entry_ptr->metal_api->GetCommandQueue(data->device);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  // data to MPSImage
  DLTensor tmp_in;
  (*f_buf2img)(data, &tmp_in);
  MPSImage* tempA = (__bridge MPSImage*)tmp_in.data;
  // weight to temp memory
  id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(weight->data);
  id<MTLBuffer> tempB = rt->GetTempBuffer(weight->device, [bufB length]);
  entry_ptr->metal_api->CopyDataFromTo((__bridge void*)bufB, 0, (__bridge void*)tempB, 0,
                                       [bufB length], weight -> device, weight -> device,
                                       tmp_in.dtype, nullptr);
  float* ptr_w = (float*)[tempB contents];
  // output to MPSImage
  DLTensor tmp_out;
  (*f_buf2img)(output, &tmp_out);
  MPSImage* tempC = (__bridge MPSImage*)tmp_out.data;
  // conv desc

  MPSCNNConvolutionDescriptor* conv_desc =
      [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kW
                                                              kernelHeight:kH
                                                      inputFeatureChannels:iCh
                                                     outputFeatureChannels:oCh];
  [conv_desc setStrideInPixelsX:stride];
  [conv_desc setStrideInPixelsY:stride];

  MPSCNNConvolution* conv = [[MPSCNNConvolution alloc] initWithDevice:dev
                                                convolutionDescriptor:conv_desc
                                                        kernelWeights:ptr_w
                                                            biasTerms:nil
                                                                flags:MPSCNNConvolutionFlagsNone];
  if (pad == 0) {
    conv.padding = [MPSNNDefaultPadding paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                                          MPSNNPaddingMethodAlignCentered |
                                                          MPSNNPaddingMethodSizeSame];
  } else if (pad == 1) {
    conv.padding = [MPSNNDefaultPadding paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                                          MPSNNPaddingMethodAlignCentered |
                                                          MPSNNPaddingMethodSizeValidOnly];
  }
  [conv encodeToCommandBuffer:cb sourceImage:tempA destinationImage:tempC];

  [cb commit];
  id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
  [encoder synchronizeResource:tempC.texture];
  [encoder endEncoding];
  [cb waitUntilCompleted];

  (*f_img2buf)(&tmp_out, output);
});

}  // namespace contrib
}  // namespace tvm
