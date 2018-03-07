/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external mps utils function
 */
#include "mps_utils.h"

namespace tvm {
namespace contrib {

// MPS Data Type
MPSDataType MPSType::DLTypeToMPSType(const DLDataType &dtype) {
  switch (dtype.code) {
  case kDLInt:
    if (dtype.bits == 8 && dtype.lanes == 1)
      return MPSDataTypeInt8;
    else if (dtype.bits == 16 && dtype.lanes == 1)
      return MPSDataTypeInt16;
    else
      LOG(FATAL) << "Unsupported type";
    break;
  case kDLUInt:
    if (dtype.bits == 8 && dtype.lanes == 1)
      return MPSDataTypeUInt8;
    else if (dtype.bits == 16 && dtype.lanes == 1)
      return MPSDataTypeUInt16;
    else if (dtype.bits == 32 && dtype.lanes == 1)
      return MPSDataTypeUInt32;
    LOG(FATAL) << "Unsupported type";
    break;
  case kDLFloat:
    if (dtype.bits == 16 && dtype.lanes == 1)
      return MPSDataTypeFloat16;
    else if (dtype.bits == 32 && dtype.lanes == 1)
      return MPSDataTypeFloat32;
    else
      LOG(FATAL) << "Unsupported type";
    break;
  default:
    LOG(FATAL) << "Unsupported type";
  }
  return MPSDataTypeFloat32;
}

// MetalThreadEntry

MPSImage *MetalThreadEntry::AllocMPSImage(id<MTLDevice> dev,
                                          MPSImageDescriptor *desc) {
  MPSImage *mpsimg = [[MPSImage alloc] initWithDevice:dev imageDescriptor:desc];
  img_table.push_back(mpsimg);
  return mpsimg;
}

MPSTemporaryImage *MetalThreadEntry::AllocTempImage(id<MTLCommandBuffer> cb,
                                                    MPSImageDescriptor *desc) {
  MPSTemporaryImage *mpsimg =
      [MPSTemporaryImage temporaryImageWithCommandBuffer:cb
                                         imageDescriptor:desc];
  return mpsimg;
}

MetalThreadEntry::MetalThreadEntry() {
  auto func = runtime::Registry::Get("device_api.metal");
  void *ret = (*func)();
  metal_api = static_cast<runtime::metal::MetalWorkspace *>(ret);
}

MetalThreadEntry::~MetalThreadEntry() {
  for (int i = 0; i < img_table.size(); ++i) {
    [img_table[i] dealloc];
  }
}

typedef dmlc::ThreadLocalStore<MetalThreadEntry> MetalThreadStore;

MetalThreadEntry *MetalThreadEntry::ThreadLocal() {
  return MetalThreadStore::Get();
}

} // namespace contrib
} // namespace tvm
