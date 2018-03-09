/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external mps utils function
 */

#ifndef TVM_CONTRIB_MPS_MPS_UTILS_H_
#define TVM_CONTRIB_MPS_MPS_UTILS_H_

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <vector>
#include "../../runtime/metal/metal_common.h"

namespace tvm {
namespace contrib {

/*! breif Convert DLTensor type to MPS type */
struct MPSType {
  static MPSDataType DLTypeToMPSType(const DLDataType &dtype);
};  // struct MPSType

struct MetalThreadEntry {
  MetalThreadEntry();
  ~MetalThreadEntry();
  MPSImage *AllocMPSImage(id<MTLDevice> dev, MPSImageDescriptor *desc);
  MPSTemporaryImage *AllocTempImage(id<MTLCommandBuffer> cb,
                                    MPSImageDescriptor *desc);
  runtime::metal::MetalWorkspace *metal_api{nullptr};
  static MetalThreadEntry *ThreadLocal();
  std::vector<MPSImage *> img_table;
};  // MetalThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MPS_MPS_UTILS_H_
