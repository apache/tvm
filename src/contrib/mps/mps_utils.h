/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external mps utils function
 */

#ifndef TVM_CONTRIB_MPS_MPS_UTILS_H_
#define TVM_CONTRIB_MPS_MPS_UTILS_H_

#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
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
  runtime::MetalWorkspace *metal_api{nullptr};
  static MetalThreadEntry* ThreadLocal();
};  // MetalThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MPS_MPS_UTILS_H_
