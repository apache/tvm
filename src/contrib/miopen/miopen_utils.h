/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external miopen utils function
 */

#ifndef TVM_CONTRIB_MIOPEN_MIOPEN_UTILS_H_
#define TVM_CONTRIB_MIOPEN_MIOPEN_UTILS_H_

#include <dmlc/logging.h>
#include <miopen/miopen.h>
#include <tvm/runtime/device_api.h>
#include <string>
#include "../../runtime/rocm/rocm_common.h"

namespace tvm {
namespace contrib {
namespace miopen {

std::string miopenGetErrorString(int error_code);

#define MIOPEN_CALL(func)                                                      \
  {                                                                            \
    miopenStatus_t e = (func);                                                 \
    CHECK_EQ(e, miopenStatusSuccess)                                           \
        << "miopen error: " << miopenGetErrorString(e);                        \
  }

struct ConvEntry {
  miopenConvolutionDescriptor_t conv_desc;
  miopenConvolutionMode_t mode{miopenConvolution};
  miopenTensorDescriptor_t filter_desc;
  miopenDataType_t data_type{miopenFloat};
  miopenTensorDescriptor_t input_desc;
  miopenTensorDescriptor_t output_desc;
  miopenConvFwdAlgorithm_t fwd_algo;
  TVMContext ctx;
  runtime::DeviceAPI *rocm_api;
  void *workspace{nullptr};
  size_t workspace_size{0};
  ConvEntry();
  ~ConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // ConvThreadEntry

struct MIOpenThreadEntry {
  MIOpenThreadEntry();
  ~MIOpenThreadEntry();
  miopenHandle_t handle{nullptr};
  ConvEntry conv_entry;
  runtime::DeviceAPI *rocm_api{nullptr};
  static MIOpenThreadEntry *ThreadLocal();
};  // MIOpenThreadEntry

}  // namespace miopen
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MIOPEN_MIOPEN_UTILS_H_
