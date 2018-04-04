/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.relu")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  cudnnTensorDescriptor_t tensor_desc;
  cudnnActivationDescriptor_t act_desc;
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  cudnnDataType_t dtype = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

  // Create Desc
  CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
  CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
  // Set Desc
  CUDNN_CALL(cudnnSetTensor4dDescriptor(tensor_desc,
                                        CUDNN_TENSOR_NCHW,
                                        dtype,
                                        static_cast<int>(x->shape[0]),
                                        static_cast<int>(x->shape[1]),
                                        static_cast<int>(x->shape[2]),
                                        static_cast<int>(x->shape[3])));
  CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                          CUDNN_ACTIVATION_RELU,
                                          CUDNN_NOT_PROPAGATE_NAN,
                                          0.0));
  // ReLU
  CUDNN_CALL(cudnnActivationForward(entry_ptr->handle,
                                    act_desc,
                                    CuDNNDataType::GetConst<1>(dtype),
                                    tensor_desc,
                                    x->data,
                                    CuDNNDataType::GetConst<0>(dtype),
                                    tensor_desc,
                                    y->data));
  // Destroy Desc
  CUDNN_CALL(cudnnDestroyActivationDescriptor(act_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
});


}  // namespace contrib
}  // namespace tvm

