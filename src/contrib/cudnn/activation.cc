/*!
 *  Copyright (c) 2017 by Contributors
 * \author Bing Xu
 * \file Use external cudnn activation library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

namespace cudnn_activation {
  enum ActivationType {kSigmoid,
                       kReLU,
                       kTanh,
                       kClippedReLU,
                       kELU};
}  // namespace cudnn_activation

struct CuDNNActivationDesc {
  cudnnDataType_t dtype;
  cudnnActivationMode_t mode;
  cudnnTensorDescriptor_t shape_desc;
  cudnnActivationDescriptor_t desc;
  cudnnNanPropagation_t nan_prop {CUDNN_NOT_PROPAGATE_NAN};
  CuDNNActivationDesc() {
    CUDNN_CALL(cudnnCreateActivationDescriptor(&desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc));
  }
  ~CuDNNActivationDesc() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(shape_desc));
  }
  static CuDNNActivationDesc* ThreadLocal() {
    static thread_local CuDNNActivationDesc inst;
    return &inst;
  }
};  // struct CuDNNActivationHandle

inline cudnnActivationMode_t ActTypeToCuDNNMode(int mode) {
  switch (mode) {
    case cudnn_activation::kSigmoid:
      return CUDNN_ACTIVATION_SIGMOID;
    case cudnn_activation::kReLU:
      return CUDNN_ACTIVATION_RELU;
    case cudnn_activation::kTanh:
      return CUDNN_ACTIVATION_TANH;
    case cudnn_activation::kClippedReLU:
      return CUDNN_ACTIVATION_CLIPPED_RELU;
    case cudnn_activation::kELU:
      return CUDNN_ACTIVATION_ELU;
    default:
      LOG(FATAL) << "Unsupported type";
      break;
  }
  return CUDNN_ACTIVATION_RELU;
}

template<int mode>
inline void CuDNNActivationForward(DLTensor *in_tensor,
                                   DLTensor *out_tensor,
                                   double coef) {
  auto p_handle = CuDNNHandle::ThreadLocal();
  auto p_desc = CuDNNActivationDesc::ThreadLocal();
  auto p_shape_vec = StackVector<kShape0, int>::ThreadLocal();
  auto p_stride_vec = StackVector<kStride0, int>::ThreadLocal();
  p_shape_vec->Set(in_tensor->shape, in_tensor->ndim);
  GetStride(p_shape_vec->size, p_shape_vec->Get(), p_stride_vec->Get());

  p_desc->mode = ActTypeToCuDNNMode(mode);
  CHECK_EQ(in_tensor->ndim, out_tensor->ndim);
  // CHECK_EQ(in_tensor->dtype, out_tensor->dtype);
  p_desc->dtype = CuDNNDataType::DLTypeToCuDNNType(in_tensor->dtype);
  CUDNN_CALL(cudnnSetActivationDescriptor(p_desc->desc,
                                          p_desc->mode, 
                                          p_desc->nan_prop,
                                          coef));
  
  CUDNN_CALL(cudnnSetTensorNdDescriptor(p_desc->shape_desc,
                                        p_desc->dtype,
                                        in_tensor->ndim,
                                        p_shape_vec->Get(),
                                        p_stride_vec->Get()));

  CUDNN_CALL(cudnnActivationForward(p_handle->handle,
                                    p_desc->desc,
                                    CuDNNDataType::GetConst<1>(p_desc->dtype),
                                    p_desc->shape_desc,
                                    in_tensor->data,
                                    CuDNNDataType::GetConst<0>(p_desc->dtype),
                                    p_desc->shape_desc,
                                    out_tensor->data));
}

// ReLU
TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.relu")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* in_tensor = args[0];
  DLTensor* out_tensor = args[1];
  CuDNNActivationForward<cudnn_activation::kReLU>(in_tensor,
                                                  out_tensor,
                                                  0.f);
  });

}  // namespace contrib
}  // namespace tvm
