/*!
 *  Copyright (c) 2017 by Contributors
 * \author Bing Xu
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

struct ConvThreadEntry {
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionMode_t mode;
  cudnnFilterDescriptor_t filter_desc;
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnMathType_t math_type;  // default of tensor op
  int group_count {0};
  ConvThreadEntry() {
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  }
  ~ConvThreadEntry() {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  }
  static ConvThreadEntry* ThreadLocal() {
    static thread_local ConvThreadEntry inst;
    return &inst;
  }
};  // ConvThreadEntry

inline void _SetNdConvDesc(int dim,
                           int mode,
                           DLTensor *pad,
                           DLTensor *filter_stride,
                           DLTensor *dilation) {
  ConvThreadEntry* entry_ptr = ConvThreadEntry::ThreadLocal();
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_desc,
                                             dim - 2,
                                             static_cast<int*>(pad->data),
                                             static_cast<int*>(filter_stride->data),
                                             static_cast<int*>(dilation->data),
                                             entry_ptr->mode,
                                             entry_ptr->data_type));
}

inline void _SetNdFilterDesc(int dim,
                             DLTensor *filter_dim) {
  ConvThreadEntry* entry_ptr = ConvThreadEntry::ThreadLocal();
  CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->filter_desc,
                                        entry_ptr->data_type,
                                        CUDNN_TENSOR_NCHW,
                                        dim,
                                        static_cast<int*>(filter_dim->data)));
}

inline void _SetNdInputDesc(int dim,
                            DLTensor *input_dim) {
  ConvThreadEntry* entry_ptr = ConvThreadEntry::ThreadLocal();
  StackVector<kStride0, int> stride;
  GetStride(dim,
            static_cast<int*>(input_dim->data),
            stride.value);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->input_desc,
                                        entry_ptr->data_type,
                                        dim,
                                        static_cast<int*>(input_dim->data),
                                        stride.value));
}

inline void ConvGetNdOutputShape(int dim,
                                 int mode,
                                 int format,
                                 DLTensor *input_dim,
                                 DLTensor *filter_dim,
                                 DLTensor *pad,
                                 DLTensor *filter_stride,
                                 DLTensor *dilation,
                                 DLTensor *out_shape) {
  ConvThreadEntry* entry_ptr = ConvThreadEntry::ThreadLocal();
  // mode
  switch (mode) {
    case 0: {
      entry_ptr->mode = CUDNN_CONVOLUTION;
      break;
    }
    case 1: {
      entry_ptr->mode = CUDNN_CROSS_CORRELATION;
      break;
    }
    default: {
      LOG(FATAL) << "Unknow conv mode";
    }
  }
  // format
  switch (format) {
    case 0: {
      entry_ptr->tensor_format = CUDNN_TENSOR_NCHW;
      break;
    }
    case 1: {
      entry_ptr->tensor_format = CUDNN_TENSOR_NHWC;
      break;
    }
    case 2: {
      entry_ptr->tensor_format = CUDNN_TENSOR_NCHW_VECT_C;
      break;
    }
    default: {
      LOG(FATAL) << "Unknow tensor format";
    }
  }
  entry_ptr->data_type = CUDNN_DATA_FLOAT;
  _SetNdConvDesc(dim,
                 mode,
                 pad,
                 filter_stride,
                 dilation);
  _SetNdFilterDesc(dim,
                   filter_dim);

  _SetNdInputDesc(dim,
                  input_dim);

  CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(entry_ptr->conv_desc,
                                                   entry_ptr->input_desc,
                                                   entry_ptr->filter_desc,
                                                   dim,
                                                   static_cast<int*>(out_shape->data)));
}


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.output_shape")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int dim = args[0];
  int mode = args[1];
  int format = args[2];
  DLTensor *input_dim = args[3];
  DLTensor *filter_dim = args[4];
  DLTensor *pad = args[5];
  DLTensor *filter_stride = args[6];
  DLTensor *dilation = args[7];
  DLTensor *out_shape = args[8];
  // data type check
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(input_dim->dtype), CUDNN_DATA_INT32);
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(filter_dim->dtype), CUDNN_DATA_INT32);
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(pad->dtype), CUDNN_DATA_INT32);
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(filter_stride->dtype), CUDNN_DATA_INT32);
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(dilation->dtype), CUDNN_DATA_INT32);
  CHECK_EQ(CuDNNDataType::DLTypeToCuDNNType(out_shape->dtype), CUDNN_DATA_INT32);
  // dim check
  CHECK_EQ(input_dim->ndim, 1);
  CHECK_EQ(filter_dim->ndim, 1);
  CHECK_EQ(pad->ndim, 1);
  CHECK_EQ(filter_stride->ndim, 1);
  CHECK_EQ(dilation->ndim, 1);
  CHECK_EQ(out_shape->ndim, 1);
  // space check
  CHECK_EQ(input_dim->shape[0], dim + 2);
  CHECK_EQ(filter_dim->shape[0], dim + 2);
  CHECK_EQ(pad->shape[0], dim);
  CHECK_EQ(filter_stride->shape[0], dim);
  CHECK_EQ(dilation->shape[0], dim);
  CHECK_EQ(out_shape->shape[0], dim + 2);

  ConvGetNdOutputShape(input_dim->shape[0],
                       mode,
                       format,
                       input_dim,
                       filter_dim,
                       pad,
                       filter_stride,
                       dilation,
                       out_shape);
  });

}  // namespace contrib
}  // namespace tvm