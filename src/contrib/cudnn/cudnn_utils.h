/*!
 *  Copyright (c) 2017 by Contributors
 * \author Bing Xu
 * \file Use external cudnn utils function
 */

#ifndef TVM_CONTRIB_CUDNN_CUDNN_UTILS_H_
#define TVM_CONTRIB_CUDNN_CUDNN_UTILS_H_

#include <dmlc/logging.h>
#include <cudnn.h>
#include "../../runtime/cuda/cuda_common.h"

namespace tvm {
namespace contrib {

const int MAX_STACK_VECTOR_SIZE = 8;
enum StackVectorType {kShape0, kStride0,
                      kShape1, kStride1,
                      kShape2, kStride2,
                      kShape3, kStride3};

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

/*! breif Convert DLTensor type to CuDNN type */
struct CuDNNDataType {
  static cudnnDataType_t DLTypeToCuDNNType(const DLDataType &dtype) {
    switch (dtype.code) {
      case kInt:
        if (dtype.bits == 8 && dtype.lanes == 1) return CUDNN_DATA_INT8;
        else if (dtype.bits == 32 && dtype.lanes == 1) return CUDNN_DATA_INT32;
        else if (dtype.bits == 8 && dtype.lanes == 4) return CUDNN_DATA_INT8x4;
        else
          LOG(FATAL) << "Unsupported type";
        break;
      case kUInt:
        LOG(FATAL) << "Unsupported type";
        break;
      case kFloat:
        if (dtype.bits == 32 && dtype.lanes == 1) return CUDNN_DATA_FLOAT;
        else if (dtype.bits == 64 && dtype.lanes == 1) return CUDNN_DATA_DOUBLE;
        else if (dtype.bits == 16 && dtype.lanes == 1) return CUDNN_DATA_HALF;
        else
          LOG(FATAL) << "Unsupported type";
        break;
    }
    return CUDNN_DATA_FLOAT;
  }

  template<int v>
  static const void* GetConst(cudnnDataType_t type);
};  // struct CuDNNDataType

template<>
const void* CuDNNDataType::GetConst<0>(cudnnDataType_t type) {
  static const int int_v = 0;
  static const float float_v = 0;
  static const double double_v = 0;
  if (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == CUDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == CUDNN_DATA_INT8 || type == CUDNN_DATA_INT32 || type == CUDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}

template<>
const void* CuDNNDataType::GetConst<1>(cudnnDataType_t type) {
  static const int int_v = 1;
  static const float float_v = 1.f;
  static const double double_v = 1.f;
  if (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == CUDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == CUDNN_DATA_INT8 || type == CUDNN_DATA_INT32 || type == CUDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}


inline void GetStride(int nbdim, const int *dims, int *strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >=0; --i) {
    mul *= dims[i];
    strides[i] = mul;
  }
}

template<int type, typename T>
struct StackVector {
  T value[MAX_STACK_VECTOR_SIZE];
  uint8_t size{0};
  void Set(int64_t *src, int ndim) {
    size = ndim;
    for (int i = 0; i < ndim; ++i) {
      value[i] = static_cast<T>(src[i]);
    }
  }
  T* Get() {
    // CHECK_LE(dim, MAX_STACK_VECTOR_SIZE);
    return value;
  }
  static StackVector* ThreadLocal() {
    static thread_local StackVector<type, T> inst;
    return &inst;
  }
};

struct CuDNNThreadEntry {
  cudnnHandle_t handle{nullptr};
  CuDNNThreadEntry() {
    auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
    CUDNN_CALL(cudnnCreate(&handle));
    CUDNN_CALL(cudnnSetStream(handle, stream));
  }
  ~CuDNNThreadEntry() {
    CUDNN_CALL(cudnnDestroy(handle));
  }
  static CuDNNThreadEntry* ThreadLocal() {
    static thread_local CuDNNThreadEntry inst;
    return &inst;
  }
};  // CuDNNThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_CUDNN_CUDNN_UTILS_H_