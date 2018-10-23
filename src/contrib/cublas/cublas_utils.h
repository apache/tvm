/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cudnn utils function
 */

#ifndef TVM_CONTRIB_CUBLAS_CUBLAS_UTILS_H_
#define TVM_CONTRIB_CUBLAS_CUBLAS_UTILS_H_

#include <dmlc/logging.h>

extern "C" {
#include <cublas_v2.h>
}

namespace tvm {
namespace contrib {

inline const char* get_cublas_error_string(int error) {
  switch(error) {
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unrecognized error";
}

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(fn)			\
  do {						\
    int error = (int)(fn);			\
    CHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << "CUBLAS: " << get_cublas_error_string(error); \
  } while(0);
#endif


struct CuBlasThreadEntry {
  CuBlasThreadEntry();
  ~CuBlasThreadEntry();
  cublasHandle_t handle{nullptr};
  static CuBlasThreadEntry* ThreadLocal();
};  // CuBlasThreadEntry


}}

#endif //
