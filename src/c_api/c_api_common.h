/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_common.h
 * \brief Common fields of all C APIs
 */
#ifndef TVM_C_API_C_API_COMMON_H_
#define TVM_C_API_C_API_COMMON_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/c_api.h>
#include <vector>
#include <string>
#include "./c_api_registry.h"

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return TVMAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &_except_) { Finalize; return TVMAPIHandleException(_except_); } return 0; // NOLINT(*)

void TVMAPISetLastError(const char* msg);

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int TVMAPIHandleException(const dmlc::Error &e) {
  TVMAPISetLastError(e.what());
  return -1;
}

#endif  // TVM_C_API_C_API_COMMON_H_
