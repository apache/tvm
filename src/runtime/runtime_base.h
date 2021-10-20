/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef TVM_RUNTIME_RUNTIME_BASE_H_
#define TVM_RUNTIME_RUNTIME_BASE_H_

#include <tvm/runtime/c_runtime_api.h>

#include <stdexcept>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END()                                         \
  }                                                       \
  catch (::tvm::runtime::EnvErrorAlreadySet & _except_) { \
    return -2;                                            \
  }                                                       \
  catch (std::exception & _except_) {                     \
    return TVMAPIHandleException(_except_);               \
  }                                                       \
  return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize)                    \
  }                                                       \
  catch (::tvm::runtime::EnvErrorAlreadySet & _except_) { \
    return -2;                                            \
  }                                                       \
  catch (std::exception & _except_) {                     \
    Finalize;                                             \
    return TVMAPIHandleException(_except_);               \
  }                                                       \
  return 0;  // NOLINT(*)

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int TVMAPIHandleException(const std::exception& e);

#endif  // TVM_RUNTIME_RUNTIME_BASE_H_
