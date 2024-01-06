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
 * \file errno_handling.h
 * \brief Common error number handling functions for socket.h and pipe.h
 */
#ifndef TVM_SUPPORT_ERRNO_HANDLING_H_
#define TVM_SUPPORT_ERRNO_HANDLING_H_
#include <errno.h>

#include "ssize.h"

namespace tvm {
namespace support {
/*!
 * \brief Call a function and retry if an EINTR error is encountered.
 *
 *  Socket operations can return EINTR when the interrupt handler
 *  is registered by the execution environment(e.g. python).
 *  We should retry if there is no KeyboardInterrupt recorded in
 *  the environment.
 *
 * \note This function is needed to avoid rare interrupt event
 *       in long running server code.
 *
 * \param func The function to retry.
 * \return The return code returned by function f or error_value on retry failure.
 */
template <typename FuncType, typename GetErrorCodeFuncType>
inline ssize_t RetryCallOnEINTR(FuncType func, GetErrorCodeFuncType fgeterrorcode) {
  ssize_t ret = func();
  // common path
  if (ret != -1) return ret;
  // less common path
  do {
    if (fgeterrorcode() == EINTR) {
      // Call into env check signals to see if there are
      // environment specific(e.g. python) signal exceptions.
      // This function will throw an exception if there is
      // if the process received a signal that requires TVM to return immediately (e.g. SIGINT).
      runtime::EnvCheckSignals();
    } else {
      // other errors
      return ret;
    }
    ret = func();
  } while (ret == -1);
  return ret;
}
}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_ERRNO_HANDLING_H_
