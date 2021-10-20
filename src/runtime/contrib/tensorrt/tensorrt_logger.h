/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_logger.h
 * \brief Contains TensorRTLogger class which is required by TRT and used to
 * print info, warnings, and errors.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_LOGGER_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_LOGGER_H_

#include <tvm/runtime/logging.h>

#include "NvInfer.h"
#include "tensorrt_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

/*! \brief Logger for TensorRT info/warning/errors. */
class TensorRTLogger : public nvinfer1::ILogger {
 public:
  TensorRTLogger() : TensorRTLogger(Severity::kWARNING) {}
  explicit TensorRTLogger(Severity severity) : reportable_severity(severity) {}
  void log(Severity severity, const char* msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        LOG(ERROR) << "INTERNAL_ERROR: " << msg;
        break;
      case Severity::kERROR:
        LOG(ERROR) << "ERROR: " << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << "WARNING: " << msg;
        break;
      case Severity::kINFO:
        LOG(INFO) << "INFO: " << msg;
        break;
#if TRT_VERSION_GE(5, 1, 5)
      case Severity::kVERBOSE:
        DLOG(INFO) << "VERBOSE: " << msg;
        break;
#endif
      default:
        LOG(INFO) << "UNKNOWN: " << msg;
        break;
    }
  }

 private:
  Severity reportable_severity{Severity::kWARNING};
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_LOGGER_H_
