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
 * \brief EdgeTPU runtime that can run tflite model compiled
 *        for EdgeTPU containing only tvm PackedFunc.
 * \file edgetpu_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_EDGETPU_EDGETPU_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_EDGETPU_EDGETPU_RUNTIME_H_

#include <string>
#include <memory>

#include "../tflite/tflite_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief EdgeTPU runtime.
 *
 *  This runtime can be accessed in various languages via
 *  the TVM runtime PackedFunc API.
 */
class EdgeTPURuntime : public TFLiteRuntime {
 public:
  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "EdgeTPURuntime";
  }

  /*!
   * \brief Initialize the edge TPU tflite runtime with tflite model and context.
   * \param tflite_model_bytes The tflite model.
   * \param ctx The context where the tflite model will be executed on.
   */
  void Init(const std::string& tflite_model_bytes,
            TVMContext ctx);

 private:
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_EDGETPU_EDGETPU_RUNTIME_H_
