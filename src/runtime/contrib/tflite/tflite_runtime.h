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
 * \brief Tflite runtime that can run tflite model
 *        containing only tvm PackedFunc.
 * \file tflite_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_TFLITE_TFLITE_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TFLITE_TFLITE_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <vector>
#include <string>
#include <memory>

namespace tvm {
namespace runtime {


/*!
 * \brief Tflite runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class TFLiteRuntime : public ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const {
    return "TFLiteRuntime";
  }

  /*!
   * \brief Invoke the internal tflite interpreter and run the whole model in 
   * dependency order.
   */
  void Invoke();

  /*!
   * \brief Initialize the tflite runtime with tflite model and context.
   * \param tflite_model_bytes The tflite model.
   * \param ctx The context where the tflite model will be executed on.
   */
  void Init(const std::string& tflite_model_bytes,
            TVMContext ctx);

  /*!
   * \brief set index-th input to the model.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in);
  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) const;
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index) const;

  // TFLite interpreter
  std::unique_ptr<tflite::Interpreter> interpreter_;
  // TVM context
  TVMContext ctx_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TFLITE_TFLITE_RUNTIME_H_
