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
 * \brief Defines an implementation of Module-based Model Runtime Interface that works with
 *        Ahead-of-Time compilation.
 * \file aot_executor.h
 */
#ifndef TVM_RUNTIME_AOT_EXECUTOR_AOT_EXECUTOR_H_
#define TVM_RUNTIME_AOT_EXECUTOR_AOT_EXECUTOR_H_

#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {

class TVM_DLL AotExecutor : public ModuleNode {
 public:
  /*!
   * \brief Implements member function lookup for this Module for the frontend.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "AotExecutor"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable; }

  void Run();

  /*!
   * \brief Initialize the AOT executor with metadata, runtime::Module, and device.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param devs A 1-element vector. The Device which AOT compute will run on. Currently, only
   *  Device(kDLCPU, 0) is supported.
   */
  AotExecutor(tvm::runtime::Module module, const std::vector<Device>& devs);

  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  int GetInputIndex(const std::string& name);

  /*!
   * \brief Get the input name given the index of input.
   * \param index The index of the input.
   * \return The name of input.
   */
  std::string GetInputName(int index);

  /*!
   * \brief Get the output index given the name of output.
   * \param name The name of the output.
   * \return The index of output.
   */
  int GetOutputIndex(const std::string& name);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in);
  /*!
   * \brief set index-th input to the graph without copying the data
   * \param index The input index.
   * \param data_ref The input data that is referred.
   */
  void SetInputZeroCopy(int index, DLTensor* data_ref);
  /*!
   * \brief set index-th output to the graph without copying the data.
   * \param index The output index.
   * \param data_ref The output data that is referred.
   */
  void SetOutputZeroCopy(int index, DLTensor* data_ref);
  /*!
   * \brief Get the number of outputs
   *
   * \return The number of outputs from graph.
   */
  int NumOutputs() const;
  /*!
   * \brief Get the number of inputs
   *
   * \return The number of inputs to the graph.
   */
  int NumInputs() const;
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
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void CopyOutputTo(int index, DLTensor* data_out);

 private:
  /*! \brief Metadata provided to the runtime from the compiler. */
  metadata::Metadata metadata_;

  /*! \brief Runtime module which contains the AOT top-level function. */
  Module module_;

  /*! \brief The devices which should be used to execute the computations. */
  std::vector<Device> devices_;

  /*! \brief Holds one NDArray per function argument in the same order. */
  std::vector<NDArray> args_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_AOT_EXECUTOR_AOT_EXECUTOR_H_
