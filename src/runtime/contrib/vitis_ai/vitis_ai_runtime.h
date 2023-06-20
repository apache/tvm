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
 * \brief Vitis-AI runtime that can run model
 *        containing only tvm PackedFunc.
 * \file vitis_ai_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_VITIS_AI_VITIS_AI_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_VITIS_AI_VITIS_AI_RUNTIME_H_
#include <dlpack/dlpack.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
// clang-format off
#include <memory>
#include <string>
#include <vector>
// clang-format on
#include <pyxir/pyxir.hpp>
#include <pyxir/runtime/run_options.hpp>

namespace tvm {
namespace runtime {

/*!
 * \brief VAI runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class VitisAIRuntime : public ModuleNode {
 public:
  /*!
   * \brief Create VitisAI runtime from serialized XGraph
   * \param symbol_name The name of the function.
   * \param const_names The names of each constant in the sub-graph.
   * \param serialized_rt_mod The serialized runtime module.
   * \param export_rt_mod_path The path to the file to be used for exporting the
   *        PyXIR runtime module.
   */
  VitisAIRuntime(const std::string& symbol_name, const Array<String> const_names,
                 const std::string& serialized_rt_mod, const std::string& export_rt_mod);

  /*!
   * \brief Create VitisAI runtime from serialized XGraph
   * \param symbol_name The name of the function.
   * \param xgraph_str serialized XGraph representation
   * \param const_names The names of each constant in the sub-graph.
   * \param dpu_target The Vitis-AI DPU target identifier (e.g. DPUCADX8G, DPUCZDX8G-zcu104).
   * \param build_dir The directory to be used for Vitis-AI build files.
   * \param work_dir The directory to be used for Vitis-AI work files.
   * \param export_rt_mod_path The path to the file to be used for exporting the
   *        PyXIR runtime module.
   */
  VitisAIRuntime(const std::string& symbol_name, const std::string& xgraph_str,
                 const Array<String> const_names, const std::string& dpu_target,
                 const std::string& build_dir, const std::string& work_dir,
                 const std::string& export_runtime_module_path);

  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const { return "VitisAIRuntime"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  };

  /*!
   * \brief Serialize the content of the pyxir directory and save it to
   *        binary stream.
   * \param stream The binary stream to save to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

 private:
  /*! \brief The only subgraph name for this module */
  std::string symbol_name_;
  /*! \brief The required constant names */
  Array<String> const_names_;
  /*! \brief The runtime module */
  pyxir::RtModHolder rt_mod_;
  /*! \brief The XGraph input tensor names in the order as provided by TVM */
  std::vector<std::string> in_tensor_names_;
  /*! \brief The XGraph output tensor names in the order as provided by TVM */
  std::vector<std::string> out_tensor_names_;
  /*! \brief The file path for exporting the runtime module if set */
  std::string export_rt_mod_path_;
  /*! \brief Whether constant tensors have been initialized */
  bool initialized_{false};
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_VITIS_AI_VITIS_AI_RUNTIME_H_
