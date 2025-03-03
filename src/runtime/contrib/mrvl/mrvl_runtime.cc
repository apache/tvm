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
 * \file src/runtime/contrib/mrvl/mrvl_runtime.cc
 * \brief runtime implementation for Marvell Software Simulator.
 */

#include <assert.h>
#include <ctype.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "mrvl_sw_runtime_lib.h"

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief A json runtime that compiles the serialized JSON format to a binary for Marvell
hardware and then runs the generated binary using the Marvell software simulator (MlModel).
 * \param symbol_name The name of the subgraph / relay function
 * \param nodes_json The serialized JSON representation of relay function
 * \param bin_code The binary code generated by the Marvell compiler for the subgraph
 */

class MarvellSimulatorModuleNode : public ModuleNode {
 public:
  MarvellSimulatorModuleNode(const std::string& symbol_name, const std::string& nodes_json,
                             const std::string& bin_code)
      : symbol_name_(symbol_name), nodes_json_(nodes_json), bin_code_(bin_code) {
    set_num_inputs_outputs();
  }

  const char* type_key() const { return "mrvl_sim"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = Array<String>{}; });
    } else if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Run(args);
        *rv = 0;
      });
    }
    return PackedFunc(nullptr);
  }

  virtual void SaveToBinary(dmlc::Stream* stream) {
    // Save the symbol name and other data and serialize them to
    // binary format.
    stream->Write(symbol_name_);
    stream->Write(nodes_json_);
    stream->Write(bin_code_);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol_name;
    std::string nodes_json;
    std::string bin_code;
    // Load the symbol_name and other data to construct the module
    ICHECK(stream->Read(&symbol_name))
        << "Marvell-Compiler-ERROR-Internal::Loading symbol name failed";
    ICHECK(stream->Read(&nodes_json))
        << "Marvell-Compiler-ERROR-Internal::Loading nodes json failed";
    ICHECK(stream->Read(&bin_code)) << "Marvell-Compiler-ERROR-Internal::Loading bin code failed";
    auto n = make_object<MarvellSimulatorModuleNode>(symbol_name, nodes_json, bin_code);
    return Module(n);
  }

  /*!
   * \brief Get the source generated by codegen.
   *
   * \param format the format to return.
   * \return A string of JSON.
   */
  String GetSource(const String& format = "json") override { return nodes_json_; }

 protected:
  std::string symbol_name_;
  std::string nodes_json_;
  std::string bin_code_;
  size_t num_inputs_;
  size_t num_outputs_;

  void Run(TVMArgs args) {
    ICHECK_EQ(args.size(), num_inputs_ + num_outputs_)
        << "Marvell-Compiler-ERROR-Internal::Mismatch in number of input & number of output args "
           "to subgraph";
    tvm::runtime::contrib::mrvl::RunMarvellSimulator(args, symbol_name_, bin_code_, num_inputs_,
                                                     num_outputs_);
  }

  void set_num_inputs_outputs() {
    const auto* get_value_from_key = runtime::Registry::Get("tvm.mrvl.find_value_in_KV_pair");

    std::string value_for_inputs = (*get_value_from_key)(nodes_json_, "num_subgraph_inputs");
    num_inputs_ = std::stoi(value_for_inputs);

    std::string value_for_outputs = (*get_value_from_key)(nodes_json_, "num_subgraph_outputs");
    num_outputs_ = std::stoi(value_for_outputs);
  }
};

runtime::Module MarvellSimulatorModuleRuntimeCreate(const String& symbol_name,
                                                    const String& nodes_json,
                                                    const String& bin_code) {
  auto n = make_object<MarvellSimulatorModuleNode>(symbol_name, nodes_json, bin_code);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.mrvl_runtime_create")
    .set_body_typed(MarvellSimulatorModuleRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_mrvl_sim")
    .set_body_typed(MarvellSimulatorModuleNode::LoadFromBinary);
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
