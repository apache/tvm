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
 * \file vitis_ai_runtime.cc
 */

#include "vitis_ai_runtime.h"

#include <tvm/runtime/registry.h>

#include <cassert>
#include <fstream>
#include <streambuf>
#include <string>
#include <vector>

using namespace pyxir::runtime;

namespace tvm {
namespace runtime {

VitisAIRuntime::VitisAIRuntime(const std::string& symbol_name, const Array<String> const_names,
                               const std::string& serialized_rt_mod,
                               const std::string& export_rt_mod_path)
    : symbol_name_(symbol_name),
      const_names_(const_names),
      export_rt_mod_path_(export_rt_mod_path) {
  std::istringstream sstream(serialized_rt_mod);
  rt_mod_.reset(new RuntimeModule());
  rt_mod_->deserialize(sstream);
  in_tensor_names_ = rt_mod_->get_in_tensor_names();
  out_tensor_names_ = rt_mod_->get_out_tensor_names();
}

VitisAIRuntime::VitisAIRuntime(const std::string& symbol_name, const std::string& xgraph_str,
                               const Array<String> const_names, const std::string& dpu_target,
                               const std::string& build_dir, const std::string& work_dir,
                               const std::string& export_rt_mod_path)
    : symbol_name_(symbol_name),
      const_names_(const_names),
      export_rt_mod_path_(export_rt_mod_path) {
  std::istringstream xgraph_sstream(xgraph_str);
  pyxir::XGraphHolder xgraph = std::make_shared<pyxir::graph::XGraph>("");
  pyxir::read(xgraph, xgraph_sstream);
  in_tensor_names_ = xgraph->get_input_names();
  out_tensor_names_ = xgraph->get_meta_attr("tvm_out_tensors").get_strings();

  pyxir::partition(xgraph, std::vector<std::string>{dpu_target}, "");

  pyxir::RunOptionsHolder run_options(new pyxir::runtime::RunOptions());
  run_options->on_the_fly_quantization = true;
  run_options->build_dir = build_dir;
  run_options->export_runtime_module_path = export_rt_mod_path_;
  if (!work_dir.empty()) run_options->work_dir = work_dir;
  rt_mod_ =
      pyxir::build_rt(xgraph, dpu_target, in_tensor_names_, out_tensor_names_, "vai", run_options);
}

Module VitisAIRuntimeCreate(const std::string& name, const std::string& xgraph_str,
                            const std::string& dpu_target, const std::string& build_dir,
                            const std::string& work_dir, const std::string& export_rt_mod_path) {
  Array<String> const_vars;
  auto exec = make_object<VitisAIRuntime>(name, xgraph_str, const_vars, dpu_target, build_dir,
                                          work_dir, export_rt_mod_path);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.vitis_ai_runtime.from_xgraph").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = VitisAIRuntimeCreate(args[0], args[1], args[2], args[3], args[4], args[5]);
});

Module VitisAIRuntimeCreate(const std::string& name, const std::string& serialized_rt_mod,
                            const std::string& export_rt_mod_path) {
  Array<String> const_vars;
  auto exec = make_object<VitisAIRuntime>(name, const_vars, serialized_rt_mod, export_rt_mod_path);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.vitis_ai_runtime.from_rt_mod").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string load_rt_mod_path = args[1];
  assert(!load_rt_mod_path.empty());
  std::ifstream in_file(load_rt_mod_path);
  std::stringstream buffer;
  buffer << in_file.rdbuf();
  std::string serialized_rt_mod = buffer.str();
  in_file.close();
  *rv = VitisAIRuntimeCreate(args[0], serialized_rt_mod, args[2]);
});

Module VitisAIRuntimeLoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string symbol_name;
  std::vector<std::string> const_vars;
  std::string serialized_rt_mod;
  std::string export_rt_mod_path;
  stream->Read(&serialized_rt_mod);
  stream->Read(&export_rt_mod_path);
  stream->Read(&symbol_name);
  stream->Read(&const_vars);
  Array<String> const_names;
  for (const auto& it : const_vars) {
    const_names.push_back(it);
  }
  auto exec =
      make_object<VitisAIRuntime>(symbol_name, const_names, serialized_rt_mod, export_rt_mod_path);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_VitisAIRuntime")
    .set_body_typed(VitisAIRuntimeLoadFromBinary);

void VitisAIRuntime::SaveToBinary(dmlc::Stream* stream) {
  std::ostringstream sstream;
  rt_mod_->serialize(sstream);
  stream->Write(sstream.str());
  stream->Write(export_rt_mod_path_);
  stream->Write(symbol_name_);
  std::vector<std::string> consts;
  for (const auto& it : const_names_) {
    consts.push_back(it);
  }
  stream->Write(consts);

  // If export_rt_mod_path_ member variable is set, we will additionally export the PyXIR
  //  runtime_module to the specified file
  if (!export_rt_mod_path_.empty()) {
    std::ofstream out_file(export_rt_mod_path_);
    out_file << sstream.str();
    out_file.close();
  }
}

PackedFunc VitisAIRuntime::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
  } else if (name == "get_const_vars") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
  } else if ("__init_" + this->symbol_name_ == name) {
    // The function to initialize constant tensors.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1U);
      this->initialized_ = true;
      *rv = 0;
    });
  } else if (this->symbol_name_ == name) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      // Initialize input tensors
      DLTensor* inputs = args[0];
      std::vector<pyxir::XBufferHolder> in_tensors;
      std::vector<ssize_t> in_shape;
      for (int i = 0; i < inputs->ndim; ++i) in_shape.push_back(inputs->shape[i]);
      in_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(
          new pyxir::XBuffer(reinterpret_cast<void*>(static_cast<float*>(inputs->data)), 4, "f",
                             in_shape.size(), in_shape, false, false)));

      // Initialize output tensors
      std::vector<pyxir::XBufferHolder> out_tensors;
      for (unsigned i = 0; i < out_tensor_names_.size(); ++i) {
        DLTensor* output_tensor = args[args.size() - out_tensor_names_.size() + i];
        std::vector<ssize_t> out_shape;
        for (int i = 0; i < output_tensor->ndim; ++i) out_shape.push_back(output_tensor->shape[i]);
        void* output_data = reinterpret_cast<void*>(static_cast<float*>(output_tensor->data));
        out_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(
            new pyxir::XBuffer(output_data, 4, "f", out_shape.size(), out_shape, false, false)));
      }

      // Execute the subgraph.
      rt_mod_->execute(in_tensors, out_tensors);
    });
  } else {
    return PackedFunc();
  }
}

}  // namespace runtime
}  // namespace tvm
