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
 * \file src/runtime/contrib/cublas/cublas_json_runtime.cc
 * \brief A simple JSON runtime for CUBLAS.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

// TODO(@apeskov): Have to mute warning from cublas headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command

#include "cublas_tensor_requisite.h"
#include "cublas_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class CUBLASJSONRuntime : public JSONRuntimeBase {
 public:
  CUBLASJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        next_unique_eid_offset_(data_entry_.size()),
        run_arg_eid_(input_var_eid_) {
  }

  /* Unused stub implementation */
  void Run() override { LOG(FATAL) << "Unreachable code"; }

  /* Thread safe implementation of Run. Keep runtime instance immutable */
  void Run(const TVMArgs& args) const {
  }

  /* Override GetFunction to reimplement Run method */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
            << "Found mismatch in the number of provided data entries and required.";

        Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

 private:
};

runtime::Module CublasJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<CublasJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CublasJSONRuntimeCreate").set_body_typed(CubblasJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cublas_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<CublasJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
