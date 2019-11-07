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
#include "gcc.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <string>

namespace tvm {
namespace runtime {
namespace contrib {

runtime::PackedFunc GccModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->Init(args[0]);
    });
  } else {
    std::string curr_id = GetSubgraphID(name);

    CHECK(IsLoaded()) << "The external module has not been built or failed to open.\n";
    // Generate an external packed function
    return PackedFunc([sptr_to_self, curr_id, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_GT(args.size(), 0U) << "No input is provided.";

      NDArray input0 = args[0];
      const DLTensor* dptr = input0.operator->();
      CHECK(dptr) << "Expect a NDArray as the input.";
      runtime::NDArray out_arg = args[args.size() - 1];
      auto out = reinterpret_cast<float*>(out_arg->data);

      // Get function from the library
      std::string encoded_name = "gcc_" + curr_id;
      auto func_s = reinterpret_cast<GccSubgraphFunc>(this->GetSymbol(encoded_name.c_str()));

      // Reinterpret data and function to the right type and invoke
      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        GccPackedArgs packed_args;
        packed_args.data = reinterpret_cast<float**>(malloc(sizeof(float*) * args.size()));
        for (int i = 0; i < args.size() - 1; ++i) {
          runtime::NDArray arg = args[i];
          packed_args.data[i] = reinterpret_cast<float*>(arg->data);
        }
        (*func_s)(packed_args, out);
      } else {
        LOG(FATAL) << "Only float32 values are supported.";
      }
      *rv = out;
    });
  }
}

TVM_REGISTER_GLOBAL("module.loadfile_gcc_so")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<DSOModuleNode> n = std::make_shared<GccModuleNode>();
    n->Init(args[0]);
    *rv = runtime::Module(n);
  });

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
