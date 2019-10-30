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

#include "dnnl.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <string>

namespace tvm {
namespace runtime {
namespace contrib {

void DNNLModule::Init() {
  if (!IsOpen()) {
    CHECK_GT(lib_path_.size(), 0U);
    Open({lib_path_});
  }
}

runtime::PackedFunc DNNLModule::GetFunction(
    const std::string& name, const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->Init();
    });
  } else {
    std::string curr_id = GetSubgraphID(name);

    CHECK(IsOpen()) << "The external module has not been built or failed to open.\n";

    return PackedFunc([sptr_to_self, curr_id, this](TVMArgs args, TVMRetValue* rv) {
      const DLTensor* dptr = ((runtime::NDArray)args[0]).operator->();
      runtime::NDArray out_arg = args[args.size() - 1];
      auto out = reinterpret_cast<float*>(out_arg->data);

      // Get function from the library
      std::string encoded_name = kDnnlPrefix + curr_id;
      auto func_s = reinterpret_cast<DnnlSubgraphFunc>(GetSymbol(encoded_name));

      // Reinterpret data and function to the right type and invoke
      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        DnnlPackedArgs packed_args;
        packed_args.data = reinterpret_cast<void**>(malloc(sizeof(float*) * args.size()));
        for (int i = 0; i < args.size() - 1; ++i) {
          runtime::NDArray arg = args[i];
          packed_args.data[i] = reinterpret_cast<float*>(arg->data);
        }
        (*func_s)(packed_args, out);
      } else {
        LOG(FATAL) << "Only support float32 type.";
      }
      *rv = out;
    });
  }
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm


