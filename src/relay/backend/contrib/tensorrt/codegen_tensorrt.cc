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
 * \file src/relay/backend/contrib/tensorrt/codegen_tensorrt.cc
 * \brief Implementation of TensorRT codegen APIs.
 */

#include <tvm/node/serialization.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Generates a TensorRTModule from a relay expression. This "compilation"
 * does not require TensorRT since the actual conversion using TensorRT APIs is
 * deferred until runtime. This step simply serializes the relay program into a
 * string.
 */
class TensorRTModuleCodegen : public CSourceModuleCodegenBase {
 public:
  runtime::Module CreateCSourceModule(const NodeRef& ref) override {
    std::string serialized_subgraph;
    if (ref->IsInstance<FunctionNode>()) {
      serialized_subgraph = SaveJSON(Downcast<Function>(ref)->body);
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      // TODO(trevmorr): support multiple functions. It is currently not
      // possible for there to be more than one TRT func, so not a problem yet.
      for (const auto& it : mod->functions) {
        serialized_subgraph = SaveJSON(Downcast<Function>(it.second)->body);
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }
    const PackedFunc* pf =
        runtime::Registry::Get("tvm.contrib.tensorrt.create");
    CHECK(pf != nullptr)
        << "tvm.contrib.tensorrt.create was not found in the registry.";
    return (*pf)(serialized_subgraph);
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compiles it into a runtime module.
 */
runtime::Module TrtCompiler(const NodeRef& ref) {
  TensorRTModuleCodegen tensorrt;
  return tensorrt.CreateCSourceModule(ref);
}

TVM_REGISTER_API("relay.ext.tensorrt").set_body_typed(TrtCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
