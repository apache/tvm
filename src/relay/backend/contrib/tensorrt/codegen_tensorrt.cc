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
#include <unordered_map>

#include "../../../../runtime/contrib/tensorrt/tensorrt_module.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"
#if TVM_GRAPH_RUNTIME_TENSORRT
#include "NvInfer.h"
#endif  // TVM_GRAPH_RUNTIME_TENSORRT

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief Generates a TensorRTModule from a relay expression. This "compilation"
 * does not require TensorRT since the actual conversion using TensorRT APIs is
 * deferred until runtime. This step simply serializes the relay functions into
 * strings.
 */
class TensorRTModuleCodegen : public CSourceModuleCodegenBase {
 public:
  /*!
   * \brief Serializes a function and stores it in serialized_subgraphs_ so that
   * it can be included in the TensorRT module.
   * \param func A relay function to add to the TensorRT module.
   */
  void GenFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    serialized_subgraphs_[sid] = SaveJSON(func);
  }

  /*!
   * \brief Creates the TensorRT module from the Relay function or IRModule.
   * \param ref An object ref that could be either a Relay function or IRModule.
   * \return The TensorRT runtime module.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    if (ref->IsInstance<FunctionNode>()) {
      GenFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }
    return runtime::TensorRTModuleCreate(serialized_subgraphs_);
  }

 private:
  /*! \brief Map of external symbol to serialized Relay functions. */
  std::unordered_map<std::string, std::string> serialized_subgraphs_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compiles it into a runtime module.
 */
runtime::Module TrtCompiler(const ObjectRef& ref) {
  TensorRTModuleCodegen tensorrt;
  return tensorrt.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.tensorrt").set_body_typed(TrtCompiler);

/*!
 * \brief Get TensorRT version that TVM was compiled against.
 * \return TensorRT version as a list of [major, minor, patch], or an empty list
 * if not compiled against TensorRT.
 */
Array<Integer> GetTrtVersion() {
#if TVM_GRAPH_RUNTIME_TENSORRT
  return {Integer(NV_TENSORRT_MAJOR), Integer(NV_TENSORRT_MINOR),
          Integer(NV_TENSORRT_PATCH)};
#else
  return {};
#endif  // TVM_GRAPH_RUNTIME_TENSORRT
}

TVM_REGISTER_GLOBAL("relay._transform.GetTrtVersion")
    .set_body_typed(GetTrtVersion);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
