/* * Licensed to the Apache Software Foundation (ASF) under one
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
#include <dlfcn.h>
#include <stdlib.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include "libs.h"

namespace tvm {
namespace relay {
namespace contrib {

typedef void (*GccBinaryFunc)(ExternalTensor a, ExternalTensor b, ExternalTensor* out);

class GccModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id = "") const override {
    return {"/tmp/relay_gcc_lib_" + id + ".so"};
  }

  const std::string GetPrefix() const override {
    return "gcc_";
  }

  /*!
   * \brief Get the source code of the external module.
   *
   * \param format The format of the source code.
   *
   * \return The source code of the external library module in the text form.
   */
  TVM_DLL std::string GetSource(const std::string& format = "") override {
    return "";
  }

  const char* type_key() const override {
    return "GccModule";
  }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "GCC expects a single op.";

    auto ashape = GetShape(call->args[0]);
    auto bshape = GetShape(call->args[1]);

    // Check shape
    CHECK(ashape.size() <= 2 && ashape.size() == bshape.size())
        << "Input shape dimensions are not consistent, " << ashape.size() << " vs. "
        << bshape.size();
    for (int i = 0; i < ashape.size(); ++i) {
      CHECK(ashape[i] == bshape[i]) << "Input shapes are not consistent at dim " << i << ":"
                                    << ashape[i] << " vs. " << bshape[i];
    }

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    SetSubgraphInfo(id, DLDataType{kDLFloat, 32, 1}, 3);
    std::string code =
        "GCC_BINARY_OP_" + std::to_string(ashape.size()) + "D(" + GetPrefix() + id + ", ";

    if (IsOp(call, "add")) {
      code += "+";
    } else if (IsOp(call, "subtract")) {
      code += "-";
    } else if (IsOp(call, "multiply")) {
      code += "*";
    } else {
      LOG(FATAL) << "Unrecognized op: ";
    }

    for (int i = 0; i < ashape.size(); ++i) {
      code += ", " + std::to_string(ashape[i]);
    }
    code += ");";

    // Prepare library source
    std::string lib_src_name = "/tmp/relay_gcc_lib_" + id + ".cc";
    std::string lib_name = "/tmp/relay_gcc_lib_" + id + ".so";
    std::string cmd = "cp src/relay/backend/contrib/gcc/libs.cc " + lib_src_name;
    std::system(cmd.c_str());
    std::system("cp src/relay/backend/contrib/gcc/libs.h /tmp/");
    
    cmd = "echo \"" + code + "\" >> " + lib_src_name;
    std::system(cmd.c_str());

    cmd = "g++ -std=c++11 -shared -fPIC -ldl " + lib_src_name +
          " -o " + lib_name;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile GCC library. Error code: " << ret;
    }
  }
};


/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 *
 * TODO(@zhiics)
 *  1. Let the external compiler ingest a Relay module instead of
 * a single expression/function.
 *  2. Return runtime::Module.
 */
runtime::Module GccCompiler(const Expr& expr) {
  std::shared_ptr<GccModuleNode> n = std::make_shared<GccModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.gcc")
.set_body_typed(GccCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
