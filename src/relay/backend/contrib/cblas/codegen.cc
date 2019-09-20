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
#include <dlpack/dlpack.h>
#include <stdlib.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/contrib_codegen.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/util.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace relay {
namespace contrib {

class CblasModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id = "") const override {
    return {"/tmp/relay_cblas_lib_" + id + ".so"};
  }

  const std::string GetPrefix() const override {
    return "cblas_";
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
    return "CblasModule";
  }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "CBLAS expects a single dense op.";

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    std::string code = "";

    // Args: ID
    std::vector<std::string> args;
    args.push_back(GetPrefix() + id);

    if (IsOp(call, "nn.dense")) {
      auto ishape = GetShape(call->args[0]);
      auto wshape = GetShape(call->args[1]);

      // Args: M, N, K
      args.push_back(std::to_string(ishape[0]));
      args.push_back(std::to_string(wshape[1]));
      args.push_back(std::to_string(ishape[1]));

      auto type_node = call->checked_type().as<TensorTypeNode>();
      CHECK(type_node != nullptr);
      CHECK(type_node->dtype.is_float()) << "Only support float types";
      auto bits = type_node->dtype.bits();
      SetSubgraphInfo(id, DLDataType{kDLFloat, static_cast<uint8_t>(bits), 1}, 3);

      code = "DENSE_FP" + std::to_string(bits) + "(" +
             args[0] + ", " + args[1] + ", " + args[2] + ", " + args[3] + ");";
    } else {
      LOG(FATAL) << "CBLAS expects a single dense op.";
    }

    if (!std::getenv("MKLROOT")) {
      LOG(FATAL) << "MKLROOT not found. Did you source mklvars.sh?";
    }
    std::string lib_src_name = "/tmp/relay_cblas_lib_" + id + ".cc";
    std::string lib_name = "/tmp/relay_cblas_lib_" + id + ".so";

    // Prepare library source
    std::string cmd = "cp src/relay/backend/contrib/cblas/libs.cc " + lib_src_name;
    std::system(cmd.c_str());

    cmd = "echo \"" + code + "\" >> " + lib_src_name;
    std::system(cmd.c_str());

    cmd = "g++ -O2 -Wall -std=c++11 -shared -fPIC " + lib_src_name + " -o " + lib_name +
          " -ldl -lpthread -lm -lmkl_rt";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile CBLAS library. Error code: " << ret;
    }
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 */
runtime::Module CblasCompiler(const Expr& expr) {
  std::shared_ptr<CblasModuleNode> n = std::make_shared<CblasModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.cblas")
.set_body_typed(CblasCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
