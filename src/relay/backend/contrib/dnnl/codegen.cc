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

class DNNLModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id) const override {
    return {"/tmp/relay_dnnl_lib_" + id + ".so"};
  }

  const std::string GetPrefix() const override {
    return "dnnl_";
  }

  /*!
   * \brief Get the source code of the external module.
   *
   * \param format The format of the source code.
   *
   * \return The source code of the external library module in the text form.
   */
  TVM_DLL std::string GetSource(const std::string& format = "") override { return ""; }

  const char* type_key() const override { return "DNNLModule"; }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op.";

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    std::string encoded_id = GetPrefix() + id;
    std::string code = "";

    // Args: ID
    std::vector<std::string> args;
    args.push_back(encoded_id);

    if (IsOp(call, "nn.conv2d")) {
      SetSubgraphInfo(id, DLDataType{kDLFloat, 32, 1}, 3);
      code = "CONV2D";
      const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();

      auto ishape = GetShape(call->args[0]);
      auto wshape = GetShape(call->args[1]);

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }

      // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
      args.push_back(std::to_string(wshape[0]));
      args.push_back(std::to_string(conv2d_attr->groups));
      args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImm>()->value));
      args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImm>()->value));
      args.push_back(std::to_string(wshape[2]));
      args.push_back(std::to_string(wshape[3]));
      args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImm>()->value));
      args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImm>()->value));
    } else if (IsOp(call, "nn.dense")) {
      SetSubgraphInfo(id, DLDataType{kDLFloat, 32, 1}, 3);
      code = "DENSE";
      auto ishape = GetShape(call->args[0]);
      auto wshape = GetShape(call->args[1]);

      // Args: N, C, O
      args.push_back(std::to_string(ishape[0]));
      args.push_back(std::to_string(ishape[1]));
      args.push_back(std::to_string(wshape[0]));

    } else if (IsOp(call, "nn.relu")) {
      SetSubgraphInfo(id, DLDataType{kDLFloat, 32, 1}, 2);
      code = "RELU";
      auto ishape = GetShape(call->args[0]);

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }
    } else if (IsOp(call, "nn.batch_norm")) {
      SetSubgraphInfo(id, DLDataType{kDLFloat, 32, 1}, 8);
      code = "BN";
      const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
      auto ishape = GetShape(call->args[0]);

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
        }

        // Args: epilson
        args.push_back(std::to_string(bn_attr->epsilon));
    } else {
      LOG(FATAL) << "DNNL expects a single convolution or dense op.";
    }
    Compile(id, code, args);
  }

 private:
  void Compile(std::string id, std::string code, std::vector<std::string> args) {
    // FIXME: Now we compile N libraries for N subgraphs, but we should merge them to one.
    std::string lib_src_name = "/tmp/relay_dnnl_lib_" + id + ".cc";
    std::string lib_name = "/tmp/relay_dnnl_lib_" + id + ".so";

    // Prepare library source
    std::string cmd = "cp src/relay/backend/contrib/dnnl/libs.cc " + lib_src_name;
    std::system(cmd.c_str());

    // Push macro implementation
    bool first = true;
    std::string macro = code + "(";
    for (auto arg : args) {
      if (!first) macro += ", ";
      first = false;
      macro += arg;
    }
    macro += ")";
    cmd = "echo \"" + macro + ";\" >> " + lib_src_name;
    std::system(cmd.c_str());

    // Compile
    cmd = "g++ -O2 -Wall -std=c++11 -shared -fPIC " + lib_src_name +
          " -o " + lib_name + " -ldl -lpthread -lm -ldnnl";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile DNNL library. Error code: " << ret;
    }
  }

  std::string _curr_id;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const Expr& expr) {
  std::shared_ptr<DNNLModuleNode> n = std::make_shared<DNNLModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.dnnl").set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
