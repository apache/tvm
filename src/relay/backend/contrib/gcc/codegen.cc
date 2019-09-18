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

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param func_s The function symbol retrieved from the external library.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  runtime::PackedFunc InvokeExternFunc(const std::string& name,
                                       const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    std::string _curr_id = GetSubgraphID(name);
    std::string encoded_id = _prefix + _curr_id;
    func_s_ = reinterpret_cast<GccBinaryFunc>(GetSymbol(encoded_id));

    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      CHECK_EQ(args.size(), 3U);
      runtime::NDArray a = args[0];
      ExternalTensor lhs;
      lhs.data = a->data;
      lhs.ndim = a.Shape().size();
      lhs.shape = new int64_t[lhs.ndim];

      runtime::NDArray b = args[1];
      ExternalTensor rhs;
      rhs.data = b->data;
      rhs.ndim = b.Shape().size();
      rhs.shape = new int64_t[rhs.ndim];

      runtime::NDArray c = args[2];
      ExternalTensor out;
      out.data = c->data;
      out.ndim = c.Shape().size();
      out.shape = c.Shape().data();

      for (int i = 0; i < lhs.ndim; i++) {
        lhs.shape[i] = a.Shape()[i];
        rhs.shape[i] = b.Shape()[i];
      }
      (*func_s_)(lhs, rhs, &out);
      *rv = c;
    });
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

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    std::string encoded_id = _prefix + id;
    std::string code = "GCC_BINARY_OP(" + encoded_id + ", ";

    if (IsOp(call, "add")) {
      code += "+";
    } else if (IsOp(call, "subtract")) {
      code += "-";
    } else if (IsOp(call, "multiply")) {
      code += "*";
    } else {
      LOG(FATAL) << "Unrecognized op: ";
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

 private:
  GccBinaryFunc func_s_;
  std::string _prefix = "gcc_";
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
