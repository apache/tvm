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

typedef void (*CblasFloat)(float* a, float* b, float* out);
typedef void (*CblasDouble)(double* a, double* b, double* out);

class CblasModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id = "") const override {
    return {"/tmp/relay_cblas_lib_" + id + ".so"};
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
    _curr_id = GetSubgraphID(name);
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      CHECK_EQ(args.size(), 3U);
      runtime::NDArray data = args[0];
      runtime::NDArray weight = args[1];
      runtime::NDArray out = args[2];

      const DLTensor* dptr = data.operator->();
      std::string encoded_name = _prefix + _curr_id;

      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        float* d_data = reinterpret_cast<float*>(data->data);
        float* weight_data = reinterpret_cast<float*>(weight->data);
        float* out_data = reinterpret_cast<float*>(out->data);

        auto func_s_ = reinterpret_cast<CblasFloat>(GetSymbol(encoded_name));
        (*func_s_)(d_data, weight_data, out_data);
      } else if (runtime::TypeMatch(dptr->dtype, kDLFloat, 64)) {
        double* d_data = reinterpret_cast<double*>(data->data);
        double* weight_data = reinterpret_cast<double*>(weight->data);
        double* out_data = reinterpret_cast<double*>(out->data);

        auto func_s_ = reinterpret_cast<CblasDouble>(GetSymbol(encoded_name));
        (*func_s_)(d_data, weight_data, out_data);
      } else {
        LOG(FATAL) << "Only support float32 and float64 types.";
      }
      
      *rv = out;
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
    return "CblasModule";
  }

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "CBLAS expects a single dense op.";

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    std::string encoded_id = _prefix + id;
    std::string code = "";

    // Args: ID
    std::vector<std::string> args;
    args.push_back(encoded_id);

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

      code = "DENSE_FP" + std::to_string(type_node->dtype.bits()) + "(" +
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

 private:
  std::string _curr_id;
  std::string _prefix = "cblas_";
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
