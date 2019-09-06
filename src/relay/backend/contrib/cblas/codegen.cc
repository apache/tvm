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

typedef void (*CblasFloat)(float* a, float* b, float* out, int M, int N, int K);
// typedef void (*CblasDouble)(float* a, float* b, float* out);

class CblasModuleNode : public runtime:: ModuleNode {
 public:
  CblasModuleNode() = default;
  ~CblasModuleNode() {
    Close();
  }

  // void Init(const std::string& bin_path);
  // void Exec(const std::string& fun_name, const TVMArgs& args);

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    CHECK(handle_) << "The external cblas module has not been built yet."
                   << "\n";
    if (name == "nn.dense") {
      func_s_ = reinterpret_cast<CblasFloat>(GetSymbol(name));
      char* error = dlerror();
      if (error != NULL) {
        LOG(FATAL) << error;
        return PackedFunc();
      }
      return CallDense(sptr_to_self);
    } else {
      LOG(INFO) << "Only nn.dense is supported so far";
      return PackedFunc();
    }
  }

  PackedFunc CallDense(const std::shared_ptr<ModuleNode>& sptr_to_self) {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      CHECK_EQ(args.size(), 3U);
      runtime::NDArray data = args[0];
      runtime::NDArray weight = args[1];
      runtime::NDArray out = args[2];

      const DLTensor* dptr = data.operator->();
      CHECK(runtime::TypeMatch(dptr->dtype, kDLFloat, 32));

      float* d_data = reinterpret_cast<float*>(data->data);
      float* weight_data = reinterpret_cast<float*>(weight->data);
      float* out_data = reinterpret_cast<float*>(out->data);

      // Blas is column major. So we pass B, A, C
      int M = CountColumn(weight);
      int N = CountRow(data);
      int K = CountColumn(data);
      (*func_s_)(weight_data, d_data, out_data, M, N, K);
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

  const char* type_key() const final {
    return "CblasModule";
  }

  void Build(const Expr& expr) {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "CBLAS expects a single convolution or dense op.";

    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "CBLAS expects a single convolution or dense op.";
    Op op = GetRef<Op>(op_node);
    if (op == Op::Get("nn.conv2d")) {
      const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
      // TODO(@zhiics) Generate the template.
    } else if (op == Op::Get("nn.dense")) {
      // TODO(@zhiics) Generate the template.
      const auto* dense_attr = call->attrs.as<DenseAttrs>();
    } else {
      LOG(FATAL) << "CBLAS expects a single convolution or dense op.";
    }

    int ret = std::system(
        "g++ -O2 -Wall -std=c++11 -shared -fPIC -I/opt/intel/mkl/include utils.cc "
        "-L/opt/intel/mkl/lib/intel64 -o /tmp/util.so -ldl -lpthread -lm -lmkl_rt");
    if (!ret) {
      LOG(FATAL) << "Command failed";
    }

    Open("/tmp/subtract.so");
  }

 private:
  // Get the number of row of a ndarray.
  int CountRow(const runtime::NDArray& data) {
    const DLTensor* tensor = data.operator->();
    return tensor->shape[0];
  }

  // Get the number of columns of a ndarray.
  int CountColumn(const runtime::NDArray& data) {
    const DLTensor* tensor = data.operator->();
    return tensor->shape[1];
  }

  // Platform dependent handlers for opening system lib.
#if defined(_WIN32)
  // The handle.
  HMODULE handle_{nullptr};

  // Open the library
  void Open(const std::string& name) {
    std::wstring wname(name.begin(), name.end());
    handle_ = LoadLibraryW(wname.c_str());
    CHECK(handle_ != nullptr) << "Failed to open the dynamic shared library " << name;
  }

  // Retrieve a symbol.
  void* GetSymbol(const std::string& name) {
    return reinterpret_cast<void*>(GetProcAddress(handle_, (LPCSTR)name.c_str()));  // NOLINT(*)
  }

  // Close the handle.
  void Close() {
    FreeLibrary(handle_);
  }
#else
  // The handle.
  void* handle_{nullptr};

  // load the library
  void Open(const std::string& name) {
    handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle_ != nullptr) << "Failed to open the dynamic shared library " << name << " "
                              << dlerror();
  }

  // Retrieve a symbol.
  void* GetSymbol(const std::string& name) {
    return dlsym(handle_, name.c_str());
  }

  void Close() {
    dlclose(handle_);
  }
#endif
  CblasFloat func_s_;
};

runtime::Module CreateCblasModule(const Expr& expr) {
  std::shared_ptr<CblasModuleNode> n = std::make_shared<CblasModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 */
runtime::Module Compiler(const Expr& expr) {
  return CreateCblasModule(expr);
}

TVM_REGISTER_API("relay.ext.cblas")
.set_body_typed(Compiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
