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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include "test_external_library.h"

namespace tvm {
namespace relay {

typedef void (*sub)(ExternalTensor a, ExternalTensor b, ExternalTensor* out);

class ExternalModuleNode : public runtime:: ModuleNode {
 public:
  ExternalModuleNode() = default;
  ~ExternalModuleNode() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
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
    if (name == "Subtract") {
      CHECK(handle_) << "You need to build the external module first";
      func_s_ = reinterpret_cast<sub>(dlsym(handle_,"Subtract"));
      char* error = dlerror();
      if (error != NULL) {
        LOG(FATAL) << error;
        return PackedFunc();
      }

      return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3U);
        runtime::NDArray a = args[0];
        ExternalTensor lhs;
        lhs.data = a->data;
        lhs.ndim = a.Shape().size();
        // lhs.shape = a.Shape().data();
        lhs.shape = new int64_t[lhs.ndim];

        runtime::NDArray b = args[1];
        ExternalTensor rhs;
        rhs.data = b->data;
        rhs.ndim = b.Shape().size();
        rhs.shape = new int64_t[rhs.ndim];
        // rhs.shape = b.Shape().data();

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
    } else {
      LOG(FATAL) << "Unknow function found when invoking extern library: " << name;
      return PackedFunc();
    }
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
    return "ExternalModule";
  }

  void Build() {
    std::system(
        "g++ -std=c++11 -shared -fPIC -ldl src/relay/backend/test_external_library.cc -o /tmp/subtract.so");
    handle_ = dlopen("/tmp/subtract.so", RTLD_LAZY);
    if (!handle_) {
      LOG(FATAL) << "Cannot open library: " << dlerror() << '\n';
    }
  }

 private:
  void* handle_{nullptr};
  sub func_s_;
};

runtime::Module CreateExternalModule() {
  std::shared_ptr<ExternalModuleNode> n = std::make_shared<ExternalModuleNode>();
  n->Build();
  return runtime::Module(n);
}

}  // namespace relay
}  // namespace tvm

namespace tvm {
namespace relay {

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
runtime::Module Compiler(const Expr& expr) {
  Function func = Downcast<Function>(expr);
  CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
  return CreateExternalModule();
}

TVM_REGISTER_API("relay.ext.gcc")
.set_body_typed(Compiler);

}  // namespace relay
}  // namespace tvm
