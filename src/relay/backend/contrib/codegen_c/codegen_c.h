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
 * \file src/relay/backend/contrib/codegen_c/codegen_c.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CODEGEN_C_CODEGEN_C_H_
#define TVM_RELAY_BACKEND_CONTRIB_CODEGEN_C_CODEGEN_C_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/container.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

struct Output {
  std::string name;
  std::string dtype;
  int size;
  bool need_copy;
};

class CSourceModuleCodegenBase {
 public:
  CSourceModuleCodegenBase() = default;
  virtual ~CSourceModuleCodegenBase() = default;

  /*!
   * \brief Create a runtime module for the external library. For example, it
   * could be a CSourceModule that can be directly compiled and linked together
   * with a DSOModule, or a json style module that emitts a json artifact that
   * is able to be executed by a customized json runtime.
   *
   * \param ref The ext_func Relay expression/module to be executed using extern ops.
   *
   * \return A runtime module.
   */
  virtual runtime::Module CreateCSourceModule(const ObjectRef& ref) = 0;

  /*!
   * \brief Get the external symbol of the Relay function name.
   *
   * \param func The provided function.
   *
   * \return An external symbol.
   */
  std::string GetExtSymbol(const Function& func) const {
    const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Fail to retrieve external symbol.";
    return std::string(name_node.value());
  }
};

// The base class to generate the declaration functions in C.
class CodegenCBase {
 public:
  virtual ~CodegenCBase() {}

 protected:
  /*! \brief Print indents using spaces. */
  void PrintIndents() {
    for (int i = 0; i < indent_; i++) {
      code_stream_ << ' ';
    }
  }

  /*!
   * \brief Enter a new scope.
   */
  void EnterScope() { indent_ += 2; }

  /*!
   * \brief Exit a scope.
   */
  void ExitScope() {
    CHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  /*!
   * \brief Gerenate C code for the external function.
   *
   * \param func_name The name of the external function.
   * \param args arguments to the external function.
   *
   * \code
   *
   * Array<NDArray> foo_consts;
   *
   * // An example code for the generated C function.
   * extern "C" int foo_wrapper_(DLTensor* arg0,
   *                              DLTensor* arg1,
   *                              DLTensor* out) {
   *   foo_(static_cast<float*>(arg0->data),
   *        static_cast<float*>(arg1->data),
   *        static_cast<float*>(out->data));
   *   return 0;
   * }
   *
   * TVM_DLL_EXPORT_TYPED_FUNC(foo, foo_wrapper_);
   *
   * int foo_init_wrapper_(Array<NDArray> arr) {
   *   foo_consts = arr;
   *   return 0;
   * }
   *
   * TVM_DLL_EXPORT_TYPED_FUNC(__init_foo, foo_init_wrapper_);
   *
   * \endcode
   */
  void GenerateBackendCFunc(const std::string& func_name, const Array<Var>& args,
                            const std::string& const_arr_name, const std::vector<Output>& outs) {
    // Print signature
    code_stream_ << "\n";
    code_stream_ << "extern \"C\" int " << func_name << "_wrapper_(";
    for (size_t i = 0; i < args.size(); i++) {
      code_stream_ << "DLTensor* arg" << i << ",\n";
      code_stream_ << "\t";
    }
    for (size_t i = 0; i < outs.size() - 1; i++) {
      code_stream_ << "DLTensor* out" << i << ",\n";
      code_stream_ << "\t";
    }
    code_stream_ << "DLTensor* out" << outs.size() - 1 << ") {\n";

    EnterScope();

    // Generate the internal call.
    PrintIndents();
    code_stream_ << func_name << "_(";
    for (size_t i = 0; i < args.size(); i++) {
      const auto& dtype_str = GetDtypeString(args[i]);
      code_stream_ << "static_cast<" << dtype_str << "*>(arg" << i << "->data),\n";
      PrintIndents();
    }
    for (size_t i = 0; i < outs.size() - 1; i++) {
      code_stream_ << "static_cast<" << outs[i].dtype << "*>(out" << i << "->data),\n";
      PrintIndents();
    }
    code_stream_ << "static_cast<" << outs.back().dtype << "*>(out" << outs.size() - 1
                 << "->data));\n";
    PrintIndents();
    code_stream_ << "return 0;\n";
    ExitScope();
    code_stream_ << "}\n\n";

    // Generate the macro
    code_stream_ << "TVM_DLL_EXPORT_TYPED_FUNC(" << func_name << ", " << func_name
                 << "_wrapper_);\n\n";

    if (!const_arr_name.empty()) {
      code_stream_ << "int " << func_name << "_init_wrapper_(Array<NDArray> arr) {\n";
      EnterScope();
      PrintIndents();
      code_stream_ << func_name << "_consts = arr;\n";
      code_stream_ << "return 0;\n";
      ExitScope();
      code_stream_ << "}\n\n";
      code_stream_ << "TVM_DLL_EXPORT_TYPED_FUNC(__init_" << func_name << ", " << func_name
                   << "_init_wrapper_);\n\n";
    }
  }

  /*!
   * \brief Emit the code for external runtime.
   *
   * \param out The outputs.
   *
   * \return The code string.
   */
  virtual std::string JIT(const std::vector<Output>& out) = 0;

  /*!
   * \brief A common interface that is used by various external runtime to
   * generate the wrapper to invoke external kernels.
   *
   * \param ext_func_id The unique id of an external function. It will be used
   * during runtime to pick the correct external function.
   * \param args The arguments used by the external function.
   * \param buf_decl The declaration of temporary buffers that used to store the
   * intermeidate of each external kernel.
   * \param body The statements of the external function.
   * \param out The name and id pairs for output.
   *
   * \return The emitted code string.
   */
  std::string JitImpl(const std::string& ext_func_id, const Array<Var>& args,
                      const std::vector<std::string>& buf_decl,
                      const std::vector<std::string>& body, const std::string& const_arr_name,
                      const std::vector<Output>& outs) {
    // Create a declaration for global ndarrays that contain constant data.
    if (!const_arr_name.empty()) {
      code_stream_ << const_arr_name << "\n\n";
    }
    // Create the signature. For example, it could be:
    // extern "C" void dnnl_0_(float* in0, float* in1, float* out0, float* out1) {}
    code_stream_ << "extern \"C\" void " << ext_func_id << "_(";

    for (const auto& arg : args) {
      const auto& dtype_str = GetDtypeString(arg);
      code_stream_ << dtype_str << "* " << arg->name_hint() << ", ";
    }
    for (size_t i = 0; i < outs.size() - 1; ++i) {
      code_stream_ << outs[i].dtype << "* out" << i << ", ";
    }
    code_stream_ << outs.back().dtype << "* out" << outs.size() - 1 << ") {\n";
    this->EnterScope();

    // Function body
    for (auto decl : buf_decl) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : body) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    // Copy output
    for (size_t i = 0; i < outs.size(); ++i) {
      if (!outs[i].need_copy) {
        continue;
      }
      this->PrintIndents();
      code_stream_ << "std::memcpy(out" << i << ", " << outs[i].name << ", 4 * " << outs[i].size
                   << ");\n";
    }

    // Free buffers
    for (size_t i = 0; i < buf_decl.size(); i++) {
      this->PrintIndents();
      code_stream_ << "std::free(buf_" << i << ");\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    // Create the wrapper to call the ext_func
    this->GenerateBackendCFunc(ext_func_id, args, const_arr_name, outs);
    return code_stream_.str();
  }

  /*!
   * \brief Returns dtype string
   *
   * \param var Var to get the dtype of
   *
   * \return The dtype string.
   */
  std::string GetDtypeString(const Var& var) {
    auto ttype = var->checked_type().as<TensorTypeNode>();
    CHECK(ttype) << "Expect TensorTypeNode";
    return GetDtypeString(ttype);
  }

  /*!
   * \brief Returns dtype string
   *
   * \param ttype TensorTypeNode* to get the dtype of
   *
   * \return The dtype string.
   */
  std::string GetDtypeString(const TensorTypeNode* ttype) {
    std::string dtype;
    if (runtime::TypeMatch(ttype->dtype, kDLFloat, 32)) {
      dtype = "float";
    } else if (runtime::TypeMatch(ttype->dtype, kDLInt, 32)) {
      dtype = "int";
    } else if (runtime::TypeMatch(ttype->dtype, kDLInt, 64)) {
      dtype = "int64_t";
    } else {
      LOG(FATAL) << "Unsupported dtype " << ttype->dtype;
    }

    return dtype;
  }

  /*!
   * \brief Creates a checker to check if the NDArray pool is initialized
   *
   * \param symobl The Symbol of the current function
   *
   * \return The created checker
   */
  std::string CreateInitChecker(const std::string& symbol) const {
    std::ostringstream oss;
    oss << "CHECK(!" << symbol
        << "_consts.empty()) << \"C source module hasn't been initialized.\";\n";
    return oss.str();
  }

  /*!
   * \brief Generates the global ndarray pool declaration
   *
   * \param symobl The Symbol of the current function
   *
   * \return The created declaration
   */
  std::string CreateNDArrayPool(const std::string& symbol) const {
    return "Array<NDArray> " + symbol + "_consts;";
  }

  /*!
   * \brief Generates the reference to the data of a constant ndarray
   *
   * \param symobl The Symbol of the current function
   * \param symobl const_id The index of the constant
   *
   * \return The created reference
   */
  std::string CreateDataReference(const std::string& symbol, int const_id) const {
    return "static_cast<float*>(" + symbol + "_consts[" + std::to_string(const_id) + "]->data)";
  }

  /*!
   * \brief Returns the variable name for a constant variable
   *
   * \param symobl The Symbol of the current function
   * \param symobl const_id The index of the constant
   *
   * \return The created variable name
   */
  std::string CreateConstVar(const std::string& symbol, int const_id) const {
    return symbol + "_const_" + std::to_string(const_id++);
  }

  /*! \brief The external function source code stream. */
  std::ostringstream code_stream_;

 private:
  /*! \brief Indent of the source code. */
  int indent_{0};
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CODEGEN_C_CODEGEN_C_H_
