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
#include <tvm/relay/op.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

class CSourceModuleCodegenBase {
 public:
  CSourceModuleCodegenBase() = default;

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
    const auto name_node =
      FunctionGetAttr(func, attr::kExternalSymbol).as<tvm::tir::StringImmNode>();
    CHECK(name_node != nullptr) << "Fail to retrieve external symbol.";
    std::string ext_symbol = name_node->value;
    return ext_symbol;
  }
};

// The base class to generate the declaration functions in C.
class CodegenCBase {
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
   * \param arg_cnt The expected number of arguments.
   *
   * \code
   *
   * // An example code for the generated C function.
   * extern "C" void foo_wrapper_(DLTensor* arg0,
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
   * \endcode
   */
  void GenerateBackendCFunc(const std::string& func_name, int arg_cnt) {
    // Print signature
    code_stream_ << "\n";
    code_stream_ << "extern \"C\" int " << func_name << "_wrapper_(";
    for (int i = 0; i < arg_cnt - 1; i++) {
      code_stream_ << "DLTensor* arg" << i << ",\n";
      code_stream_ << "\t";
    }
    if (arg_cnt > 0) {
      code_stream_ << "DLTensor* arg" << arg_cnt - 1 << ") {\n";
    }

    EnterScope();

    // Generate the internal call.
    PrintIndents();
    code_stream_ << func_name << "_(";
    for (int i = 0; i < arg_cnt - 1; i++) {
      code_stream_ << "static_cast<float*>(arg" << i << "->data),\n";
      PrintIndents();
    }
    if (arg_cnt > 0) {
      code_stream_ << "static_cast<float*>(arg" << arg_cnt - 1 << "->data)";
    }
    code_stream_ << ");\n";
    PrintIndents();
    code_stream_ << "return 0;\n";
    ExitScope();
    code_stream_ << "}\n\n";

    // Generate the macro
    code_stream_ << "TVM_DLL_EXPORT_TYPED_FUNC(" << func_name << ", "
                 << func_name << "_wrapper_);\n\n";
  }

  /*!
   * \brief Emit the code for external runtime.
   *
   * \return The code string.
   */
  virtual std::string JIT() = 0;

  /*!
   * \brief Extract the shape from a Relay tensor type.
   *
   * \param type The provided type.
   *
   * \return The extracted shape in a list.
   */
  std::vector<int> GetShape(const Type& type) const {
    const auto* ttype = type.as<TensorTypeNode>();
    CHECK(ttype) << "Expect TensorTypeNode";
    std::vector<int> shape;
    for (size_t i = 0; i < ttype->shape.size(); ++i) {
      auto* val = ttype->shape[i].as<IntImmNode>();
      CHECK(val);
      shape.push_back(val->value);
    }
    return shape;
  }

  /*!
   * \brief Check if a call has the provided name.
   *
   * \param call A Relay call node.
   * \param op_name The name of the expected call.
   *
   * \return true if the call's name is equivalent to the given name. Otherwise,
   * false.
   */
  bool IsOp(const CallNode* call, std::string op_name) const {
    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "Expects a single op.";
    Op op = GetRef<Op>(op_node);
    return op == Op::Get(op_name);
  }

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
  std::string JitImpl(std::string ext_func_id, std::vector<std::string> args,
                      std::vector<std::string> buf_decl, std::vector<std::string> body,
                      std::vector<std::pair<std::string, int>> out) {
    // Create the signature. For example, it could be:
    // extern "C" void dnnl_0_(float* input0, float* input1, float* out, int M, int N) {}
    code_stream_ << "extern \"C\" void " << ext_func_id << "_(";

    for (const auto& arg : args) {
      code_stream_ << "float* " << arg << ", ";
    }
    code_stream_ << "float* out) {\n";
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
    CHECK_EQ(out.size(), 1U) << "Internal error: only single output is support.";
    this->PrintIndents();
    code_stream_ << "std::memcpy(out, " << out[0].first << ", 4 * " << out[0].second << ");\n";

    // Free buffers
    for (size_t i = 0; i < buf_decl.size(); i++) {
      this->PrintIndents();
      code_stream_ << "std::free(buf_" << i << ");\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    // Create the wrapper to call the ext_func
    this->GenerateBackendCFunc(ext_func_id, args.size() + 1 /* output */);
    return code_stream_.str();
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
