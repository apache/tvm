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
 * \file src/relay/backend/contrib/contrib_codegen.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_

#include <tvm/relay/expr.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

class ExternCodegenBase {
 public:
  ExternCodegenBase() = default;

  /*!
   * \brief Create a runtime module for the external library. For example, it
   * could be a CSourceModule that can be directly compiled and linked together
   * with a DSOModule, or a json style module that emitts a json artifact that
   * is able to be executed by a customized json runtime.
   *
   * \param ref The subgraph Relay expression/module to be executed using extern ops.
   *
   * \return A runtime module.
   */
  virtual runtime::Module CreateExternModule(const NodeRef& ref) = 0;

  /*!
   * \brief Split the Relay function name to tokens.
   *
   * \param func The provided function.
   * \param prefix The prefix of the function name, i.e. dnnl.
   *
   * \return A vector of tokenized function name splitted by "_".
   */
  std::string GetSubgraphID(const Function& func, const std::string& prefix) const {
    const auto name_node =
        FunctionGetAttr(func, attr::kFuncName).as<tvm::ir::StringImm>();
    CHECK(name_node != nullptr) << "Fail to retrieve subgraph name.";
    std::string name = name_node->value;
    return GetSubgraphID(name, prefix);
  }

  /*!
   * \brief Split the encoded function name to tokens.
   *
   * \param the function name string.
   *
   * \return a vector of tokenized function name splitted by "_".
   */
  std::string GetSubgraphID(const std::string& name, const std::string& prefix) const {
    std::string temp = name;
    std::vector<std::string> tokens;
    std::string delimiter = "_";
    size_t pos = 0;
    std::string token;
    while ((pos = temp.find(delimiter)) != std::string::npos) {
      token = temp.substr(0, pos);
      tokens.push_back(token);
      temp.erase(0, pos + delimiter.length());
    }
    tokens.push_back(temp);

    CHECK(tokens.size() >= 2) << "Invalid subgraph name: " << name;
    CHECK(tokens[0] == prefix)
        << "Function name: " << name
        << " does not start with: " << prefix;
    return tokens[1];
  }
};

// A helper class to write the declaration of external functions.
class ExternSourcePrinter {
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
   * \brief Gerenate a wrapper for the subgraph that will use external codegen.
   *
   * \param func_name The name of wrapper function.
   * \param arg_cnt The expected number of arguments for the wrapper.
   *
   * \code
   *
   * // An example code for the wrapper.
   * extern "C" void foo(TVMValue* value, int* type_code, int nargs) {
   *   if (nargs != 3) {
   *     printf("foo expects 3 args, but received %d\n", nargs);
   *     return 1;
   *   }
   *
   *   DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
   *   DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
   *   DLTensor* out = static_cast<DLTensor*>(value[2].v_handle);
   *
   *   foo_(static_cast<float*>(arg0->data),
   *        static_cast<float*>(arg1->data),
   *        static_cast<float*>(out->data));
   *   return 0;
   * }
   *
   * \endcode
   */
  void GenerateSubgraphWrapper(const std::string& func_name, int arg_cnt) {
    // Print signature
    code_stream_ << "\n";
    code_stream_ << "extern \"C\" int " << func_name;
    code_stream_ << "(TVMValue* value, int* type_code, int nargs) {\n";
    EnterScope();
    // Print guard
    PrintIndents();
    code_stream_ << "if (nargs != " << arg_cnt << "){\n";
    EnterScope();
    PrintIndents();
    code_stream_ << "printf(\"" << func_name << " expects " << arg_cnt
                 << " arguments, but received %d\\n\", nargs);\n";
    PrintIndents();
    code_stream_ << "return 1;\n";
    ExitScope();
    PrintIndents();
    code_stream_ << "}\n";

    // According to TVM's calling convention, the last one is output.
    for (int i = 0; i < arg_cnt; i++) {
      PrintIndents();
      code_stream_ << "DLTensor* arg" << i << " = "
                   << "static_cast<DLTensor*>(value[" << i << "].v_handle);\n";
    }
    // Generate the call.
    PrintIndents();
    code_stream_ << func_name << "_(";
    for (int i = 0; i < arg_cnt - 1; i++) {
      code_stream_ << "static_cast<float*>(arg" << i << "->data), ";
    }
    if (arg_cnt > 0) {
      code_stream_ << "static_cast<float*>(arg" << arg_cnt - 1 << "->data)";
    }
    code_stream_ << ");\n\n";
    PrintIndents();
    code_stream_ << "return 0;\n";
    ExitScope();
    code_stream_ << "}";
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
      auto* val = ttype->shape[i].as<IntImm>();
      CHECK(val);
      shape.push_back(val->value);
    }
    return shape;
  }

  /*!
   * \briefa A common interface that that used by various external runtime to
   * generate the wrapper to invoke external kernels.
   *
   * \param subgraph_id The unique id of an external function. It will be used
   * during runtime to pick the correct external function.
   * \param args The arguments used by the external function.
   * \param buf_decl The declaration of temporary buffers that used to store the
   * intermeidate of each external kernel.
   * \param body The statements of the external function.
   * \param out The name and id pairs for output.
   *
   * \return The emitted code string.
   */
  std::string JitImpl(std::string subgraph_id,
                  std::vector<std::string> args,
                  std::vector<std::string> buf_decl,
                  std::vector<std::string> body,
                  std::vector<std::pair<std::string, int>> out) {
    // Create the signature. For example, it could be:
    // extern "C" void dnnl_0_(float* input0, float* input1, float* out, int M, int N) {}
    code_stream_ << "extern \"C\" void " << subgraph_id << "_(";

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

    // Create the wrapper to call the subgraph
    this->GenerateSubgraphWrapper(subgraph_id, args.size() + 1 /* output */);
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
#endif  // TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_
