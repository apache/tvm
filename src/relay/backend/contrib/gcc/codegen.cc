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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include "../contrib_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief An example codegen that is only used for quick prototyping and testing
 * purpose. Only several binary options are covered in the GCC builder. Users
 * may need to extend them to cover more operators.
 */
class GccBuilder : public ExprVisitor, public ExternSourcePrinter {
 public:
  explicit GccBuilder(const std::string& id) { this->subgraph_id_ = id; }

  void VisitExpr_(const VarNode* node) {
    subgraph_args_.push_back(node->name_hint());
    out_.clear();
    out_.push_back({node->name_hint(), 0});
  }

  void VisitExpr_(const CallNode* call) final {
    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    auto op_node = call->op.as<OpNode>();
    std::string func_name = subgraph_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    macro_stream << "GCC_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

    if (GetRef<Op>(op_node) == Op::Get("add")) {
      macro_stream << "+";
    } else if (GetRef<Op>(op_node) == Op::Get("subtract")) {
      macro_stream << "-";
    } else if (GetRef<Op>(op_node) == Op::Get("multiply")) {
      macro_stream << "*";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    auto in_shape = GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      macro_stream << ", " << in_shape[i];
    }
    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : out_) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.first;
      }
    }

    auto type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
        << "Only support single output tensor with float type";
    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    buf_stream << "float* " << out << " = (float*)std::malloc(4 * " << out_size << ");";
    buf_decl_.push_back(buf_stream.str());

    decl_stream << ", " << out << ");";
    subgraph_body.push_back(decl_stream.str());

    // Update output buffer
    out_.clear();
    out_.push_back({out, out_size});
  }

  std::string jit_csource() {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }

    // Write subgraph function declaration
    code_stream_ << "extern  \"C\" void " << subgraph_id_ << "_(";

    for (const auto& arg : subgraph_args_) {
      code_stream_ << "float* " << arg << ", ";
    }

    code_stream_ << "float* out) {\n";
    this->EnterScope();

    // Function body
    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : subgraph_body) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    // Copy output
    CHECK(out_.size() == 1) << "Internal error";
    this->PrintIndents();
    code_stream_ << "std::memcpy(out, " << out_[0].first << ", 4 * " << out_[0].second << ");\n";

    // Free buffers
    for (size_t i = 0; i < buf_decl_.size(); i++) {
      this->PrintIndents();
      code_stream_ << "std::free(buf_" << i << ");\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    // Create the wrapper to call the subgraph
    this->GenerateSubgraphWrapper(subgraph_id_,
                                  subgraph_args_.size() + 1 /* output */);
    return code_stream_.str();
  }

 private:
  std::string subgraph_id_ = "";
  int func_idx = 0;
  int buf_idx_ = 0;
  std::vector<std::string> subgraph_args_;
  std::vector<std::string> subgraph_body;
  std::vector<std::string> func_decl_;
  std::vector<std::string> buf_decl_;
  std::vector<std::pair<std::string, int>> out_;

  std::vector<int> GetShape(const Type& type) const {
    const auto* ttype = type.as<TensorTypeNode>();
    CHECK(ttype);
    std::vector<int> shape;
    for (size_t i = 0; i < ttype->shape.size(); ++i) {
      auto* val = ttype->shape[i].as<IntImm>();
      CHECK(val);
      shape.push_back(val->value);
    }
    return shape;
  }
};

class GccCodegen : public ExternCodegenBase {
 public:
  void CreateExternFunction(const Function& func) {
    CHECK(func.defined())
        << "Input error: external codegen expects a Relay function.";

    // Record subgraph ID for runtime invoke.
    auto sid = GetSubgraphID(func, "gcc");

    auto builder = GccBuilder("gcc_" + sid);
    builder.VisitExpr(func->body);
    code_stream_ << builder.jit_csource();
  }

  runtime::Module CreateExternModule(const NodeRef& ref) {
    // Create headers
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <iostream>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <stdio.h>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    // Append some common macro for operator definition.
    const char* operator_macro = R"op_marco(
    #define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          out[i] = a[i] p_OP_ b[i];                           \
        }                                                     \
      }
    
    #define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          for (int64_t j = 0; j < p_DIM2_; ++j) {             \
            int64_t k = i * p_DIM2_ + j;                      \
            out[k] = a[k] p_OP_ b[k];                         \
            std::cout << a[k] << "  " << b[k] << out[k] << std::endl;        \
          }                                                   \
        }                                                     \
      }
    )op_marco";

    code_stream_ << operator_macro << "\n\n";

    if (ref->IsInstance<FunctionNode>()) {
      CreateExternFunction(Downcast<Function>(ref));
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      for (const auto& it : mod->functions) {
        CreateExternFunction(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }
    LOG(INFO) << code_stream_.str();
    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external function";
    return (*pf)(code_stream_.str(), "cc");
  }

 private:
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM, so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module GccCompiler(const NodeRef& ref) {
  GccCodegen gcc;
  return gcc.CreateExternModule(ref);
}

TVM_REGISTER_API("relay.ext.gcc")
.set_body_typed(GccCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
