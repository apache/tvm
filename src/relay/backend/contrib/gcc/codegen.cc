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

#include <random>
#include <fstream>
#include <sstream>
#include <streambuf>

#include "../contrib_codegen.h"
#include "../../../../runtime/contrib/gcc/gcc.h"

namespace tvm {
namespace relay {
namespace contrib {

// FIXME: This is an experimental implementation. We should implement all utilities
// and make a base claaa such as ExternBuilder for users to implement.
class GccBuilder : public ExprVisitor {
 public:
  explicit GccBuilder(const std::string& id) { this->subgraph_id_ = id; }

  void VisitExpr_(const VarNode* node) {
    subgraph_args_.push_back(node->name_hint());
    out_.clear();
    out_.push_back({node->name_hint(), 0});
  }

  void VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    std::string func_name = subgraph_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    std::string decl = "GCC_BINARY_OP_" + std::to_string(call->args.size()) +
                       "D(" + func_name + ", ";

    if (GetRef<Op>(op_node) == Op::Get("add")) {
      decl += "+";
    } else if (GetRef<Op>(op_node) == Op::Get("subtract")) {
      decl += "-";
    } else if (GetRef<Op>(op_node) == Op::Get("multiply")) {
      decl += "*";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    auto in_shape = GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      decl += ", " + std::to_string(in_shape[i]);
    }
    decl += ");";
    func_decl_.push_back(decl);

    // Make function call when visiting arguments
    bool first = true;
    std::string gcc_call = func_name + "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : out_) {
        if (!first) {
          gcc_call += ", ";
        }
        first = false;
        gcc_call += out.first;
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
    std::string buf_decl =
        "float* " + out + " = (float*)malloc(4 * " + std::to_string(out_size) + ");";
    buf_decl_.push_back(buf_decl);

    gcc_call += ", " + out + ");";
    subgraph_body.push_back(gcc_call);

    // Update output buffer
    out_.clear();
    out_.push_back({out, out_size});
  }

  std::string build() {
    std::string code = "";

    // Write function macros
    for (auto decl : func_decl_) {
      code += decl + "\n";
    }

    // Write subgraph function declaration
    code += "extern  \"C\" void " + subgraph_id_ + "(GccPackedArgs args, float* out) {\n";

    // Unpack inputs
    for (size_t i = 0; i < subgraph_args_.size(); ++i) {
      code += "  float* " + subgraph_args_[i] + " = args.data[" + std::to_string(i) + "];\n";
    }
    // Function body
    for (auto decl : buf_decl_) {
      code += "  " + decl + "\n";
    }
    for (auto stmt : subgraph_body) {
      code += "  " + stmt + "\n";
    }

    // Copy output
    CHECK(out_.size() == 1) << "Internal error";
    code += "  memcpy(out, " + out_[0].first + ", 4 *" + std::to_string(out_[0].second) + ");\n";

    code += "}\n";
    return code;
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
  std::string GetLibPath() const {
    return lib_path_;
  }

  void CreateExternSignature(const Function& func, bool update) {
    CHECK(func.defined())
        << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "Unknown error";  // comaniac: Don't know in what case this will fail.

    // Record subgraph ID for runtime invoke.
    auto sid = GetSubgraphID(func);

    // Prepare library source
    if (update) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<uint64_t> distr;
      std::stringstream ss;
      ss << std::hex << distr(gen);
      std::ifstream lib_file("src/relay/backend/contrib/gcc/libs.cc");
      code_.assign((std::istreambuf_iterator<char>(lib_file)),
                    std::istreambuf_iterator<char>());
      lib_path_ = "/tmp/relay_gcc_lib_" + ss.str() + ".so";
    }

    auto builder = GccBuilder(runtime::contrib::kGccPrefix + sid);
    builder.VisitExpr(func->body);
    std::string code = builder.build();

    // Append the signature.
    code_ = code_ + code;
  }

  void CompileExternLib() override {
    // Compile from pipe and generate the library.
    std::string code = "echo \'" + code_ + "\'";
    std::string cmd = "g++ -std=c++11 -shared -fPIC -ldl -o " + lib_path_ + " -xc++ -";
    cmd = code + " | " + cmd;

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile GCC library. Error code: " << ret;
    }
  }

  void Build(const NodeRef& ref) override {
    if (ref->IsInstance<FunctionNode>()) {
      CreateExternSignature(Downcast<Function>(ref), true);
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      bool update = true;
      for (const auto& it : mod->functions) {
        CreateExternSignature(Downcast<Function>(it.second), update);
        update = false;
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }
    CompileExternLib();
  }

 private:
  std::string code_;
  std::string lib_path_;
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
  gcc.Build(ref);
  std::shared_ptr<runtime::contrib::GccModule> n =
    std::make_shared<runtime::contrib::GccModule>(gcc.GetLibPath());
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.gcc")
.set_body_typed(GccCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
