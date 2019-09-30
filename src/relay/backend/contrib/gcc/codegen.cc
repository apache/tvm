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
#include <random>
#include <sstream>
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

typedef void (*GccSubgraphFunc)(GccPackedArgs in, float* out);

// FIXME: This is an experimental implementation. We should implement all utilities
// and make a base claaa such as ExternBuilder for users to implement.
class GccBuilder : public ExprVisitor {
 public:
  GccBuilder(const std::string& id) { this->_subgraph_id = id; }

  void VisitExpr_(const VarNode* node) {
    _subgraph_args.push_back(node->name_hint());
    _out.clear();
    _out.push_back({node->name_hint(), 0});
  }
 
  void VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    std::string func_name = _subgraph_id + "_" + std::to_string(_func_idx++);

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
    for (int i = 0; i < in_shape.size(); ++i) {
      decl += ", " + std::to_string(in_shape[i]);
    }
    decl += ");";
    _func_decl.push_back(decl);

    // Make function call when visiting arguments
    bool first = true;
    std::string gcc_call = func_name + "(";
    for (int i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : _out) {
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
    std::string out = "buf_" + std::to_string(_buf_idx++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (int i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    std::string buf_decl =
        "float* " + out + " = (float*)malloc(4 * " + std::to_string(out_size) + ");";
    _buf_decl.push_back(buf_decl);

    gcc_call += ", " + out + ");";
    _subgraph_body.push_back(gcc_call);

    // Update output buffer
    _out.clear();
    _out.push_back({out, out_size});
  }

  std::string build() {
    std::string code = "";

    // Write function macros
    for (auto decl : _func_decl) {
      code += decl + "\n";
    }

    // Write subgraph function declaration
    code += "extern \\\"C\\\" void " + _subgraph_id + "(GccPackedArgs args, float* out) {\n";

    // Unpack inputs
    for (int i = 0; i < _subgraph_args.size(); ++i) {
      code += "  float* " + _subgraph_args[i] + " = args.data[" + std::to_string(i) + "];\n";
    }
    // Function body
    for (auto decl : _buf_decl) {
      code += "  " + decl + "\n";
    }
    for (auto stmt : _subgraph_body) {
      code += "  " + stmt + "\n";
    }

    // Copy output
    CHECK(_out.size() == 1) << "Internal error";
    code += "  memcpy(out, " + _out[0].first + ", 4 *" + std::to_string(_out[0].second) + ");\n";

    code += "}\n";
    return code;
  }

 private:
  std::string _subgraph_id = "";
  int _func_idx = 0;
  int _buf_idx = 0;
  std::vector<std::string> _subgraph_args;
  std::vector<std::string> _subgraph_body;
  std::vector<std::string> _func_decl;
  std::vector<std::string> _buf_decl;
  std::vector<std::pair<std::string, int>> _out;

  std::vector<int> GetShape(const Type& type) const {
    const auto* ttype = type.as<TensorTypeNode>();
    CHECK(ttype);
    std::vector<int> _shape;
    for (int i = 0; i < ttype->shape.size(); ++i) {
      auto* val = ttype->shape[i].as<IntImm>();
      CHECK(val);
      _shape.push_back(val->value);
    }
    return _shape;
  }
};

class GccModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(const std::string& id = "") const override {
    CHECK_GT(src_lib_path_.count(id), 0U);
    const auto& pair = src_lib_path_.at(id);
    return {pair.second};
  }

  const std::string GetPrefix() const override {
    return "gcc_";
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

  runtime::PackedFunc GetFunction(const std::string& name,
                                  const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    std::string curr_id = GetSubgraphID(name);
    if (!IsOpen()) {
      Open(this->GetExternLibPaths(curr_id));
    }
    CHECK(IsOpen()) << "The external module has not been built or failed to open.\n";
    // Generate an external packed function
    return PackedFunc([sptr_to_self, curr_id, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      const DLTensor* dptr = ((runtime::NDArray)args[0]).operator->();
      runtime::NDArray out_arg = args[args.size() - 1];
      auto out = reinterpret_cast<float*>(out_arg->data);

      // Get function from the library
      std::string encoded_name = GetPrefix() + curr_id;
      auto func_s = reinterpret_cast<GccSubgraphFunc>(GetSymbol(encoded_name));

      // Reinterpret data and function to the right type and invoke
      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        GccPackedArgs packed_args;
        packed_args.data = (float**)malloc(sizeof(float*) * args.size());
        for (int i = 0; i < args.size() - 1; ++i) {
          runtime::NDArray arg = args[i];
          packed_args.data[i] = reinterpret_cast<float*>(arg->data);
        }
        (*func_s)(packed_args, out);
      } else {
        LOG(FATAL) << "Only support float32 type.";
      }
      *rv = out;
    });
  }

  void CreateExternSignature (const Function& func, bool update) {
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
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
      src_path_ = "/tmp/relay_gcc_lib_" + ss.str() + ".cc";
      lib_path_ = "/tmp/relay_gcc_lib_" + ss.str() + ".so";
      std::string cmd = "cp src/relay/backend/contrib/gcc/libs.cc " + src_path_;
      std::system(cmd.c_str());
      std::system("cp src/relay/backend/contrib/gcc/libs.h /tmp/");
    }
    // Save the src and lib path.
    src_lib_path_.emplace(sid, std::make_pair(src_path_, lib_path_));

    auto builder = GccBuilder(GetPrefix() + sid);
    builder.VisitExpr(func->body);
    std::string code = builder.build();

    // Append the signature.
    auto cmd = "echo \"" + code + "\" >> " + src_path_;
    std::system(cmd.c_str());
  }

  void CompileExternLib() override {
    std::string cmd =
        "g++ -std=c++11 -shared -fPIC -ldl " + src_path_ + " -o " + lib_path_;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile GCC library. Error code: " << ret;
    }
  }

  void Build(const NodeRef& ref) override {
    if (ref->derived_from<FunctionNode>()) {
      CreateExternSignature(Downcast<Function>(ref), true);
      CompileExternLib();
    } else if (ref->derived_from<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      bool update = true;
      for (const auto& it : mod->functions) {
        CreateExternSignature(Downcast<Function>(it.second), update);
        update = false;
      }
      CompileExternLib();
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module" << "\n";
    }
  }

 private:
  std::string src_path_;
  std::string lib_path_;
  std::unordered_map<std::string, std::pair<std::string, std::string> > src_lib_path_;
};


/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module GccCompiler(const NodeRef& ref) {
  std::shared_ptr<GccModuleNode> n = std::make_shared<GccModuleNode>();
  n->Build(ref);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.gcc")
.set_body_typed(GccCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
