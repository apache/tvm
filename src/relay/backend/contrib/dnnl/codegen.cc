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

#include "libs.h"

namespace tvm {
namespace relay {
namespace contrib {

typedef void (*DnnlSubgraphFunc)(DnnlPackedArgs in, float* out);

// FIXME: This is an experimental implementation. We should implement all utilities
// and make a base claaa such as ExternBuilder for users to implement.
class DnnlBuilder : public ExprVisitor {
 public:
  DnnlBuilder(std::string id) { this->_subgraph_id = id; }

  void VisitExpr_(const VarNode* node) final {
    _subgraph_args.push_back(node->name_hint());
    _out.clear();
    _out.push_back({node->name_hint(), 0});
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ; // Do nothing
  }

  void VisitExpr_(const CallNode* call) final {
    std::string func_name = _subgraph_id + "_" + std::to_string(_func_idx++);

    // Make function declaration
    std::string decl = "";

    // Args: ID
    std::string macro = "";
    std::vector<std::string> args;

    if (IsOp(call, "nn.conv2d")) {
      macro = "CONV2D";
      const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();

      auto ishape = GetShape(call->args[0]->checked_type());
      auto wshape = GetShape(call->args[1]->checked_type());

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }

      // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
      args.push_back(std::to_string(wshape[0]));
      args.push_back(std::to_string(conv2d_attr->groups));
      args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImm>()->value));
      args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImm>()->value));
      args.push_back(std::to_string(wshape[2]));
      args.push_back(std::to_string(wshape[3]));
      args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImm>()->value));
      args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImm>()->value));
    } else if (IsOp(call, "nn.dense")) {
      macro = "DENSE";
      auto ishape = GetShape(call->args[0]->checked_type());
      auto wshape = GetShape(call->args[1]->checked_type());

      // Args: N, C, O
      args.push_back(std::to_string(ishape[0]));
      args.push_back(std::to_string(ishape[1]));
      args.push_back(std::to_string(wshape[0]));

    } else if (IsOp(call, "nn.relu")) {
      macro = "RELU";
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }
    } else if (IsOp(call, "nn.batch_norm")) {
      macro = "BN";
      const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }

      // Args: epilson
      args.push_back(std::to_string(bn_attr->epsilon));
    } else if (IsOp(call, "add")) {
      macro = "ADD";
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }

    decl = macro + "(" + func_name;
    for (int i = 0; i < args.size(); ++i) {
      decl += ", " + args[i];
    }
    decl += ");";
    _func_decl.push_back(decl);

    // Make function call when visiting arguments
    bool first = true;
    std::string func_call = func_name + "(";
    for (int i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : _out) {
        if (!first) {
          func_call += ", ";
        }
        first = false;
        func_call += out.first;
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

    func_call += ", " + out + ");";
    _subgraph_body.push_back(func_call);

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
    code += "extern \\\"C\\\" void " + _subgraph_id + "(DnnlPackedArgs args, float* out) {\n";

    // Unpack inputs
    for (int i = 0; i < _subgraph_args.size(); ++i) {
      code += "float* " + _subgraph_args[i] + " = (float*) args.data[" + std::to_string(i) + "];\n";
    }
    // Function body
    for (auto decl : _buf_decl) {
      code += decl + "\n";
    }
    for (auto stmt : _subgraph_body) {
      code += stmt + "\n";
    }

    // Copy output
    CHECK(_out.size() == 1) << "Internal error";
    code += "memcpy(out, " + _out[0].first + ", 4 *" + std::to_string(_out[0].second) + ");\n";

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

  bool IsOp(const CallNode* call, std::string op_name) {
    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "Expects a single op.";
    Op op = GetRef<Op>(op_node);
    return op == Op::Get(op_name);
  }
};

class DNNLModuleNode : public ExternModuleNodeBase {
 public:
  const std::vector<std::string> GetExternLibPaths(std::string id) const override {
    return {"/tmp/relay_dnnl_lib_" + id + ".so"};
  }

  const std::string GetPrefix() const override {
    return "dnnl_";
  }

  /*!
   * \brief Get the source code of the external module.
   *
   * \param format The format of the source code.
   *
   * \return The source code of the external library module in the text form.
   */
  TVM_DLL std::string GetSource(const std::string& format = "") override { return ""; }

  const char* type_key() const override { return "DNNLModule"; }

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  runtime::PackedFunc GetFunction(const std::string& name,
                                  const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    curr_id_ = GetSubgraphID(name);
    Open(this->GetExternLibPaths(curr_id_));
    CHECK(handle_) << "The external module has not been built or failed to open.\n";

    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      const DLTensor* dptr = ((runtime::NDArray)args[0]).operator->();
      runtime::NDArray out_arg = args[args.size() - 1];
      auto out = reinterpret_cast<float*>(out_arg->data);

      // Get function from the library
      std::string encoded_name = GetPrefix() + curr_id_;
      auto func_s = reinterpret_cast<DnnlSubgraphFunc>(GetSymbol(encoded_name));

      // Reinterpret data and function to the right type and invoke
      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        DnnlPackedArgs packed_args;
        packed_args.data = (void**)malloc(sizeof(float*) * args.size());
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

  void Build(const Expr& expr) override {
    Function func = Downcast<Function>(expr);
    CHECK(func.defined()) << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op";

    // Record subgraph ID for runtime invoke.
    auto id = GetSubgraphID(func);
    auto builder = DnnlBuilder(GetPrefix() + id);
    builder.VisitExpr(func->body);
    std::string code = builder.build();

    // Prepare library source
    // FIXME: Now we compile N libraries for N subgraphs, but we should merge them to one.
    std::string lib_src_name = "/tmp/relay_dnnl_lib_" + id + ".cc";
    std::string lib_name = "/tmp/relay_dnnl_lib_" + id + ".so";
    std::string cmd = "cp src/relay/backend/contrib/dnnl/libs.cc " + lib_src_name;
    std::system(cmd.c_str());
    std::system("cp src/relay/backend/contrib/dnnl/libs.h /tmp/");

    cmd = "echo \"" + code + "\" >> " + lib_src_name;
    std::system(cmd.c_str());

    cmd = "g++ -O2 -Wall -std=c++11 -shared -fPIC " + lib_src_name + " -o " + lib_name +
          " -ldl -lpthread -lm -ldnnl";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      LOG(FATAL) << "Failed to compile DNNL library. Error code: " << ret;
    }
  }

 private:
  std::string curr_id_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const Expr& expr) {
  std::shared_ptr<DNNLModuleNode> n = std::make_shared<DNNLModuleNode>();
  n->Build(expr);
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.dnnl").set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
