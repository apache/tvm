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
 * \file src/relay/backend/contrib/dnnl/codegen.cc
 * \brief Implementation of DNNL codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>

#include <random>
#include <fstream>
#include <sstream>
#include <streambuf>

#include "../../../../runtime/contrib/dnnl/dnnl.h"
#include "../contrib_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

// FIXME: This is an experimental implementation. We should implement all utilities
// and make a base class such as ExternBuilder for users to implement.
class DnnlBuilder : public ExprVisitor {
 public:
  explicit DnnlBuilder(const std::string& id) { this->subgraph_id_ = id; }

  void VisitExpr_(const VarNode* node) final {
    subgraph_args_.push_back(node->name_hint());
    out_.clear();
    out_.push_back({node->name_hint(), 0});
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    // Do nothing
  }

  void VisitExpr_(const CallNode* call) final {
    std::string func_name = subgraph_id_ + "_" + std::to_string(func_idx_++);

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
    for (size_t i = 0; i < args.size(); ++i) {
      decl += ", " + args[i];
    }
    decl += ");";
    func_decl_.push_back(decl);

    // Make function call when visiting arguments
    bool first = true;
    std::string func_call = func_name + "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : out_) {
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
    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    std::string buf_decl =
        "float* " + out + " = (float*)malloc(4 * " + std::to_string(out_size) + ");";
    buf_decl_.push_back(buf_decl);

    func_call += ", " + out + ");";
    subgraph_body.push_back(func_call);

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
    code += "extern \"C\" void " + subgraph_id_ + "(DnnlPackedArgs args, float* out) {\n";

    // Unpack inputs
    for (size_t i = 0; i < subgraph_args_.size(); ++i) {
      code +=
          "  float* " + subgraph_args_[i] + " = (float*) args.data[" + std::to_string(i) + "];\n";
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
  int func_idx_ = 0;
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

  bool IsOp(const CallNode* call, std::string op_name) {
    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "Expects a single op.";
    Op op = GetRef<Op>(op_node);
    return op == Op::Get(op_name);
  }
};

class DNNLCodegen : public ExternCodegenBase {
 public:
  std::string GetLibPath() const {
    return lib_path_;
  }

  void CreateExternSignature(const Function& func, bool update) {
    CHECK(func.defined())
        << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op";

    // Record subgraph ID for runtime invoke.
    auto sid = GetSubgraphID(func);

    if (update) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<uint64_t> distr;
      std::stringstream ss;
      ss << std::hex << distr(gen);
      std::ifstream lib_file("src/relay/backend/contrib/dnnl/libs.cc");
      code_.assign((std::istreambuf_iterator<char>(lib_file)),
                    std::istreambuf_iterator<char>());
      lib_path_ = "/tmp/relay_dnnl_lib_" + ss.str() + ".so";
    }

    auto builder = DnnlBuilder(runtime::contrib::kDnnlPrefix + sid);
    builder.VisitExpr(func->body);
    std::string code = builder.build();
    code_ = code_ + code;
  }

  void CompileExternLib() override {
    std::string code = "echo \'" + code_ + "\'";
    std::string cmd = "g++ -O2 -Wall -std=c++11 -shared -fPIC -xc++ - -o " + lib_path_ +
                      " -ldl -lpthread -lm -ldnnl";
    cmd = code + " | " + cmd;
    int ret = std::system(cmd.c_str());
    if (ret < 0) {
      LOG(FATAL) << "Failed to compile DNNL library. Error code: " << ret;
    }
  }

  void Build(const NodeRef& ref) override {
    if (ref->IsInstance<FunctionNode>()) {
      CreateExternSignature(Downcast<Function>(ref), true);
      CompileExternLib();
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      bool update = true;
      for (const auto& it : mod->functions) {
        CreateExternSignature(Downcast<Function>(it.second), update);
        update = false;
      }
      CompileExternLib();
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }
  }

 private:
  std::string code_;
  std::string lib_path_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const NodeRef& ref) {
  DNNLCodegen dnnl;
  dnnl.Build(ref);
  std::shared_ptr<runtime::contrib::DNNLModule> n =
      std::make_shared<runtime::contrib::DNNLModule>(dnnl.GetLibPath());
  return runtime::Module(n);
}

TVM_REGISTER_API("relay.ext.dnnl")
.set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
