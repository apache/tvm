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
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>

#include "../contrib_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

// TODO(@zhiics, @comaniac): This is basic implementation. We should implement
// all utilities and make a base class for users to implement.
class DnnlBuilder : public ExprVisitor, public ExternSourcePrinter {
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
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;
    // Args: ID
    std::vector<std::string> args;

    if (IsOp(call, "nn.conv2d")) {
      decl_stream << "dnnl_conv2d";
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
      decl_stream << "dnnl_dense";
      auto ishape = GetShape(call->args[0]->checked_type());
      auto wshape = GetShape(call->args[1]->checked_type());

      // Args: N, C, O
      args.push_back(std::to_string(ishape[0]));
      args.push_back(std::to_string(ishape[1]));
      args.push_back(std::to_string(wshape[0]));

    } else if (IsOp(call, "nn.relu")) {
      decl_stream << "dnnl_relu";
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }
    } else if (IsOp(call, "nn.batch_norm")) {
      decl_stream << "dnnl_bn";
      const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: N, C, H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }

      // Args: epsilon
      args.push_back(std::to_string(bn_attr->epsilon));
    } else if (IsOp(call, "add")) {
      decl_stream << "dnnl_add";
      auto ishape = GetShape(call->args[0]->checked_type());

      // Args: H, W
      for (auto s : ishape) {
        args.push_back(std::to_string(s));
      }
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }

    // Make function call with input buffers when visiting arguments
    bool first = true;
    decl_stream << "(";
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

    // Analyze the output buffer
    auto type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
        << "Only support single output tensor with float type";
    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    this->PrintIndents();
    buf_stream << "float* " << out << " = (float*)std::malloc(4 * " << out_size << ");";
    buf_decl_.push_back(buf_stream.str());
    decl_stream << ", " << out;

    // Attach attribute arguments
    for (size_t i = 0; i < args.size(); ++i) {
      decl_stream  << ", " << args[i];
    }
    decl_stream << ");";
    subgraph_body.push_back(decl_stream.str());

    // Update output buffer
    out_.clear();
    out_.push_back({out, out_size});
  }

  std::string JIT(void) {
    return JitImpl(subgraph_id_, subgraph_args_, buf_decl_, subgraph_body, out_);
  }

 private:
  /*! \brief The id of the external dnnl subgraph. */
  std::string subgraph_id_{""};
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped external function. */
  std::vector<std::string> subgraph_args_;
  /*! \brief statement of the external function. */
  std::vector<std::string> subgraph_body;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The name of the the outputs. */
  std::vector<std::pair<std::string, int>> out_;

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
};

/*!
 * \brief The DNNL codegen helper to generate wrapepr function calls of DNNL
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class DNNLCodegen : public ExternCodegenBase {
 public:
  // Create a corresponding external function for the given relay Function.
  void CreateExternFunction(const Function& func) {
    CHECK(func.defined())
        << "Input error: external codegen expects a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op";

    // Record subgraph ID for runtime invoke.
    auto sid = GetSubgraphID(func, "dnnl");

    auto builder = DnnlBuilder("dnnl_" + sid);
    builder.VisitExpr(func->body);
    code_stream_ << builder.JIT();
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and dnnl specific ones. To make
   * linking simpiler, the DNNL kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/dnnl folder.
   *
   * \param ref A object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateExternModule(const NodeRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // dnnl_kernel file is saved under src/runtime/contrib/dnnl so that we don't
    // expose it to ordinary users. To make export_library use it, users need to
    // pass -I${PATH_TO_TVM}/src/runtime/contrib
    code_stream_ << "#include <dnnl/dnnl_kernel.h>\n";
    code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "\n";

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

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external function";
    return (*pf)(code_stream_.str(), "cc");
  }

 private:
  /*! \brief The code stream that prints the external functions. */
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const NodeRef& ref) {
  DNNLCodegen dnnl;
  return dnnl.CreateExternModule(ref);
}

TVM_REGISTER_API("relay.ext.dnnl")
.set_body_typed(DNNLCompiler);

TVM_REGISTER_GLOBAL("relay.contrib.dnnl.enable")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = 1;
});

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
