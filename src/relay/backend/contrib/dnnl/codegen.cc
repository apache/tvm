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

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

// TODO(@zhiics, @comaniac): This is a basic implementation. We should implement
// all utilities and make a base class for users to implement.
class CodegenDNNL : public ExprVisitor, public CodegenCBase {
 public:
  explicit CodegenDNNL(const std::string& id) { this->ext_func_id_ = id; }

  void VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(node->name_hint());
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

    // Get the arguments for various DNNL kernels.
    if (IsOp(call, "nn.conv2d")) {
      decl_stream << "dnnl_conv2d";
      args = Conv2d(call);
    } else if (IsOp(call, "nn.dense")) {
      decl_stream << "dnnl_dense";
      args = Dense(call);
    } else if (IsOp(call, "nn.relu")) {
      decl_stream << "dnnl_relu";
      args = Relu(call);
    } else if (IsOp(call, "nn.batch_norm")) {
      decl_stream << "dnnl_bn";
      args = BatchNorm(call);
    } else if (IsOp(call, "add")) {
      decl_stream << "dnnl_add";
      args = Add(call);
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
      decl_stream << ", " << args[i];
    }
    decl_stream << ");";
    ext_func_body.push_back(decl_stream.str());

    // Update output buffer
    out_.clear();
    out_.push_back({out, out_size});
  }

  std::string JIT(void) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

 private:
  std::vector<std::string> Conv2d(const CallNode* call) {
    std::vector<std::string> args;
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    CHECK(conv2d_attr);

    auto ishape = GetShape(call->args[0]->checked_type());
    auto wshape = GetShape(call->args[1]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
    args.push_back(std::to_string(wshape[0]));
    args.push_back(std::to_string(conv2d_attr->groups));
    args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
    args.push_back(std::to_string(wshape[2]));
    args.push_back(std::to_string(wshape[3]));
    args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

    return args;
  }

  std::vector<std::string> Dense(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());
    auto wshape = GetShape(call->args[1]->checked_type());

    // Args: N, C, O
    args.push_back(std::to_string(ishape[0]));
    args.push_back(std::to_string(ishape[1]));
    args.push_back(std::to_string(wshape[0]));

    return args;
  }

  std::vector<std::string> Relu(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

  std::vector<std::string> BatchNorm(const CallNode* call) {
    std::vector<std::string> args;
    const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    // Args: epsilon
    args.push_back(std::to_string(bn_attr->epsilon));

    return args;
  }

  std::vector<std::string> Add(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

  /*! \brief The id of the external dnnl ext_func. */
  std::string ext_func_id_{""};
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
  std::vector<std::string> ext_func_args_;
  /*! \brief statement of the function that will be compiled using DNNL kernels. */
  std::vector<std::string> ext_func_body;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The name of the the outputs. */
  std::vector<std::pair<std::string, int>> out_;
};

/*!
 * \brief The DNNL codegen helper to generate wrapepr function calls of DNNL
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class DNNLModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding DNNL function for the given relay Function.
  void GenDNNLFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenDNNL builder(sid);
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
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // dnnl_kernel file is saved under src/runtime/contrib/dnnl so that we don't
    // expose it to ordinary users. To make export_library use it, users need to
    // pass -I${PATH_TO_TVM}/src/runtime/contrib
    code_stream_ << "#include <dnnl/dnnl_kernel.h>\n";
    code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "\n";

    if (ref->IsInstance<FunctionNode>()) {
      GenDNNLFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenDNNLFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc");
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const ObjectRef& ref) {
  DNNLModuleCodegen dnnl;
  return dnnl.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.dnnl").set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
