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
#include <numeric>

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

  void VisitExpr_(const CallNode* call) final {
    struct Output {
      std::string decl, buf;
      int out_size = 1;
      std::string out;
    };

    auto generate_body = [&](const CallNode* root_call, const std::string& func_name,
                             const std::vector<std::string>& args,
                             const std::vector<std::string>& fused_func_args) {
      // Make function call with input buffers when visiting arguments
      bool first = true;
      std::ostringstream arg_stream;
      arg_stream << "(";
      for (size_t i = 0; i < root_call->args.size(); ++i) {
        VisitExpr(root_call->args[i]);
        for (auto out : out_) {
          if (!first) {
            arg_stream << ", ";
          }
          first = false;
          arg_stream << out.first;
        }
      }

      for (auto arg_name : fused_func_args) {
        arg_stream << ", " << arg_name;
      }

      // Analyze the output buffer
      auto type_node = root_call->checked_type().as<TensorTypeNode>();
      CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
          << "Only support single output tensor with float type";

      auto out_shape = GetShape(root_call->checked_type());

      Output ret;
      ret.out = "buf_" + std::to_string(buf_idx_++);
      ret.out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

      this->PrintIndents();

      std::ostringstream buf_stream;
      buf_stream << "float* " << ret.out << " = (float*)std::malloc(4 * " << ret.out_size << ");";
      ret.buf = buf_stream.str();

      arg_stream << ", " << ret.out;
      // Attach attribute arguments
      for (size_t i = 0; i < args.size(); ++i) {
        arg_stream << ", " << args[i];
      }
      arg_stream << ");";
      ret.decl = func_name + arg_stream.str();

      return ret;
    };

    Output ret;
    if (auto conv_call = DetectFusedConv2DBiasReLU(call)) {
      LOG(INFO) << "found fused op, num_args = " << call->args.size();
      ret = generate_body(conv_call, "dnnl_fused_conv2d_bias_relu",
                          FusedConv2dBiasReLU(conv_call), ext_fused_func_args_);
    } else if (IsOp(call, "nn.conv2d")) {
      ret = generate_body(call, "dnnl_conv2d", Conv2d(call), {});
    } else if (IsOp(call, "nn.dense")) {
      ret = generate_body(call, "dnnl_dense", Dense(call), {});
    } else if (IsOp(call, "nn.relu")) {
      ret = generate_body(call, "dnnl_relu", Relu(call), {});
    } else if (IsOp(call, "nn.batch_norm")) {
      ret = generate_body(call, "dnnl_bn", BatchNorm(call), {});
    } else if (IsOp(call, "add")) {
      ret = generate_body(call, "dnnl_add", Add(call), {});
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }

    buf_decl_.push_back(ret.buf);
    ext_func_body.push_back(ret.decl);

    // Update output buffer
    out_.clear();
    out_.push_back({ret.out, ret.out_size});
  }

  std::string JIT(void) {
    ext_func_args_.insert(ext_func_args_.end(),
                          ext_fused_func_args_.begin(),
                          ext_fused_func_args_.end());
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

 private:
  const CallNode* DetectFusedConv2DBiasReLU(const CallNode* call) {
    auto arg = call->args[0];
    // if (auto next_call = arg.as<CallNode>()) {
    //   if (IsOp(next_call, "nn.conv2d")) {
    //   }
    // }
    return nullptr;
  }

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

  std::vector<std::string> FusedConv2dBiasReLU(const CallNode* call) {
    return Conv2d(call);
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
  std::vector<std::string> ext_fused_func_args_;
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
        LOG(INFO) << "Invoking  GenDNNLFunc on FuncNode";
      GenDNNLFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        LOG(INFO) << "Invoking  GenDNNLFunc";
        GenDNNLFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    LOG(INFO) << code_stream_.str();
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
  LOG(INFO) << "Invoking DNNLCompiler";
  auto ret = dnnl.CreateCSourceModule(ref);
  LOG(INFO) << "Done invoking DNNLCompiler";
  return ret;
}

TVM_REGISTER_GLOBAL("relay.ext.dnnl").set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
