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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include "../../utils.h"
#include "codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief An example codegen that is only used for quick prototyping and testing
 * purpose. Only several binary options are covered. Users
 * may need to extend them to cover more operators.
 */
class CodegenC : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenC(const std::string& id) { this->ext_func_id_ = id; }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "C codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) final {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      CHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
    auto res = VisitExpr(op->tuple);
    CHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    Output output;
    output.name = ext_func_id_ + "const_" + std::to_string(const_idx_++);
    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    CHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    CHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
    output.dtype = dtype;

    CHECK_EQ(metadata_.count(output.name), 0U) << "variable must be unique: " << output.name;
    metadata_.Set(output.name, cn->data);

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

    if (IsOp(call, "add")) {
      macro_stream << "+";
    } else if (IsOp(call, "subtract")) {
      macro_stream << "-";
    } else if (IsOp(call, "multiply")) {
      macro_stream << "*";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    auto in_shape = GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      macro_stream << ", " << in_shape[i];
    }

    const auto* type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    macro_stream << ", " << dtype;

    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (auto out : res) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.name;
      }
    }

    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    buf_stream << dtype << "* " << out << " = (" << dtype << "*)std::malloc(4 * " << out_size
               << ");";
    buf_decl_.push_back(buf_stream.str());

    decl_stream << ", " << out << ");";
    ext_func_body.push_back(decl_stream.str());

    // Update output buffer
    // Note C codegen only handles TensorType. Therefore, we don't flatten
    // tuples and only return a single vaule.
    Output output;
    output.name = out;
    output.dtype = dtype;
    output.need_copy = true;
    output.size = out_size;
    return {output};
  }

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT(const std::vector<Output>& out) {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out);
  }

  Map<String, runtime::NDArray> GetMetadata() const { return metadata_; }

 private:
  /*! \brief The function id that represents a C source function. */
  std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The index of global constants. */
  int const_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Map<String, runtime::NDArray> metadata_;
};

class CSourceCodegen : public CSourceModuleCodegenBase {
 public:
  std::pair<std::string, Map<String, runtime::NDArray>> GenCFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenC builder(sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);

    return std::make_pair(sid, builder.GetMetadata());
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    // Append some common macro for operator definition.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_, p_DTYPE)       \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {    \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                        \
          out[i] = a[i] p_OP_ b[i];                                    \
        }                                                              \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_, p_DTYPE)  \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }
    )op_macro";

    code_stream_ << operator_macro << "\n\n";

    Map<String, String> code;
    Map<String, Map<String, runtime::NDArray>> metadata;
    if (ref->IsInstance<FunctionNode>()) {
      auto ret = GenCFunc(Downcast<Function>(ref));
      String sym = std::get<0>(ret);
      Map<String, runtime::NDArray> consts = std::get<1>(ret);
      std::string code_str = code_stream_.str();
      if (!consts.empty()) {
        code_str = "#include \"metadata.h\"\n" + code_str;
        metadata.Set(sym, consts);
      }
      code.Set(sym, code_str);
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        auto ret = GenCFunc(Downcast<Function>(it.second));
        Map<String, runtime::NDArray> consts = std::get<1>(ret);
        if (!consts.empty()) {
          metadata.Set(std::get<0>(ret), consts);
        }
      }
      std::string code_str = code_stream_.str();
      if (!metadata.empty()) {
        code_str = "#include \"metadata.h\"\n" + code_str;
      }
      code.Set("all", code_str);
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a SourceMetadataModuleNode
    const auto* pf = runtime::Registry::Get("runtime.SourceMetadataModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", metadata);
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
runtime::Module CCompiler(const ObjectRef& ref) {
  CSourceCodegen csource;
  return csource.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
