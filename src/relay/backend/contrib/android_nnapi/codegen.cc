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
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace android_nnapi {

class CodegenNNAPI : public backend::MemoizedExprTranslator< ::std::vector<Output> >,
                     public CodegenCBase {
 public:
  explicit CodegenNNAPI(const ::std::string& id) { this->ext_func_id_ = id; }

  ::std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Android NNAPI codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  ::std::vector<Output> VisitExpr_(const VarNode* var) final {
    ICHECK(var->checked_type()->IsInstance<TensorTypeNode>());
    Output output;
    output.name = var->name_hint();
    output.dtype = GetDtypeString(var->checked_type().as<TensorTypeNode>());
    return {output};
  }

  ::std::vector<Output> VisitExpr_(const FunctionNode* func) final {
    const ::std::string func_name = this->ext_func_id_ + "_" + std::to_string(this->func_idx_++);

    /* set function attrs */
    auto func_ref = GetRef<Function>(func);
    func_ref = WithAttr(::std::move(func_ref), "NnapiClassName", runtime::String(func_name));

    /* generate function body */
    {
      ::std::ostringstream def_stream;
      const auto* pf = backend::GetPackedFunc("relay.ext.android_nnapi.convert_relayir_to_nnapi");
      ICHECK(pf) << "Cannot find relay.ext.android_nnapi.convert_relayir_to_nnapi";
      const ::std::string nnapi_code = (*pf)(func_ref);
      def_stream << nnapi_code << "\n";
      this->func_decl_.push_back(def_stream.str());
    }

    /* create output buffer */
    ICHECK(func_ref->body->checked_type()->IsInstance<TensorTypeNode>())
        << "Expects single output Function to be converted for NNAPI";
    const TensorTypeNode* out_ttype = func_ref->body->checked_type().as<TensorTypeNode>();
    Output out;
    out.name = "buf_" + std::to_string(this->buf_idx_++);
    out.dtype = GetDtypeString(out_ttype);
    /* compute output buffer element count */
    {
      out.size = 1;
      const auto shape = backend::GetShape(func_ref->body->checked_type());
      for (const auto& dim : shape) {
        out.size *= dim;
      }
    }
    out.need_copy = true;
    {
      ::std::ostringstream buf_stream;
      buf_stream << out.dtype << " * " << out.name << " = static_cast< " << out.dtype
                 << " * >(::std::malloc(" << out_ttype->dtype.bytes() * out.size << "));";
      this->buf_decl_.push_back(buf_stream.str());
    }

    /* generate call to the generated function */
    {
      ::std::ostringstream call_stream;
      const ::std::string func_instance = func_name + "_instance";
      call_stream << "static " << func_name << " " << func_instance << "; ";
      call_stream << func_instance << ".execute(";
      for (size_t i = 0; i < func_ref->params.size(); ++i) {
        const auto& param = func_ref->params[i];
        ICHECK(param->IsInstance<VarNode>()) << "Function parameter should be relay.Var";
        this->ext_func_args_.push_back(param);
        const auto out = this->VisitExpr(param).front();
        call_stream << "reinterpret_cast< " << out.dtype << " * >(" << out.name << "), ";
      }
      call_stream << out.name << ");\n"; /* append the generated function call with output buffer */
      this->ext_func_body_.push_back(call_stream.str());
    }

    return {out};
  }

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  ::std::string JIT(const ::std::vector<Output>& out) {
    for (auto decl : this->func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(this->ext_func_id_, this->ext_func_args_, this->buf_decl_, this->ext_func_body_,
                   this->const_array_name_, out);
  }

 private:
  /*! \brief The function id that represents a C source function. */
  ::std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx_ = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  ::std::vector< ::std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration statements of a C compiler compatible function. */
  ::std::vector< ::std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  ::std::vector< ::std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  friend class NNAPICSourceCodegen;
};

class NNAPICSourceCodegen : public CSourceModuleCodegenBase {
 public:
  ::std::pair< ::std::string, Array<String> > GenCFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function";

    // Record the external symbol for runtime lookup.
    auto sid = backend::GetExtSymbol(func);

    CodegenNNAPI builder(sid);
    auto out = builder.VisitExpr(func);
    code_stream_ << builder.JIT(out);

    return {sid, builder.const_vars_};
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <vector>\n";
    code_stream_ << "#include <fcntl.h>\n";
    code_stream_ << "#include <unistd.h>\n";
    code_stream_ << "#include <sys/mman.h>\n";
    code_stream_ << "#include <android/NeuralNetworks.h>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/container.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenCFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();

    String sym = ::std::get<0>(res);
    Array<String> variables = ::std::get<1>(res);

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", Array<String>{sym}, variables);
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
  NNAPICSourceCodegen codegen;
  return codegen.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.android_nnapi").set_body_typed(CCompiler);

}  // namespace android_nnapi
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
