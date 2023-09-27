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
 * \file codegen_vhls.cc
 */
#include "codegen_vhls.h"

#include <string>
#include <vector>

#include "../../runtime/opencl/sdaccel/sdaccel_module.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

void CodeGenVivadoHLS::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);

  this->stream << "#include <ap_int.h>\n\n";
  this->stream << "#include <algorithm>\n\n";
}

void CodeGenVivadoHLS::PrintType(DataType t, std::ostream& os) {
  if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
        os << "unsigned char";
        break;
      case 16:
        os << "unsigned short";
        break;
      case 32:
        os << "unsigned int";
        break;
      case 64:
        os << "unsigned long long";
        break;
      default:
        os << "ap_uint<" << t.bits() << ">";
        break;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long long";
        break;
      default:
        os << "ap_int<" << t.bits() << ">";
        break;
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenVivadoHLS::PrintFuncPrefix(std::ostream& os) { os << "extern \"C\" "; }

void CodeGenVivadoHLS::PreFunctionBody(const PrimFunc& f) {
  for (size_t i = 0; i < f->params.size(); ++i) {
    Var v = f->params[i];
    std::string vid = GetVarID(v.get());
    if (v.dtype().is_handle()) {
      this->stream << "#pragma HLS INTERFACE m_axi port=" << vid << "  offset=slave bundle=gmem\n";
    }
    this->stream << "#pragma HLS INTERFACE s_axilite port=" << vid << " bundle=control\n";
  }
  this->stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenVivadoHLS* p) {
  os << opstr << '(';
  p->PrintExpr(op->a, os);
  os << ", ";
  p->PrintExpr(op->b, os);
  os << ')';
}

void CodeGenVivadoHLS::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  const char* opstr = "std::min";
  if (op->dtype.is_float()) {
    switch (op->dtype.bits()) {
      case 32:
        opstr = "fminf";
        break;
      case 64:
        opstr = "fmin";
        break;
    }
  }

  PrintBinaryExpr(op, opstr, os, this);
}

void CodeGenVivadoHLS::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  const char* opstr = "std::max";
  if (op->dtype.is_float()) {
    switch (op->dtype.bits()) {
      case 32:
        opstr = "fmaxf";
        break;
      case 64:
        opstr = "fmax";
        break;
    }
  }

  PrintBinaryExpr(op, opstr, os, this);
}

runtime::Module BuildSDAccel(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenVivadoHLS cg;

  // Generate source code for get_source().
  cg.Init(output_ssa);

  Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenVHLS: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenVLHS: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string whole_code = cg.Finish();

  // Generate source code for compilation.
  Array<Array<runtime::String>> kernel_info;

  for (auto [gvar, prim_func] : functions) {
    CodeGenVivadoHLS cg;
    cg.Init(output_ssa);

    for (auto [other_gvar, other_prim_func] : functions) {
      cg.DeclareFunction(other_gvar, other_prim_func);
    }
    cg.AddFunction(gvar, prim_func);
    std::string code = cg.Finish();
    if (const auto* f = runtime::Registry::Get("tvm_callback_vhls_postproc")) {
      code = (*f)(code, target).operator std::string();
    }

    auto function_name = cg.GetFunctionName(gvar);
    kernel_info.push_back({function_name, code});
  }

  std::string xclbin;
  if (const auto* f = Registry::Get("tvm_callback_sdaccel_compile")) {
    xclbin = (*f)(kernel_info, target).operator std::string();
  } else {
    LOG(FATAL) << "Cannot compile Vivado HLS code.";
  }
  return SDAccelModuleCreate(xclbin, "xclbin", ExtractFuncInfo(mod), whole_code);
}

TVM_REGISTER_GLOBAL("target.build.sdaccel").set_body_typed(BuildSDAccel);

}  // namespace codegen
}  // namespace tvm
