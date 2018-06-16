/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <vector>
#include <string>
#include "./codegen_vhls.h"
#include "./build_common.h"
#include "../runtime/opencl/opencl_module.h"

namespace tvm {
namespace codegen {

void CodeGenVHLS::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);

  this->stream << "#include <ap_int.h>\n\n";
}

void CodeGenVHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
        os << "unsigned char"; break;
      case 16:
        os << "unsigned short"; break;
      case 32:
        os << "unsigned int"; break;
      case 64:
        os << "unsigned long long"; break;
      default:
        os << "ap_uint<" << t.bits() << ">"; break;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8:
        os << "char"; break;
      case 16:
        os << "short"; break;
      case 32:
        os << "int"; break;
      case 64:
        os << "long long"; break;
      default:
        os << "ap_int<" << t.bits() << ">"; break;
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenVHLS::AddFunction(LoweredFunc f) {
  this->stream << "extern \"C\" ";
  CodeGenC::AddFunction(f);
}

void CodeGenVHLS::PreFunctionBody(LoweredFunc f) {
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = GetVarID(v.get());
    if (v.type().is_handle()) {
      this->stream << "#pragma HLS INTERFACE m_axi port=" << vid << "  offset=slave bundle=gmem\n";
    }
    this->stream << "#pragma HLS INTERFACE s_axilite port=" << vid << " bundle=control\n";
  }
  this->stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n";
}


runtime::Module BuildSDAccel(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenVHLS cg;

  CHECK_EQ(funcs.size(), 1);
  const std::string funcname = funcs[0]->name;

  cg.Init(output_ssa);

  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  if (const auto* f = runtime::Registry::Get("tvm_callback_vhls_postproc")) {
    code = (*f)(code).operator std::string();
  }

  std::string xclbin;
  if (const auto* f = Registry::Get("tvm_callback_sdaccel_compile")) {
    xclbin = (*f)(code, funcname).operator std::string();
  } else {
    LOG(FATAL) << "Cannot compile Vivado HLS code.";
  }
  return OpenCLModuleCreate(xclbin, "xclbin", ExtractFuncInfo(funcs), code);
}

TVM_REGISTER_API("codegen.build_sdaccel")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSDAccel(args[0]);
  });

}  // namespace codegen
}  // namespace tvm
