/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <tvm/build_module.h>
#include <vector>
#include <string>
#include "./codegen_vhls.h"
#include "./build_common.h"
#include "../runtime/opencl/sdaccel/sdaccel_module.h"

namespace tvm {
namespace codegen {

void CodeGenVivadoHLS::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);

  this->stream << "#include <ap_int.h>\n\n";
}

void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
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

void CodeGenVivadoHLS::AddFunction(LoweredFunc f) {
  this->stream << "extern \"C\" ";
  CodeGenC::AddFunction(f);
}

void CodeGenVivadoHLS::PreFunctionBody(LoweredFunc f) {
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


runtime::Module BuildSDAccel(Array<LoweredFunc> funcs, std::string target_str) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenVivadoHLS cg;

  // Generate source code for get_source().
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string whole_code = cg.Finish();

  // Generate source code for compilation.
  Array<Array<Expr> > kernel_info;
  for (LoweredFunc f : funcs) {
    CodeGenVivadoHLS cg;
    cg.Init(output_ssa);
    cg.AddFunction(f);
    std::string code = cg.Finish();
    if (const auto* f = runtime::Registry::Get("tvm_callback_vhls_postproc")) {
      code = (*f)(code).operator std::string();
    }
    kernel_info.push_back(Array<Expr>({f->name, code}));
  }

  std::string xclbin;
  if (const auto* f = Registry::Get("tvm_callback_sdaccel_compile")) {
    Target target = Target::create(target_str);
    xclbin = (*f)(kernel_info, target->device_name).operator std::string();
  } else {
    LOG(FATAL) << "Cannot compile Vivado HLS code.";
  }
  return SDAccelModuleCreate(xclbin, "xclbin", ExtractFuncInfo(funcs), whole_code);
}

TVM_REGISTER_API("codegen.build_sdaccel")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSDAccel(args[0], args[1]);
  });

}  // namespace codegen
}  // namespace tvm
