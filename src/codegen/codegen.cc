/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen.cc
 * \brief Common utilities to generated C style code.
 */
#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <dmlc/memory_io.h>
#include <sstream>
#include <iostream>

namespace tvm {
namespace codegen {

runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target) {
  std::string mode = target;
  size_t pos = mode.find(' ');
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  std::string build_f_name = "codegen.build_" + mode;
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = (*bf)(funcs, target);
  return m;
}

std::string PackImportsToC(const runtime::Module& mod, bool system_lib) {
  std::string bin;
  dmlc::MemoryStringStream ms(&bin);
  dmlc::Stream* stream = &ms;
  uint64_t sz = static_cast<uint64_t>(mod->imports().size());
  stream->Write(sz);
  for (runtime::Module im : mod->imports()) {
    CHECK_EQ(im->imports().size(), 0U)
        << "Only support simply one-level hierarchy";
    std::string tkey = im->type_key();
    std::string bin;
    stream->Write(tkey);
    im->SaveToBinary(stream);
  }
  // translate to C program
  std::ostringstream os;
  os << "#ifdef _WIN32\n"
     << "#define TVM_EXPORT __declspec(dllexport)\n"
     << "#else\n"
     << "#define TVM_EXPORT\n"
     << "#endif\n";
  os << "#ifdef __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif\n";
  os << "TVM_EXPORT extern const char " << runtime::symbol::tvm_dev_mblob << "[];\n";
  uint64_t nbytes = bin.length();
  os << "const char " << runtime::symbol::tvm_dev_mblob
     << "[" << bin.length() + sizeof(nbytes) << "] = {\n  ";
  os << std::hex;
  size_t nunit = 80 / 4;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    // sperators
    if (i != 0) {
      os << ",";
    }
    os << "0x" << ((nbytes >> (i * 8)) & 0xffUL);
  }
  for (size_t i = 0; i < bin.length(); ++i) {
    // sperators
    if ((i + sizeof(nbytes)) % nunit == 0) {
      os << ",\n  ";
    } else {
      os << ",";
    }
    int c = bin[i];
    os << "0x" << (c & 0xff);
  }
  os << "\n};\n";
  if (system_lib) {
    os << "extern int TVMBackendRegisterSystemLibSymbol(const char*, void*);\n";
    os << "static int " << runtime::symbol::tvm_dev_mblob << "_reg_ = "
       << "TVMBackendRegisterSystemLibSymbol(\"" << runtime::symbol::tvm_dev_mblob << "\", (void*)"
       << runtime::symbol::tvm_dev_mblob << ");\n";
  }
  os << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n";
  return os.str();
}
}  // namespace codegen
}  // namespace tvm
