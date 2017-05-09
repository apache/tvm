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
  size_t pos = mode.find("-");
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  std::string build_f_name = "codegen.build_" + mode;
  // Lower intrinsic functions
  Array<LoweredFunc> func_list;
  for (LoweredFunc f : funcs) {
    func_list.push_back(ir::LowerIntrin(f, target));
  }
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = (*bf)(func_list, target);
  return m;
}

std::string PackImportsToC(const runtime::Module& mod) {
  std::string bin;
  dmlc::MemoryStringStream ms(&bin);
  dmlc::Stream* stream = &ms;
  uint64_t sz = static_cast<uint64_t>(mod->imports().size());
  stream->Write(sz);
  for (runtime::Module im : mod->imports()) {
    CHECK_EQ(im->imports().size(), 0U)
        << "Only support simply one-level hierachy";
    std::string tkey = im->type_key();
    std::string bin;
    stream->Write(tkey);
    im->SaveToBinary(stream);
  }
  // translate to C program
  std::ostringstream os;
  os << "#ifdef __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif\n";
  os << "extern const char " << runtime::symbol::tvm_dev_mblob << "[];\n";
  os << "extern const unsigned long " << runtime::symbol::tvm_dev_mblob_nbytes << ";\n";
  os << "const char " << runtime::symbol::tvm_dev_mblob
     << "[" << bin.length() << "] = {\n  ";
  os << std::hex;
  size_t nunit = 80 / 4;
  for (size_t i = 0; i < bin.length(); ++i) {
    // sperators
    if (i != 0) {
      if (i % nunit == 0) {
        os << ",\n  ";
      } else {
        os << ",";
      }
    }
    int c = bin[i];
    os << "0x" << (c & 0xff);
  }
  os << "\n};\n"
     << "const unsigned long " << runtime::symbol::tvm_dev_mblob_nbytes
     << " = " << std::dec << bin.length() << "UL;\n"
     << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n";
  return os.str();
}
}  // namespace codegen
}  // namespace tvm
