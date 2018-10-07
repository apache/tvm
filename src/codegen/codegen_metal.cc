/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_metal.cc
 */
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include <algorithm>
#include "codegen_metal.h"
#include "build_common.h"
#include "../runtime/metal/metal_module.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

void CodeGenMetal::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  // analyze the data;
  for (Var arg : f->args) {
    if (arg.type().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenMetal::CodeGenMetal() {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "using namespace metal;\n\n";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int;\n"
              << "};\n\n";
}

void CodeGenMetal::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }
  // Function header.
  this->stream << "kernel void " << f->name << "(\n";
  // Buffer arguments
  size_t num_buffer = 0;
  for (size_t i = 0; i < f->args.size(); ++i, ++num_buffer) {
    Var v = f->args[i];
    if (!v.type().is_handle())  break;
    stream << "  ";
    std::string vid = AllocVarID(v.get());
    auto it = alloc_storage_scope_.find(v.get());
    CHECK(it != alloc_storage_scope_.end());
    PrintStorageScope(it->second, stream);
    stream << ' ';
    if (handle_data_type_.count(v.get())) {
      PrintType(handle_data_type_.at(v.get()), stream);
      stream << "*";
    } else {
      PrintType(v.type(), stream);
    }
    stream << ' ' << vid
           << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = f->args.size() - num_buffer;
  std::string varg = GetUniqueName("arg");
  if (nargs != 0) {
    std::string arg_buf_type = f->name + "_args_t";
    stream << "  constant " << arg_buf_type << "& " << varg
           << " [[ buffer(" << num_buffer << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < f->args.size(); ++i) {
      Var v = f->args[i];
      CHECK(!v.type().is_handle());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (v.type().bits() == 32) {
        decl_stream << "  ";
        PrintType(v.type(), decl_stream);
        decl_stream << " " << vid << ";\n";
        vref << varg << "." << vid;
      } else {
        // For non 32bit type, ref through arg union.
        decl_stream << "  __TVMArgUnion " << vid << ";\n";
        vref << varg << "." << vid << ".v_";
        PrintType(v.type(), vref);
      }
      var_idmap_[v.get()] = vref.str();
    }
    decl_stream << "};\n\n";
  }
  // Setup the thread group info.
  CHECK_EQ(GetUniqueName("threadIdx"), "threadIdx");
  CHECK_EQ(GetUniqueName("blockIdx"), "blockIdx");
  int work_dim = 0;
  for (IterVar iv : f->thread_axis) {
    runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
    work_dim = std::max(work_dim, scope.dim_index + 1);
  }
  if (work_dim != 0) {
    // use ushort by default for now
    stream << "  ";
    PrintType(UInt(thread_index_bits_, work_dim), stream);
    stream << " blockIdx [[threadgroup_position_in_grid]],\n";
    stream << "  ";
    PrintType(UInt(thread_index_bits_, work_dim), stream);
    stream << " threadIdx [[thread_position_in_threadgroup]]\n";
  }
  // bind thread axis
  for (IterVar iv : f->thread_axis) {
    CHECK(!var_idmap_.count(iv->var.get()));
    std::string vname = iv->thread_tag;
    if (work_dim <= 1) {
      vname = vname.substr(0, iv->thread_tag.length() - 2);
    }
    var_idmap_[iv->var.get()] =
        CastFromTo(vname, UInt(thread_index_bits_), iv->var.type());
  }
  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenMetal::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, UInt(thread_index_bits_), iv->var.type());
}

void CodeGenMetal::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  if (t == Bool()) {
    os << "bool"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: os << "half"; break;
      case 32: os << "float"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int"; return;
    }
    switch (t.bits()) {
      case 8: os << "char"; break;
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 1: os << "bool"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}

void CodeGenMetal::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier not supported";
  }
}

void CodeGenMetal::PrintVecElemLoad(const std::string& vec,
                                    Type t, int i,
                                    std::ostream& os) {  // NOLINT(*)
  os << vec << "[" << i << "]";
}

void CodeGenMetal::PrintVecElemStore(const std::string& vec,
                                     Type t, int i,
                                     const std::string& value) {
  this->PrintIndent();
  stream << vec << "[" << i << "]"
         << " = " << value << ";\n";
}

void CodeGenMetal::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global") {
    os << "device";
  } else if (scope == "shared") {
    os << "threadgroup";
  } else {
    os << "thread";
  }
}

void CodeGenMetal::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  PrintType(op->type, os);
  os << "(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

runtime::Module BuildMetal(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenMetal cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  std::string fmt = "metal";
  std::string source = "";
  if (const auto* f = Registry::Get("tvm_callback_metal_compile")) {
    source = code;
    code = (*f)(code).operator std::string();
    fmt = "metallib";
  }
  return MetalModuleCreate(code, fmt, ExtractFuncInfo(funcs), source);
}

TVM_REGISTER_API("codegen.build_metal")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildMetal(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
