/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opencl.cc
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_opencl.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

CodeGenOpenCL::CodeGenOpenCL() {
  restrict_keyword_ = "restrict";
}

void CodeGenOpenCL::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->args) {
    if (arg.type().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

void CodeGenOpenCL::AddFunction(LoweredFunc f) {
  this->stream << "__kernel ";
  CodeGenC::AddFunction(f);
}

std::string CodeGenOpenCL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream
        << "#ifdef cl_khr_fp16\n"
           "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
           "#elif defined(cl_amd_fp16)\n"
           "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
           "#else\n"
           "#error \"Half precision floating point not supported"
                    "by OpenCL implementation on your device.\" \n"
           "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream
        << "#ifdef cl_khr_fp64\n"
           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
           "#elif defined(cl_amd_fp64)\n"
           "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
           "#else\n"
           "#error \"Double precision floating point not supported"
                    "by OpenCL implementation on your device.\" \n"
           "#endif\n\n";
  }

  return CodeGenC::Finish();
}

void CodeGenOpenCL::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::make(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else {
    os << "get_group_id(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] =
      CastFromTo(os.str(), UInt(64), iv->var.type());
}

void CodeGenOpenCL::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        enable_fp16_ = true;
        break;
      case 32: os << "float"; break;
      case 64:
        os << "double";
        enable_fp64_ = true;
        break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
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
      case 64: os << "long"; break;
      case 1: os << "int"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to OpenCL type";
}

void CodeGenOpenCL::PrintVecAddr(const Variable* buffer, Type t,
                                 Expr base, std::ostream& os) {  // NOLINT(*)
  if (!HandleTypeMatch(buffer, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    os << ' ';
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer) << " + ";
  PrintExpr(base, os);
}
std::string CodeGenOpenCL::GetVecLoad(
    Type t, const Variable* buffer, Expr base) {
  std::ostringstream os;
  os << "vload" << t.lanes() << "(0, ";
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

void CodeGenOpenCL::PrintVecStore(const Variable* buffer,
                                  Type t, Expr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
}

void CodeGenOpenCL::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    LOG(FATAL) << "warp sync not supported in opencl";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenOpenCL::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global") {
    os << "__global";
  } else if (scope == "shared") {
    os << "__local";
  }
}

std::string CodeGenOpenCL::CastFromTo(std::string value, Type from, Type target) {
  if (from == target) return value;
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
  }
  return os.str();
}

void CodeGenOpenCL::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->type, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}
}  // namespace codegen
}  // namespace tvm
