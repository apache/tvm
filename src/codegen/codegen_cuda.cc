/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_cuda.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

void CodeGenCUDA::AddFunction(LoweredFunc f) {
  this->stream << "extern \"C\" __global__ ";
  CodeGenC::AddFunction(f);
}

void CodeGenCUDA::PrintStmt(const ir::For* op) {
  int ext;
  CHECK(is_zero(op->min));
  if (arith::GetConstInt(op->extent, &ext) &&
      ext <= max_auto_unroll_) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::PrintStmt(op);
}

void CodeGenCUDA::PrintType(Type t, std::ostream& os) const {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: os << "half"; break;
      case 32: os << "float"; break;
      case 64: os << "double"; break;
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
      case 64: {
        if (lanes != 1 && sizeof(long) == 64) {  // NOLINT(*)
          os << "long"; break;
        } else {
          os << "int64_t"; break;
        }
      }
      case 1: os << "int"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecBinaryOp(
    const std::string&op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  // unpacking operations.
  int lanes = t.lanes();

  {
    // default: unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.type());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.type());
    std::string sret = GetUniqueName("_");
    {
      // delcare type.
      this->PrintIndent();
      this->PrintType(t, stream);
      stream << ' ' << sret << ";\n";
    }
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
    os << sret;
  }
}

void CodeGenCUDA::PrintVecElemLoad(
    const std::string& vec, Type t, int i, std::ostream& os) {  // NOLINT(*)
  const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  os << vec << "." << access[i];
}

void CodeGenCUDA::PrintVecElemStore(
    const std::string& vec, Type t, int i, const std::string& value) {
  this->PrintIndent();
  const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  stream << vec << "." << access[i] << " = " << value << ";\n";
}

void CodeGenCUDA::PrintStorageSync(const std::string& sync) {
  if (sync == "shared") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenCUDA::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_NE(scope, "global");
  if (scope == "shared") {
    os << "__shared__ ";
  }
}
}  // namespace codegen
}  // namespace tvm
