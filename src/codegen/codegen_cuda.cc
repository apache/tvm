/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_cuda.h"
#include "./codegen_stack_vm.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_module.h"

namespace tvm {
namespace codegen {

std::string CodeGenCUDA::Compile(
    LoweredFunc f,
    bool output_ssa) {
  this->stream << "extern \"C\" __global__ ";
  return CodeGenC::Compile(f, output_ssa);
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

#if TVM_CUDA_RUNTIME
std::unordered_map<LoweredFunc, PackedFunc>
MakeNVRTC(Array<LoweredFunc> funcs) {
  std::ostringstream os;
  bool output_ssa = false;
  for (LoweredFunc f : funcs) {
    os << CodeGenCUDA().Compile(f, output_ssa);
    os << '\n';
  }
  std::string code = os.str();

  if (PackedFunc::GlobalExist("tvm_callback_cuda_postproc")) {
    const auto& f = PackedFunc::GetGlobal("tvm_callback_cuda_postproc");
    code = f(code).operator std::string();
  }
    LOG(INFO) << code;
  std::string ptx;
  if (PackedFunc::GlobalExist("tvm_callback_cuda_compile")) {
    const auto& f = PackedFunc::GetGlobal("tvm_callback_cuda_compile");
    ptx = f(code).operator std::string();
  } else {
    ptx = runtime::NVRTCCompile(os.str());
  }
  std::unordered_map<LoweredFunc, PackedFunc> ret;

  runtime::CUDAModule m = runtime::CUDAModule::Create(ptx);
  for (LoweredFunc f : funcs) {
    std::vector<TVMType> arg_types(f->args.size());
    std::vector<std::string> thread_axis_tags(f->thread_axis.size());

    for (size_t i = 0; i < f->args.size(); ++i) {
      arg_types[i] = Type2TVMType(f->args[i].type());
    }
    for (size_t i = 0; i < f->thread_axis.size(); ++i) {
      thread_axis_tags[i] = f->thread_axis[i]->thread_tag;
    }
    ret[f] = m.GetPackedFunc(f->name, arg_types, thread_axis_tags);
  }

  return ret;
}

PackedFunc BuildNVRTC(Array<LoweredFunc> fsplits, std::string host_mode) {
  Array<LoweredFunc> device_list(fsplits.begin() + 1, fsplits.end());
  std::unordered_map<LoweredFunc, PackedFunc> device_funcs = MakeNVRTC(device_list);
  if (host_mode == "stackvm") {
    StackVM vm = codegen::CodeGenStackVM().Compile(fsplits[0], device_funcs);
    auto f = [vm](TVMArgs args, TVMRetValue* rv) {
      runtime::AutoSetCUDADevice(args);
      vm(args);
    };
    return PackedFunc(f);
  } else {
    LOG(FATAL) << "unknown host mode " << host_mode;
    return PackedFunc();
  }
}
#else
// dummy function when cuda is not available
PackedFunc BuildNVRTC(Array<LoweredFunc> func, std::string host_mode) {
  LOG(FATAL) << "CUDA is not enabled";
  return PackedFunc();
}
#endif   // TVM_CUDA_RUNTIME
}  // namespace codegen
}  // namespace tvm
