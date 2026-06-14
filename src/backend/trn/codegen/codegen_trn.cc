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
 * \file codegen_trn.cc
 */
#include "codegen_trn.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../../runtime/thread_storage_scope.h"
#include "../../../target/build_common.h"

namespace tvm {
namespace codegen {
namespace {
std::string PrintShapeAsList(const ffi::Array<PrimExpr>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) os << ", ";
    os << shape[i];
  }
  os << "]";
  return os.str();
}
}  // namespace

void CodeGenTrainium::InitFuncState(const PrimFunc& f) { CodeGenC::InitFuncState(f); }

CodeGenTrainium::CodeGenTrainium(Target target) : target_(target) {
  decl_stream << "import neuronxcc.nki.language as nl\n";
  decl_stream << "from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace\n";
  decl_stream << "import numpy as np\n";
  decl_stream << "import neuronxcc.nki.isa as nisa\n";
  decl_stream << "import math\n";
  decl_stream << "import neuronxcc.nki as nki\n";
  decl_stream << "import neuronxcc.nki.typing as nt\n";
  decl_stream << "import neuronxcc.nki.compiler as ncc\n";
  decl_stream << "@nki.compiler.enable_stack_allocator\n";
  decl_stream << "@nki.compiler.skip_middle_end_transformations\n";
  decl_stream << "@baremetal(experimental_flags='enable-mutable-parameter', "
                 "additional_compile_opt='--internal-skip-backend-allocation-opt-nki')\n";
  opcode_map_ = {{"sqrt", "nki.language.sqrt"},    {"add", "nki.language.add"},
                 {"sub", "nki.language.subtract"}, {"mul", "nki.language.multiply"},
                 {"max", "nki.language.maximum"},  {"min", "nki.language.minimum"},
                 {"exp", "nki.language.exp"}};
}

void CodeGenTrainium::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  // NOTE: There is no inter-function calls among Trainium kernels.
  // For now we keep the Trainium codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Trainium function declaration is properly printed.
  // In Trainium, for PrimFuncs with signature
  //    def func(A: Buffer, B: Buffer, x: int, y: float) -> None
  // where there are trailing pod parameters, the codegen emits a struct
  //    struct func_params{ x: int; y: float; }
  // for the function. In the flow of inter-function call process,
  // the struct will be emitted for every time a function is declared.
  // So consequently there are duplicate appearances of a same struct,
  // which makes the Trainium compiler unable to recognize.

  // clear previous generated state.
  this->InitFuncState(func);
  buffer_idmap_.clear();
  data_buffer_idmap_.clear();
  data_decl_buffer_map_.clear();
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("v_");

  // add to alloc buffer type.
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  TVM_FFI_ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "def " << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  auto num_inputs = func->GetAttr<int64_t>(tvm::attr::kNumInputs);
  TVM_FFI_ICHECK(num_inputs.has_value());
  std::vector<std::string> output_vids;
  size_t num_buffer = 0;
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle()) {
      LOG(FATAL) << "Trainium codegen currently only support buffer arguments";
    };
    std::string vid = AllocVarID(v.get());
    if (i >= static_cast<size_t>(num_inputs.value())) {
      this->stream << vid << ": nt.mutable_tensor, ";
      output_vids.push_back(vid);
    } else {
      this->stream << vid << ", ";
    }
  }

  // the function scope.
  stream << "):\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(func->body);
  this->PrintIndent();
  stream << "return ";
  for (size_t i = 0; i < output_vids.size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << output_vids[i];
  }
  this->EndScope(func_scope);
}

void CodeGenTrainium::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  TVM_FFI_ICHECK(lanes == 1) << "Trainium codegen does not support vector types";
  TVM_FFI_ICHECK(!t.is_handle()) << "Trainium codegen does not support handle type";
  TVM_FFI_ICHECK(!t.is_void()) << "Trainium codegen does not support void type";
  if (t == DataType::Bool()) {
    os << "np.bool";
    return;
  }
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "np.float16";
        break;
      case 32:
        os << "np.float32";
        break;
      default:
        LOG(FATAL) << "Trainium codegen does not support float type with bits " << t.bits();
        break;
    }
    return;
  }
  if (t.is_uint() || t.is_int()) {
    if (t.bits() == 1) {
      os << "np.bool";
      return;
    }
    os << "np.";
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8";
        break;
      case 16:
        os << "int16";
        break;
      case 32:
        os << "int32";
        break;
      case 64:
        os << "int64";
        break;
      default:
        LOG(FATAL) << "Trainium codegen does not support int type with bits " << t.bits();
        break;
    }
    return;
  }
  if (t.is_bfloat16()) {
    os << "nl.bfloat16";
    return;
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Trainium type";
}

std::string CodeGenTrainium::GetStorageScopeStr(const std::string& scope) {  // NOLINT(*)
  if (scope == "global") {
    return "nl.hbm";
  } else if (scope == "trn.sbuf") {
    return "nl.sbuf";
  } else if (scope == "trn.psum") {
    return "nl.psum";
  } else {
    LOG(FATAL) << "Unknown storage scope `" << scope << "`";
    return "";
  }
}

void CodeGenTrainium::VisitStmt_(const AllocBufferNode* op) {
  TVM_FFI_ICHECK(op->buffer.defined());
  std::string vid = AllocVarID(op->buffer->data.get());

  this->PrintIndent();
  auto scope = GetPtrStorageScope(op->buffer->data);
  std::ostringstream dtype_os;
  PrintType(op->buffer->dtype, dtype_os);
  std::string dtype_str = dtype_os.str();
  if (scope == "trn.psum") {
    stream << vid << " = nl.ndarray(shape=[";
    TVM_FFI_ICHECK(op->buffer->shape.size() == 3);
    stream << PrintExpr(op->buffer->shape[0]) << ", nl.par_dim(" << PrintExpr(op->buffer->shape[1])
           << "), " << PrintExpr(op->buffer->shape[2]) << "], dtype=" << dtype_str << ", buffer=";
  } else {
    stream << vid << " = nl.ndarray(shape=" << PrintShapeAsList(op->buffer->shape)
           << ", dtype=" << dtype_str << ", buffer=";
  }
  Array<PrimExpr> addr;
  if (auto allocated_addr = op->annotations.Get(tirx::attr::buffer_allocated_addr)) {
    addr = Downcast<Array<PrimExpr>>(allocated_addr.value());
  } else {
    // AllocBuffer is a leaf stmt after rebase; in that path allocated_addr is carried by Buffer.
    addr = op->buffer->allocated_addr;
  }
  if (addr.empty()) {
    stream << GetStorageScopeStr(scope) << ")\n";
  } else {
    if (scope == "trn.psum") {
      TVM_FFI_ICHECK(addr.size() == 2);
      TVM_FFI_ICHECK(addr[0]->IsInstance<IntImmNode>())
          << "allocated_addr[0] must be a constant integer, got: " << addr[0];
      TVM_FFI_ICHECK(addr[1]->IsInstance<IntImmNode>())
          << "allocated_addr[1] must be a constant integer, got: " << addr[1];
      int64_t base_bank = Downcast<IntImm>(addr[0])->value;
      int64_t base_addr = Downcast<IntImm>(addr[1])->value;
      stream << "ncc.psum.mod_alloc(base_bank=" << base_bank << ", base_addr=" << base_addr;
      stream << ", num_bank_tiles=(" << op->buffer->shape[0] << ",)))\n";
    } else {
      TVM_FFI_ICHECK(addr.size() == 1);
      TVM_FFI_ICHECK(addr[0]->IsInstance<IntImmNode>())
          << "allocated_addr[0] must be a constant integer, got: " << addr[0];
      int64_t base_addr = Downcast<IntImm>(addr[0])->value;
      stream << "ncc.sbuf.mod_alloc(base_addr=" << base_addr << "))\n";
    }
  }
}

void CodeGenTrainium::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tirx::attr::tensorized_nki_instruction) {
    ctx_.tensorizing = true;
    ctx_.mask = PrimExpr(nullptr);
    ctx_.loopvar2dim.clear();
    ctx_.is_matmul_input = false;
  }
  this->PrintStmt(op->body);
  if (op->attr_key == tirx::attr::tensorized_nki_instruction) {
    ctx_.tensorizing = false;
  }
}

void CodeGenTrainium::VisitStmt_(const ForNode* op) {
  bool is_outermost_loop = is_outermost_loop_;
  is_outermost_loop_ = false;
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  TVM_FFI_ICHECK(is_zero(op->min));
  if (ctx_.tensorizing) {
    stream << vid << " = nl.arange(" << extent << ")\n";
    if (op->annotations.count("nki_dim")) {
      ctx_.loopvar2dim[op->loop_var.get()] = Downcast<ffi::String>(op->annotations["nki_dim"]);
    }
    ctx_.tensorized_loop_vars.insert(op->loop_var.get());
    TVM_FFI_ICHECK(ctx_.loopvar2dim.empty() ||
                   ctx_.loopvar2dim.size() == ctx_.tensorized_loop_vars.size())
        << "nki_dim attribute must be specified for all tensorized loop variables or none of them";
    PrintStmt(op->body);
    ctx_.tensorized_loop_vars.erase(op->loop_var.get());
  } else {
    if (is_outermost_loop) {
      stream << "for " << vid << " in nl.sequential_range(" << extent
             << ", body_no_reorder=True):\n";
    } else {
      stream << "for " << vid << " in nl.sequential_range(" << extent << "):\n";
    }
    int for_scope = BeginScope();
    PrintStmt(op->body);
    EndScope(for_scope);
  }
  is_outermost_loop_ = is_outermost_loop;
}

std::string CodeGenTrainium::PrintIndices(const Array<PrimExpr>& indices) {
  std::ostringstream os;
  ctx_.buffer_index = 0;
  ctx_.used_var_cnt = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    PreOrderVisit(indices[i], [&](const ffi::ObjectRef& node) {
      if (const auto* v = node.as<VarNode>()) {
        if (ctx_.tensorized_loop_vars.count(v)) {
          ctx_.used_var_cnt++;
        }
      }
      return true;
    });
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << PrintExpr(indices[i]);
  }
  ctx_.buffer_index = -1;
  return os.str();
}

void CodeGenTrainium::VisitStmt_(const BufferStoreNode* op) {
  LOG(FATAL) << "Trainium codegen does not support buffer store";
}

void CodeGenTrainium::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << vid << "\n";
  }
}

void CodeGenTrainium::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  std::string buffer_str;
  if (buffer_idmap_.count(op->buffer)) {
    buffer_str = buffer_idmap_[op->buffer];
  } else {
    buffer_str = GetVarID(op->buffer->data.get());
  }
  os << buffer_str << "[";
  os << PrintIndices(op->indices);
  os << "]";
}

std::string PrintBool(bool b) { return b ? "True" : "False"; }

void CodeGenTrainium::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  TVM_FFI_ICHECK(!op->op.as<GlobalVarNode>())
      << "CodegenTrainium does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc " << op->op;
  const auto* op_node = op->op.as<OpNode>();
  auto is_op = [&](const Op& compat, const char* canonical_name) {
    return op->op.same_as(compat) || (op_node != nullptr && op_node->name == canonical_name);
  };
  static const Op& nki_matmul_op = Op::Get("tirx.nki_matmul");
  static const Op& nki_load_op = Op::Get("tirx.nki_load");
  static const Op& nki_store_op = Op::Get("tirx.nki_store");
  static const Op& nki_tensor_copy_op = Op::Get("tirx.nki_tensor_copy");
  static const Op& nki_activation_op = Op::Get("tirx.nki_activation");
  static const Op& nki_reciprocal_op = Op::Get("tirx.nki_reciprocal");
  static const Op& nki_tensortensor_op = Op::Get("tirx.nki_tensortensor");
  static const Op& nki_tensorscalar_op = Op::Get("tirx.nki_tensorscalar");
  static const Op& nki_memset_op = Op::Get("tirx.nki_memset");
  static const Op& nki_tensorreduce_op = Op::Get("tirx.nki_tensorreduce");
  static const Op& nki_activation_reduce_op = Op::Get("tirx.nki_activation_reduce");
  static const Op& nki_tensorscalar_reduce_op = Op::Get("tirx.nki_tensorscalar_reduce");
  static const Op& nki_identity_op = Op::Get("tirx.nki_identity");
  static const Op& nki_scalar_tensor_tensor_op = Op::Get("tirx.nki_scalar_tensor_tensor");
  static const Op& nki_scalar_tensor_scalar_op = Op::Get("tirx.nki_scalar_tensor_scalar");
  static const Op& nki_affine_select_op = Op::Get("tirx.nki_affine_select");

  if (is_op(nki_matmul_op, "tirx.nki.matmul")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 4);
    std::string accum = is_one(op->args[3]) ? " += " : " = ";
    os << PrintExpr(op->args[0]) << accum;
    ctx_.is_matmul_input = true;
    os << "nisa.nc_matmul(" << PrintExpr(op->args[1]) << "," << PrintExpr(op->args[2]);
  } else if (is_op(nki_load_op, "tirx.nki.load")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nl.load(" << PrintExpr(op->args[1]);
  } else if (is_op(nki_store_op, "tirx.nki.store")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2);
    os << "nl.store(" << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1]);
  } else if (is_op(nki_tensor_copy_op, "tirx.nki.tensor_copy")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nisa.tensor_copy(" << PrintExpr(op->args[1]);
  } else if (is_op(nki_activation_op, "tirx.nki.activation")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5);
    // nki_activation(result, data, opcode, bias, scale)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[2].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[2].as<StringImmNode>()->value];
    os << PrintExpr(op->args[0]) << " = nisa.activation(op=" << nki_op
       << ", data=" << PrintExpr(op->args[1]) << ",";
    os << "bias=" << PrintExpr(op->args[3]) << ", scale=" << PrintExpr(op->args[4]);
  } else if (is_op(nki_reciprocal_op, "tirx.nki.reciprocal")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nisa.reciprocal(" << PrintExpr(op->args[1]);
  } else if (is_op(nki_tensortensor_op, "tirx.nki.tensortensor")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 4);
    // nki_tensortensor(result, data1, data2, opcode)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[3].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[3].as<StringImmNode>()->value];
    os << PrintExpr(op->args[0]) << " = nisa.tensor_tensor(" << PrintExpr(op->args[1]) << ", ";
    os << PrintExpr(op->args[2]) << ", op=" << nki_op;
  } else if (is_op(nki_tensorscalar_op, "tirx.nki.tensorscalar")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5);
    // nki_tensorscalar(result, operand0, operand1, opcode, reverse)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[3].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[3].as<StringImmNode>()->value];
    bool reverse = op->args[4].as<IntImmNode>()->value != 0;
    os << PrintExpr(op->args[0]) << " = nisa.tensor_scalar(" << PrintExpr(op->args[1])
       << ", operand0=";
    os << PrintExpr(op->args[2]) << ", op0=" << nki_op << ", reverse0=" << PrintBool(reverse);
  } else if (is_op(nki_memset_op, "tirx.nki.memset")) {
    TVM_FFI_ICHECK_GE(op->args.size(), 2);
    // result, value
    os << PrintExpr(op->args[0]) << " = " << PrintExpr(op->args[1]);
    TVM_FFI_ICHECK(!ctx_.mask.defined()) << "memset cannot have mask";
    return;
  } else if (is_op(nki_tensorreduce_op, "tirx.nki.tensorreduce")) {
    TVM_FFI_ICHECK(op->args.size() >= 5)
        << "nki_tensorreduce expects at least 5 arguments, but got " << op->args.size();
    // nki_tensorreduce(result, data, opcode, negate, *axes)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[2].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[2].as<StringImmNode>()->value];
    bool negate = op->args[3].as<IntImmNode>()->value != 0;
    Array<PrimExpr> axes(op->args.begin() + 4, op->args.end());
    os << PrintExpr(op->args[0]) << " = nisa.tensor_reduce(data=" << PrintExpr(op->args[1])
       << ", op=" << nki_op << ", negate=" << PrintBool(negate) << ", axis=" << axes;
  } else if (is_op(nki_activation_reduce_op, "tirx.nki.activation_reduce")) {
    TVM_FFI_ICHECK(op->args.size() == 7)
        << "nki_activation_reduce expects 7 arguments, but got " << op->args.size();
    // nki_activation_reduce(reduce_res, act_res, data, opcode, reduce_opcode, bias, scale)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[3].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[3].as<StringImmNode>()->value];
    TVM_FFI_ICHECK(opcode_map_.count(op->args[4].as<StringImmNode>()->value));
    std::string reduce_nki_op = opcode_map_[op->args[4].as<StringImmNode>()->value];
    os << PrintExpr(op->args[1]) << " = nisa.activation_reduce(data=" << PrintExpr(op->args[2])
       << ", op=" << nki_op;
    os << ", reduce_op=" << reduce_nki_op << ", reduce_res=" << PrintExpr(op->args[0])
       << ", bias=" << PrintExpr(op->args[5]) << ", scale=" << PrintExpr(op->args[6]);
  } else if (is_op(nki_tensorscalar_reduce_op, "tirx.nki.tensorscalar_reduce")) {
    TVM_FFI_ICHECK(op->args.size() == 7)
        << "nki_tensorscalar_reduce expects 7 arguments, but got " << op->args.size();
    // nki_tensorscalar_reduce(reduce_res, tensorscalar_res, operand0, operand1, opcode,
    // reduce_opcode, reverse)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[4].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[4].as<StringImmNode>()->value];
    TVM_FFI_ICHECK(opcode_map_.count(op->args[5].as<StringImmNode>()->value));
    std::string reduce_nki_op = opcode_map_[op->args[5].as<StringImmNode>()->value];
    bool reverse = op->args[6].as<IntImmNode>()->value != 0;
    os << PrintExpr(op->args[1]) << " = nisa.tensor_scalar_reduce(data=" << PrintExpr(op->args[2])
       << ", op0=" << nki_op << ", operand0=" << PrintExpr(op->args[3])
       << ", reduce_op=" << reduce_nki_op << ", reduce_res=" << PrintExpr(op->args[0])
       << ", reverse0=" << PrintBool(reverse);
  } else if (is_op(nki_identity_op, "tirx.nki.identity")) {
    // nki_identity(result, size)
    TVM_FFI_ICHECK_EQ(op->args.size(), 2);
    auto identity_np_name = name_supply_->FreshName("identity_np");
    os << identity_np_name << " = nl.shared_constant(np.identity(" << PrintExpr(op->args[1])
       << ", dtype=np.int8), dtype=nl.bfloat16)" << std::endl;
    for (int i = 0; i < indent_; ++i) {
      os << ' ';
    }
    os << PrintExpr(op->args[0]) << " = nl.load(" << identity_np_name;
  } else if (is_op(nki_scalar_tensor_tensor_op, "tirx.nki.scalar_tensor_tensor")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 8);
    // nki_scalar_tensor_tensor(result, data, operand0, operand1, opcode0, opcode1, reverse0,
    // reverse1)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[4].as<StringImmNode>()->value));
    std::string nki_op0 = opcode_map_[op->args[4].as<StringImmNode>()->value];
    TVM_FFI_ICHECK(opcode_map_.count(op->args[5].as<StringImmNode>()->value));
    std::string nki_op1 = opcode_map_[op->args[5].as<StringImmNode>()->value];
    bool reverse0 = op->args[6].as<IntImmNode>()->value != 0;
    bool reverse1 = op->args[7].as<IntImmNode>()->value != 0;
    os << PrintExpr(op->args[0]) << " = nisa.scalar_tensor_tensor(data=" << PrintExpr(op->args[1])
       << ", operand0=" << PrintExpr(op->args[2]) << ", op0=" << nki_op0
       << ", reverse0=" << PrintBool(reverse0) << ", operand1=" << PrintExpr(op->args[3])
       << ", op1=" << nki_op1 << ", reverse1=" << PrintBool(reverse1);
  } else if (is_op(nki_scalar_tensor_scalar_op, "tirx.nki.scalar_tensor_scalar")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 8);
    // nki_scalar_tensor_scalar(result, data, operand0, operand1, opcode0, opcode1, reverse0,
    // reverse1)
    TVM_FFI_ICHECK(opcode_map_.count(op->args[4].as<StringImmNode>()->value));
    std::string nki_op0 = opcode_map_[op->args[4].as<StringImmNode>()->value];
    TVM_FFI_ICHECK(opcode_map_.count(op->args[5].as<StringImmNode>()->value));
    std::string nki_op1 = opcode_map_[op->args[5].as<StringImmNode>()->value];
    bool reverse0 = op->args[6].as<IntImmNode>()->value != 0;
    bool reverse1 = op->args[7].as<IntImmNode>()->value != 0;
    os << PrintExpr(op->args[0]) << " = nisa.tensor_scalar(data=" << PrintExpr(op->args[1])
       << ", operand0=" << PrintExpr(op->args[2]) << ", op0=" << nki_op0
       << ", reverse0=" << PrintBool(reverse0) << ", operand1=" << PrintExpr(op->args[3])
       << ", op1=" << nki_op1 << ", reverse1=" << PrintBool(reverse1);
  } else if (is_op(nki_affine_select_op, "tirx.nki.affine_select")) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 4);
    // nki_affine_select(result, pred, true_value, false_value)
    os << PrintExpr(op->args[0]) << " = nisa.affine_select(pred=" << PrintExpr(op->args[1])
       << ", on_true_tile=" << PrintExpr(op->args[2])
       << ", on_false_value=" << PrintExpr(op->args[3]);
  } else {
    LOG(FATAL) << "Trainium codegen does not support call to " << op->op;
  }
  if (ctx_.mask.defined()) {
    PreOrderVisit(ctx_.mask, [&](const ffi::ObjectRef& node) {
      if (const auto* v = node.as<VarNode>()) {
        if (ctx_.tensorized_loop_vars.count(v)) {
          TVM_FFI_ICHECK(ctx_.loopvar2dim.count(v))
              << "nki_dim must be specified for tensorized loop variables used in mask. However, "
                 "it is not specified for "
              << ffi::GetRef<Var>(v);
          auto dim_str = ctx_.loopvar2dim[v];
          TVM_FFI_ICHECK(dim_str == "P" || dim_str == "F")
              << "Only nki_dim = P or F is allowed for tensorized loop variables used in mask. "
                 "However, "
              << ffi::GetRef<Var>(v) << " has nki_dim = " << dim_str;
        }
      }
      return true;
    });
    os << ", mask=" << PrintExpr(ctx_.mask);
  }
  os << ")";
}

void CodeGenTrainium::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "math.inf";
  } else if (std::isnan(op->value)) {
    LOG(FATAL) << "Trainium codegen does not support NaN";
  } else {
    temp << std::scientific << op->value;
  }
  MarkConst(temp.str());
  os << temp.str();
}

void CodeGenTrainium::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
  if (!ctx_.tensorized_loop_vars.count(op)) {
    // this var is not a tensorized loop variable
    return;
  }
  int total_dim_num, dim;
  if (ctx_.loopvar2dim.count(op)) {
    // nki_dim is specified for this loop variable
    auto dim_str = ctx_.loopvar2dim[op];
    if (dim_str == "P") {
      dim = 0;
    } else if (dim_str == "F" || dim_str == "rhs_F") {
      dim = 1;
    } else if (dim_str == "lhs_F") {
      dim = ctx_.is_matmul_input ? 1 : 0;
    } else {
      LOG(FATAL) << "Invalid nki_dim: " << dim_str;
    }
    total_dim_num = 2;
  } else {
    // nki_dim is not specified for this loop variable
    // we need to use the buffer dimension where the variable appears
    if (ctx_.buffer_index == -1) {
      // this var is not under BufferLoad. We don't know which dim it belongs to.
      return;
    }
    dim = ctx_.buffer_index;
    total_dim_num = ctx_.used_var_cnt;
  }
  os << "[";
  for (int i = 0; i < total_dim_num; i++) {
    if (i == dim) {
      os << ":, ";
    } else {
      os << "None, ";
    }
  }
  os << "]";
  ctx_.buffer_index++;
}

void CodeGenTrainium::VisitExpr_(const CastNode* op, std::ostream& os) {
  ctx_.dst_dtype = op->dtype;
  CodeGenTrainium::VisitExpr(op->value, os);
}

void CodeGenTrainium::VisitExpr_(const FloorDivNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " // " << PrintExpr(op->b);
}

void CodeGenTrainium::VisitExpr_(const FloorModNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " % " << PrintExpr(op->b);
}

void CodeGenTrainium::VisitStmt_(const DeclBufferNode* op) {
  if (op->buffer.scope() == "trn.psum" || op->buffer.scope() == "trn.sbuf") {
    return;
  }
  const VarNode* data = op->buffer->data.get();
  auto it = data_buffer_idmap_.find(data);
  if (it != data_buffer_idmap_.end()) {
    const Buffer& prev_buffer = data_decl_buffer_map_.at(data);
    if (ffi::StructuralEqual()(prev_buffer->shape, op->buffer->shape) &&
        prev_buffer->dtype == op->buffer->dtype) {
      buffer_idmap_[op->buffer] = it->second;
      return;
    }
  }
  std::string data_vid = GetVarID(data);
  std::string buffer_vid = name_supply_->FreshName(data_vid + "_buffer");
  buffer_idmap_[op->buffer] = buffer_vid;
  data_buffer_idmap_[data] = buffer_vid;
  data_decl_buffer_map_[data] = op->buffer;
  PrintIndent();
  stream << buffer_vid << " = " << data_vid << ".reshape(" << PrintShapeAsList(op->buffer->shape)
         << ")\n";
}

ffi::Module BuildTrainium(IRModule mod, Target target) {
  bool output_ssa = false;

  std::ostringstream source_maker;
  std::unordered_map<std::string, std::string> smap;
  static auto fTrainium_compile = ffi::Function::GetGlobal("tvm_callback_Trainium_compile");
  std::string fmt = fTrainium_compile.has_value() ? "Trainiumlib" : "Trainium";

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTrainium: Can only take PrimFunc";
    auto global_symbol = kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();
    source_maker << "# Function: " << func_name << "\n";
    CodeGenTrainium cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    smap[func_name] = fsource;
  }

  return codegen::DeviceSourceModuleCreate(source_maker.str(), fmt, ExtractFuncInfo(mod), "nki");
}

void CodeGenTrainium::VisitStmt_(const IfThenElseNode* op) {
  if (ctx_.tensorizing) {
    TVM_FFI_ICHECK(!op->else_case.defined()) << "Else not allowed in tensorized instruction";
    TVM_FFI_ICHECK(!ctx_.mask.defined()) << "Only one if stmt allowed in tensorized instruction";
    ctx_.mask = op->condition;
    VisitStmt(op->then_case);
    return;
  }
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if " << cond << " :\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);
  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
}

void CodeGenTrainium::VisitExpr_(const AndNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " & " << PrintExpr(op->b);
}

void CodeGenTrainium::VisitExpr_(const OrNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " | " << PrintExpr(op->b);
}

void RegisterTRNCodegen() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.trn", BuildTrainium);
}

}  // namespace codegen
}  // namespace tvm
