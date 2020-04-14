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
 * \file codegen_stackvm.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/ir/module.h>
#include <tvm/tir/op.h>
#include <tvm/tir/function.h>
#include <limits>
#include <utility>
#include "codegen_stackvm.h"
#include "../../runtime/stackvm/stackvm_module.h"

namespace tvm {
namespace codegen {

using namespace tir;

// map struct field kind to runtime variants
// We keep two separate enums to ensure runtime/compiler isolation.
StackVM::StructFieldKind MapFieldKind(int64_t kind) {
  auto val = static_cast<intrinsic::TVMStructFieldKind>(kind);
  switch (val) {
    case intrinsic::kArrData: return StackVM::kArrData;
    case intrinsic::kArrShape: return StackVM::kArrShape;
    case intrinsic::kArrAddr: return StackVM::kArrAddr;
    case intrinsic::kArrStrides: return StackVM::kArrStrides;
    case intrinsic::kArrNDim: return StackVM::kArrNDim;
    case intrinsic::kArrTypeCode: return StackVM::kArrTypeCode;
    case intrinsic::kArrTypeBits: return StackVM::kArrTypeBits;
    case intrinsic::kArrTypeLanes: return StackVM::kArrTypeLanes;
    case intrinsic::kArrByteOffset: return StackVM::kArrByteOffset;
    case intrinsic::kArrDeviceId: return StackVM::kArrDeviceId;
    case intrinsic::kArrDeviceType: return StackVM::kArrDeviceType;
    case intrinsic::kTVMValueContent: return StackVM::kTVMValueContent;
    default: LOG(FATAL) << "Do not know how to map field " << kind;
  }
  return StackVM::kArrData;
}

StackVM CodeGenStackVM::Compile(const PrimFunc& f) {
  CHECK_EQ(f->buffer_map.size(), 0U)
      << "Cannot codegen function with buffer_map, please lower them first";
  for (size_t i = 0; i < f->params.size(); ++i) {
    Var v = f->params[i];
    int vid = AllocVarID(v.get());
    CHECK_EQ(static_cast<size_t>(vid), i);
  }
  this->Push(f->body);
  vm_.InitCache();
  return std::move(vm_);
}

void CodeGenStackVM::Push(const Stmt& n) {
  VisitStmt(n);
  if (debug_) {
    this->PushOp(StackVM::ASSERT_SP, 0);
  }
}

void CodeGenStackVM::PushOp(StackVM::OpCode opcode) {
  StackVM::Code code;
  code.op_code = opcode;
  vm_.code.push_back(code);
}

void CodeGenStackVM::SetOperand(int64_t operand_index, int64_t operand) {
  CHECK(operand >= std::numeric_limits<int>::min() &&
        operand <= std::numeric_limits<int>::max());
  vm_.code.at(operand_index).v_int = static_cast<int>(operand);
}

int64_t CodeGenStackVM::PushOp(StackVM::OpCode opcode, int operand) {
  int64_t pc = static_cast<int64_t>(vm_.code.size());
  StackVM::Code code;
  code.op_code = opcode;
  vm_.code.push_back(code);
  code.v_int = operand;
  vm_.code.push_back(code);
  return pc + 1;
}

int CodeGenStackVM::GetStrID(const std::string& key) {
  auto it = str_idmap_.find(key);
  if (it != str_idmap_.end()) return it->second;
  int sid = static_cast<int>(vm_.str_data.size());
  vm_.str_data.push_back(key);
  str_idmap_[key] = sid;
  return sid;
}

int CodeGenStackVM::AllocVarID(const VarNode* v) {
  CHECK(!var_idmap_.count(v));
  int vid = static_cast<int>(vm_.heap_size);
  CHECK_EQ(vm_.heap_size, var_idmap_.size());
  vm_.heap_id_name.push_back(v->name_hint);
  ++vm_.heap_size;
  var_idmap_[v] = vid;
  return vid;
}

int CodeGenStackVM::GetVarID(const VarNode* v) const {
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
      << "Find undefined Variable " << v->name_hint;
  return it->second;
}

void CodeGenStackVM::VisitExpr_(const LoadNode* op) {
  this->Push(op->buffer_var);
  StackVM::OpCode code = StackVM::GetLoad(op->dtype);
  if (const IntImmNode* index = op->index.as<IntImmNode>()) {
    this->PushOp(code, index->value);
  } else {
    this->Push(op->index);
    this->PushOp(StackVM::PUSH_I64, op->dtype.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
    this->PushOp(code, 0);
  }
}

void CodeGenStackVM::VisitStmt_(const StoreNode* op) {
  this->Push(op->buffer_var);
  StackVM::OpCode code = StackVM::GetStore(op->value.dtype());
  if (const IntImmNode* index = op->index.as<IntImmNode>()) {
    this->Push(op->value);
    this->PushOp(code, index->value);
  } else {
    this->Push(op->index);
    this->PushOp(StackVM::PUSH_I64, op->value.dtype().element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
    this->Push(op->value);
    this->PushOp(code, 0);
  }
}

void CodeGenStackVM::VisitStmt_(const AllocateNode* op) {
  LOG(FATAL) << "Dynamic allocation not supported";
}

void CodeGenStackVM::VisitExpr_(const CallNode* op) {
  if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const LoadNode *l = op->args[0].as<LoadNode>();
    CHECK(op->args.size() == 1 && l);
    this->PushOp(StackVM::LOAD_HEAP, GetVarID(l->buffer_var.get()));
    this->Push(l->index);
    this->PushOp(StackVM::PUSH_I64, l->dtype.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
  } else if (op->is_intrinsic(CallNode::reinterpret)) {
    this->Push(op->args[0]);
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImmNode>()->value;
    this->Push(op->args[0]);
    const IntImmNode* index = op->args[1].as<IntImmNode>();
    CHECK(index != nullptr);
    StackVM::Code code;
    code.op_code = StackVM::TVM_STRUCT_GET;
    vm_.code.push_back(code);
    code.v_int = index->value;
    vm_.code.push_back(code);
    code.v_int = MapFieldKind(kind);
    vm_.code.push_back(code);
  } else if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    CHECK_GE(op->args.size(), 5U);
    const StringImmNode* s = op->args[0].as<StringImmNode>();
    CHECK(s != nullptr) << "tvm_call_global expect first argument as function name";
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    int begin = op->args[3].as<IntImmNode>()->value;
    int end = op->args[4].as<IntImmNode>()->value;
    // find the fuction id.
    const std::string& func_name = s->value;
    auto it = extern_fun_idmap_.find(func_name);
    int fid;
    if (it != extern_fun_idmap_.end()) {
      fid = it->second;
    } else {
      fid = static_cast<int>(vm_.extern_func_name.size());
      vm_.extern_func_name.push_back(func_name);
      extern_fun_idmap_[func_name] = fid;
    }
    // CALL_PACKED_FUNC
    StackVM::Code code;
    code.op_code = StackVM::CALL_PACKED_LOWERED;
    vm_.code.push_back(code);
    code.v_int = fid;
    vm_.code.push_back(code);
    code.v_int = begin;
    vm_.code.push_back(code);
    code.v_int = end;
    vm_.code.push_back(code);
  } else if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
    CHECK_EQ(op->args.size(), 2U);
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    CHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(DLTensor) == 0, "invariant");
    // static_assert(alignof(TVMValue) % alignof(tvm_index_t) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    // add stack size to be safe.
    vm_.stack_size += size;
    this->PushOp(StackVM::TVM_STACK_ALLOCA_BY_8BYTE, static_cast<int>(size));
  } else if (op->name == "TVMBackendAllocWorkspace") {
    CHECK_EQ(op->args.size(), 5U);
    this->Push(op->args[0]);
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    this->Push(op->args[3]);
    this->Push(op->args[4]);
    this->PushOp(StackVM::TVM_DEVICE_ALLOCA);
  } else if (op->name == "TVMBackendFreeWorkspace") {
    CHECK_EQ(op->args.size(), 3U);
    this->Push(op->args[0]);
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    this->PushOp(StackVM::TVM_DEVICE_FREE);
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    this->PushOp(StackVM::TVM_THROW_LAST_ERROR);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    this->Push(op->args[0]);
    this->PushOp(StackVM::PUSH_I64, 0);
    this->PushOp(StackVM::EQ_HANDLE);
  } else {
    LOG(FATAL) << "unknown function call " << op->name;
  }
}

void CodeGenStackVM::PushBinary(StackVM::OpCode op_int64,
                                const PrimExpr& a,
                                const PrimExpr& b) {
  this->Push(a);
  this->Push(b);
  DataType t = a.dtype();
  if (t.is_int()) {
    this->PushOp(op_int64);
  } else if (t.is_uint()) {
    this->PushOp(op_int64);
  } else {
    this->PushOp(StackVM::CodeI64ToF64(op_int64));
  }
}

void CodeGenStackVM::PushCast(DataType dst, DataType src) {
  if (dst.is_int()) {
    if (src.is_int() || src.is_uint()) return;
  } else if (dst.is_uint()) {
    if (src.is_int() || src.is_uint()) return;
  } else if (dst.is_float()) {
    if (src.is_float()) return;
  }
}

void CodeGenStackVM::VisitExpr_(const StringImmNode* op) {
  int sid = this->GetStrID(op->value);
  this->PushOp(StackVM::PUSH_I64, sid);
}

void CodeGenStackVM::VisitExpr_(const IntImmNode* op) {
  CHECK(op->value >= std::numeric_limits<int>::min() &&
        op->value <= std::numeric_limits<int>::max())
      << "Int constant exceed bound";
    this->PushOp(StackVM::PUSH_I64, static_cast<int>(op->value));
}

void CodeGenStackVM::VisitExpr_(const FloatImmNode* op) {
  LOG(FATAL) << "Float Imm is not supported";
}

void CodeGenStackVM::VisitExpr_(const VarNode* op) {
  int vid = this->GetVarID(op);
  this->PushOp(StackVM::LOAD_HEAP, vid);
}

void CodeGenStackVM::VisitExpr_(const CastNode* op) {
  this->Push(op->value);
  PushCast(op->dtype, op->value.dtype());
}

void CodeGenStackVM::VisitExpr_(const AddNode* op) {
  PushBinary(StackVM::ADD_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const SubNode* op) {
  PushBinary(StackVM::SUB_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const MulNode* op) {
  PushBinary(StackVM::MUL_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const DivNode* op) {
  PushBinary(StackVM::DIV_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const ModNode* op) {
  PushBinary(StackVM::MOD_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const MinNode* op) {
  this->Push(op->a);
  this->Push(op->b);
  this->PushOp(StackVM::PUSH_VALUE, -1);
  this->PushOp(StackVM::PUSH_VALUE, -1);
  this->PushOp(StackVM::LT_I64);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitExpr_(const MaxNode* op) {
  this->Push(op->a);
  this->Push(op->b);
  this->PushOp(StackVM::PUSH_VALUE, 0);
  this->PushOp(StackVM::PUSH_VALUE, -2);
  this->PushOp(StackVM::LT_I64);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitExpr_(const EQNode* op) {
  PushBinary(StackVM::EQ_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const LENode* op) {
  PushBinary(StackVM::LE_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const NENode* op) {
  PushBinary(StackVM::EQ_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const LTNode* op) {
  PushBinary(StackVM::LT_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const GENode* op) {
  PushBinary(StackVM::LT_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const GTNode* op) {
  PushBinary(StackVM::LE_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const AndNode* op) {
  this->Push(op->a);
  int64_t pc_jump = this->GetPC();
  int64_t opr_index = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->b);
  int64_t diff = this->GetPC() - pc_jump;
  this->SetOperand(opr_index, diff);
}

void CodeGenStackVM::VisitExpr_(const OrNode* op) {
  this->Push(op->a);
  int64_t pc_jump = this->GetPC();
  int64_t opr_index = this->PushOp(StackVM::RJUMP_IF_TRUE, 0);
  this->Push(op->b);
  int64_t diff = this->GetPC() - pc_jump;
  this->SetOperand(opr_index, diff);
}

void CodeGenStackVM::VisitExpr_(const NotNode* op) {
  this->Push(op->a);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitStmt_(const ForNode* op) {
  CHECK(is_zero(op->min));
  int vid = this->AllocVarID(op->loop_var.get());
  this->PushOp(StackVM::PUSH_I64, 0);
  int64_t loop_head = this->GetPC();
  this->PushOp(StackVM::STORE_HEAP, vid);
  this->PushOp(StackVM::LOAD_HEAP, vid);
  this->Push(op->extent);
  this->PushOp(StackVM::LT_I64);
  int64_t label_fjump = this->GetPC();
  int64_t foward_jump = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->body);
  this->PushOp(StackVM::LOAD_HEAP, vid);
  this->PushOp(StackVM::PUSH_I64, 1);
  this->PushOp(StackVM::ADD_I64);
  int64_t label_bjump = this->GetPC();
  int64_t backward_jump = this->PushOp(StackVM::RJUMP, 0);
  int64_t loop_end = this->GetPC();
  this->PushOp(StackVM::POP);
  this->SetOperand(foward_jump, loop_end - label_fjump);
  this->SetOperand(backward_jump, loop_head - label_bjump);
}

void CodeGenStackVM::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    this->Push(stmt);
  }
}

void CodeGenStackVM::VisitStmt_(const EvaluateNode *ev) {
  if (is_const(ev->value)) return;
  const CallNode* op = ev->value.as<CallNode>();
  if (op && op->is_intrinsic(intrinsic::tvm_struct_set)) {
    CHECK_EQ(op->args.size(), 4U);
    this->Push(op->args[0]);
    this->Push(op->args[3]);
    const IntImmNode* index = op->args[1].as<IntImmNode>();
    CHECK(index != nullptr);
    StackVM::Code code;
    code.op_code = StackVM::TVM_STRUCT_SET;
    vm_.code.push_back(code);
    code.v_int = index->value;
    vm_.code.push_back(code);
    code.v_int = MapFieldKind(op->args[2].as<IntImmNode>()->value);
    vm_.code.push_back(code);
  } else {
    this->Push(ev->value);
    this->PushOp(StackVM::POP);
  }
}

void CodeGenStackVM::VisitStmt_(const IfThenElseNode* op) {
  this->Push(op->condition);
  int64_t label_ejump = this->GetPC();
  int64_t else_jump = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->then_case);
  if (op->else_case.defined()) {
    int64_t label_then_jump = this->GetPC();
    int64_t then_jump = this->PushOp(StackVM::RJUMP, 0);
    int64_t else_begin = this->GetPC();
    this->SetOperand(else_jump, else_begin - label_ejump);
    this->PushOp(StackVM::POP);
    this->Push(op->else_case);
    int64_t if_end = this->GetPC();
    this->SetOperand(then_jump, if_end - label_then_jump);
  } else {
    int64_t if_end = this->GetPC();
    this->SetOperand(else_jump, if_end - label_ejump);
    this->PushOp(StackVM::POP);
  }
}

void CodeGenStackVM::VisitStmt_(const LetStmtNode* op) {
  this->Push(op->value);
  int64_t vid = this->AllocVarID(op->var.get());
  this->PushOp(StackVM::STORE_HEAP, static_cast<int>(vid));
  this->Push(op->body);
}

void CodeGenStackVM::VisitExpr_(const RampNode* op) {
  LOG(FATAL) << "Ramp is not supported";
}

void CodeGenStackVM::VisitExpr_(const BroadcastNode* op) {
  LOG(FATAL) << "Broadcast is not supported";
}

void CodeGenStackVM::VisitExpr_(const SelectNode* op) {
  this->Push(op->true_value);
  this->Push(op->false_value);
  this->Push(op->condition);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitStmt_(const AssertStmtNode* op) {
  if (const auto* str = op->message.as<StringImmNode>()) {
    int sid = this->GetStrID(str->value);
    this->Push(op->condition);
    this->PushOp(StackVM::ASSERT, sid);
  }
  this->Push(op->body);
}

void CodeGenStackVM::VisitStmt_(const AttrStmtNode* op) {
  this->Push(op->body);
}

void CodeGenStackVM::VisitExpr_(const LetNode* op) {
  this->Push(op->value);
  int64_t vid = this->AllocVarID(op->var.get());
  this->PushOp(StackVM::STORE_HEAP, static_cast<int>(vid));
  this->Push(op->body);
}

runtime::Module BuildStackVM(const IRModule& mod) {
  std::unordered_map<std::string, StackVM> fmap;
  std::string entry_func;

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenStackVM: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(global_symbol.defined())
        << "CodeGenStackVM: Expect PrimFunc to have the global_symbol attribute";
    std::string f_name = global_symbol.value();
    StackVM vm = codegen::CodeGenStackVM().Compile(f);
    CHECK(!fmap.count(f_name))
        << "Function name " << f_name << "already exist in list";
    fmap[f_name] = std::move(vm);

    if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
      entry_func = f_name;
    }
  }

  return runtime::StackVMModuleCreate(fmap, entry_func);
}

TVM_REGISTER_GLOBAL("target.build.stackvm")
.set_body_typed(BuildStackVM);
}  // namespace codegen
}  // namespace tvm
