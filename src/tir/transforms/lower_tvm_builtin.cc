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
 *  Lower TVM related builtin intrinsics such as packed call.
 * \file tir/transforms/lower_tvm_buildin.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "ir_utils.h"

namespace tvm {
namespace tir {

inline PrimExpr ConstInt32(size_t index) {
  ICHECK_LE(index, std::numeric_limits<int>::max());
  return make_const(DataType::Int(32), static_cast<int>(index));
}

inline PrimExpr StackAlloca(std::string type, size_t num) {
  Array<PrimExpr> args = {StringImm(type), ConstInt32(num)};
  return Call(DataType::Handle(), builtin::tvm_stack_alloca(), args);
}

// Calculate the statistics of packed function.
// These information are needed during codegen.
class BuiltinLower : public StmtExprMutator {
 public:
  Stmt Build(Stmt stmt) {
    stmt = this->VisitStmt(stmt);
    return stmt;
  }

  Stmt VisitStmt(const Stmt& s) final {
    auto stmt = StmtExprMutator::VisitStmt(s);
    ICHECK_EQ(run_shape_stack_, -1);
    ICHECK_EQ(run_array_stack_, 0);

    if (prep_seq_.size() != 0) {
      stmt = SeqStmt::Flatten(prep_seq_, stmt);
      prep_seq_.clear();
    }

    // Always generated "tvm_stack_alloca" intrincis next to the "tvm_packed_func",
    // which makes the stacks allocated thread-local and every tvm_packed_func will have
    // it's own stack, rather than a shared one. This could help resolve the race
    // -condition issue in parallel execution.

    if (emit_stack_shape_) {
      ICHECK_NE(max_shape_stack_, -1);
      stmt = LetStmt(stack_shape_, StackAlloca("shape", max_shape_stack_), stmt);

      max_shape_stack_ = -1;
      emit_stack_shape_ = false;
    }
    if (emit_stack_array_) {
      ICHECK_NE(max_array_stack_, 0);
      stmt = LetStmt(stack_array_, StackAlloca("array", max_array_stack_), stmt);

      max_array_stack_ = 0;
      emit_stack_array_ = false;
    }
    if (emit_stack_value_tcode_) {
      stmt = LetStmt(stack_value_, StackAlloca("arg_value", arg_stack_size_), stmt);
      stmt = LetStmt(stack_tcode_, StackAlloca("arg_tcode", arg_stack_size_), stmt);

      emit_stack_value_tcode_ = false;
      arg_stack_size_ = 0;
    }

    return stmt;
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    // Lower allocate to device allocate when needed.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    // Get constant allocation bound.
    int64_t nbytes = GetVectorBytes(op->dtype);
    if (device_type_.defined()) {
      if (const auto* dev_type = device_type_.as<IntImmNode>()) {
        if (dev_type->value == kDLCPU) {
          int32_t constant_size = op->constant_allocation_size();
          if (constant_size > 0 && constant_size * nbytes < runtime::kMaxStackAlloca) {
            return stmt;
          }
        }
      }
    }
    PrimExpr total_bytes = make_const(op->extents[0].dtype(), nbytes);
    for (size_t i = 0; i < op->extents.size(); ++i) {
      total_bytes = total_bytes * op->extents[i];
    }
    ICHECK(device_type_.defined()) << "Unknown device type in current IR";
    ICHECK(device_id_.defined()) << "Unknown device id in current IR";
    Stmt throw_last_error = Evaluate(Call(DataType::Int(32), builtin::tvm_throw_last_error(), {}));

    Stmt body = SeqStmt({IfThenElse(Call(DataType::Bool(1), builtin::isnullptr(), {op->buffer_var}),
                                    throw_last_error),
                         op->body});
    Stmt alloca = LetStmt(
        op->buffer_var,
        Call(op->buffer_var.dtype(), Op::Get("tir.TVMBackendAllocWorkspace"),
             {cast(DataType::Int(32), device_type_), cast(DataType::Int(32), device_id_),
              cast(DataType::UInt(64), total_bytes), IntImm(DataType::Int(32), op->dtype.code()),
              IntImm(DataType::Int(32), op->dtype.bits())}),
        body);

    PrimExpr free_op = Call(DataType::Int(32), Op::Get("tir.TVMBackendFreeWorkspace"),
                            {cast(DataType::Int(32), device_type_),
                             cast(DataType::Int(32), device_id_), op->buffer_var});
    Stmt free_stmt = IfThenElse(free_op != make_zero(DataType::Int(32)), throw_last_error);
    body = SeqStmt({alloca, free_stmt});
    body = AttrStmt(op->buffer_var, attr::storage_alignment,
                    make_const(DataType::Int(32), runtime::kTempAllocaAlignment), body);
    return body;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_id) {
      ICHECK(!device_id_.defined());
      device_id_ = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::device_context_type) {
      ICHECK(!device_type_.defined());
      device_type_ = op->value;
      return this->VisitStmt(op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_call_packed())) {
      return MakeCallPacked(op);
    } else if (op->op.same_as(builtin::tvm_call_trace_packed())) {
      return MakeCallTracePacked(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_shape())) {
      return MakeShape(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_array())) {
      return MakeArray(op);
    } else if (op->op.same_as(builtin::tvm_context_id())) {
      return make_zero(op->dtype);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  std::string GetUniqueName(std::string prefix) {
    for (size_t i = 0; i < prefix.size(); ++i) {
      if (prefix[i] == '.') prefix[i] = '_';
    }
    auto it = name_alloc_map_.find(prefix);
    if (it != name_alloc_map_.end()) {
      while (true) {
        std::ostringstream os;
        os << prefix << (++it->second);
        std::string name = os.str();
        if (name_alloc_map_.count(name) == 0) {
          prefix = name;
          break;
        }
      }
    }
    name_alloc_map_[prefix] = 0;
    return prefix;
  }
  // call shape
  PrimExpr MakeShape(const CallNode* op) {
    // if args.size() == 0, it represents a scalar shape ()
    if (run_shape_stack_ == -1) {
      run_shape_stack_ = 0;
    }
    int64_t stack_begin = run_shape_stack_;
    run_shape_stack_ += op->args.size();
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    // Unlike stack_value and stack_tcode, stack_shape is allowed to be
    // shared in the same stmt.
    //
    // For example:
    // >  @tir.tvm_call_packed("tvm.contrib.cblas.matmul", @tir.tvm_stack_make_array(A,
    // >                 @tvm_stack_make_shape(...), ...), @tir.tvm_stack_make_array(B,
    // >                 @tvm_stack_make_shape(...), ...), ...)
    // In the stmt above, those tvm_stack_make_shape won't be executed parallel, thus it's ok for
    // them to share a same stack.
    //
    // To let all tvm_stack_make_array in the same stmt share a stack, we check "emit_stack_shape_":
    // 1. false: This is the first occurrence of tvm_stack_make_shape expr of current stmt. A new
    // stack_shape_ is generated, and mark emit_stack_shape_ as true;
    // 2. true:  Not the first occurrence of current stmt. Just reuse the previous stack_shape_;
    if (!emit_stack_shape_) {
      stack_shape_ = Var(GetUniqueName("stack_shape"), DataType::Handle());
      emit_stack_shape_ = true;
    }

    // no need to perform any store for a scalar shape
    for (size_t i = 0; i < op->args.size(); ++i) {
      prep_seq_.emplace_back(Store(stack_shape_, cast(DataType::Int(64), op->args[i]),
                                   ConstInt32(stack_begin + i), const_true(1)));
    }
    return AddressOffset(stack_shape_, DataType::Int(64), stack_begin);
  }
  // make array
  PrimExpr MakeArray(const CallNode* op) {
    size_t idx = run_array_stack_;
    run_array_stack_ += 1;
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    // Unlike stack_value and stack_tcode, stack_array is allowed to be
    // shared in the same stmt.
    //
    // For example:
    // >  @tir.tvm_call_packed("tvm.contrib.cblas.matmul", @tir.tvm_stack_make_array(A,
    // >                 @tvm_stack_make_shape(...), ...), @tir.tvm_stack_make_array(B,
    // >                 @tvm_stack_make_shape(...), ...), ...)
    // In the stmt above, those tvm_stack_make_shape won't be executed parallel, thus it's ok for
    // them to share a same stack.
    //
    // To let all tvm_stack_make_array in the same stmt share a stack, we check "emit_stack_array_":
    // 1. false: This is the first occurrence of tvm_stack_make_shape expr of current stmt. A new
    // stack_array_ is generated, and we mark emit_stack_array_ as true;
    // 2. true:  Not the first occurrence of current stmt. Just reuse the previous stack_array_;
    if (!emit_stack_array_) {
      stack_array_ = Var(GetUniqueName("stack_array"), DataType::Handle());
      emit_stack_array_ = true;
    }

    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrData, op->args[0]));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrShape, op->args[1]));
    PrimExpr strides = op->args[2];
    if (!strides.defined() || is_zero(strides)) {
      strides = make_zero(DataType::Handle());
    }
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrStrides, strides));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrNDim, op->args[3]));
    DataType dtype = op->args[4].dtype();
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, builtin::kArrTypeCode,
                     make_const(DataType::UInt(8), static_cast<int>(dtype.code()))));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrTypeBits,
                                        make_const(DataType::UInt(8), dtype.bits())));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrTypeLanes,
                                        make_const(DataType::UInt(16), dtype.lanes())));
    // set byte offset
    int data_bytes = GetVectorBytes(dtype);
    PrimExpr byte_offset = op->args[5];
    if (!is_zero(byte_offset)) {
      byte_offset = byte_offset * make_const(byte_offset.dtype(), data_bytes);
    }
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrByteOffset,
                                        cast(DataType::UInt(64), byte_offset)));
    ICHECK(device_type_.defined()) << "Unknown device type in current IR";
    ICHECK(device_id_.defined()) << "Unknown device id in current IR";
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrDeviceId,
                                        cast(DataType::Int(32), device_id_)));
    prep_seq_.emplace_back(TVMStructSet(stack_array_, idx, builtin::kArrDeviceType,
                                        cast(DataType::Int(32), device_type_)));
    return TVMStructGet(DataType::Handle(), stack_array_, idx, builtin::kArrAddr);
  }
  // call packed.
  PrimExpr MakeCallPacked(const CallNode* op) {
    int64_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    // stack_value_ and stack_tcode_ pair are generated for call packed; Use "GetUniqueName()" is
    // to fullfill the SSA constrains.
    ICHECK_EQ(emit_stack_value_tcode_, false);
    emit_stack_value_tcode_ = true;
    stack_value_ = Var(GetUniqueName("stack_value"), DataType::Handle());
    stack_tcode_ = Var(GetUniqueName("stack_tcode"), DataType::Handle());

    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = Cast(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(stack_value_, static_cast<int>(arg_stack_begin + i - 1),
                                          builtin::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      if (api_type.is_handle() && arg.as<StringImmNode>()) {
        arg_tcode = kTVMStr;
      }
      if (IsArrayHandle(arg)) arg_tcode = kTVMDLTensorHandle;
      prep_seq_.emplace_back(
          Store(stack_tcode_, ConstInt32(arg_tcode), stack_index, const_true(1)));
    }
    // UPDATE stack value
    arg_stack_size_ = run_shape_stack_;
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    run_arg_stack_ = arg_stack_begin;
    Array<PrimExpr> packed_args = {op->args[0], stack_value_, stack_tcode_,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1)};
    return Call(DataType::Int(32), builtin::tvm_call_packed_lowered(), packed_args);
  }

  PrimExpr MakeCallTracePacked(const CallNode* op) {
    int64_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    size_t args_size = op->args.size();
    ICHECK_GT(args_size, 0);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    // stack_value_ and stack_tcode_ pair are generated for call packed; Use "GetUniqueName()" is
    // to fullfill the SSA constrains.
    ICHECK_EQ(emit_stack_value_tcode_, false);
    emit_stack_value_tcode_ = true;
    stack_value_ = Var(GetUniqueName("stack_value"), DataType::Handle());
    stack_tcode_ = Var(GetUniqueName("stack_tcode"), DataType::Handle());

    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = Cast(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(stack_value_, static_cast<int>(arg_stack_begin + i - 1),
                                          builtin::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      ICHECK(!IsArrayHandle(arg)) << "Trace does not support Buffers";
      prep_seq_.emplace_back(
          Store(stack_tcode_, ConstInt32(arg_tcode), stack_index, const_true(1)));
    }
    // UPDATE stack value
    arg_stack_size_ = run_shape_stack_;
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    // Update the top of the stack, so we can use more than one
    // packed function's arguments with the one stack.
    run_arg_stack_ = arg_stack_begin + args_size - 1;
    Array<PrimExpr> packed_args = {op->args[0], stack_value_, stack_tcode_,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1),
                                   // Pass traced value.
                                   op->args[args_size - 1]};
    return Call(op->dtype, builtin::tvm_call_trace_packed_lowered(), packed_args);
  }

 private:
  bool IsArrayHandle(const PrimExpr& arg) {
    // specially set array handle.
    if (const CallNode* buf = arg.as<CallNode>()) {
      if (buf->op.same_as(builtin::tvm_struct_get()) &&
          buf->args[2].as<IntImmNode>()->value == builtin::kArrAddr) {
        return true;
      }
    }
    return false;
  }

  // The prepration sequence to be emitted.
  std::vector<Stmt> prep_seq_;
  PrimExpr device_type_;
  PrimExpr device_id_;
  Var stack_shape_;
  Var stack_array_;
  Var stack_tcode_;
  Var stack_value_;

  // Mark the occurence of tvm_stack_make_shape of current stmt:
  // 1. Set to true when the first tvm_stack_make_shape is met;
  // 2. Reset to false at the end of VisitStmt();
  bool emit_stack_shape_{false};

  // Mark the occurence of tvm_stack_make_array of current stmt:
  // 1. Set to true when the first tvm_stack_make_array is met;
  // 2. Reset to false at the end of VisitStmt().
  bool emit_stack_array_{false};

  // Mark the occurence of tvm_call_packed of current stmt:
  // 1. Set to true when tvm_call_packed intrinsic is met;
  // 2. Reset to false at the end of VisitStmt().
  bool emit_stack_value_tcode_{false};

  // The running statistics
  int64_t run_shape_stack_{-1};
  uint64_t run_array_stack_{0};
  uint64_t run_arg_stack_{0};
  // statistics of stacks
  int64_t max_shape_stack_{-1};
  uint64_t max_array_stack_{0};
  // Size of current pack func's arg stack
  uint64_t arg_stack_size_;
  // Name allocation map
  std::unordered_map<std::string, int> name_alloc_map_;
};

namespace transform {

Pass LowerTVMBuiltin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BuiltinLower().Build(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerTVMBuiltin", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerTVMBuiltin").set_body_typed(LowerTVMBuiltin);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
