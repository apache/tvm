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

class StackSizeChecker : public StmtExprVisitor {
 public:
  struct StackSizes {
    // If a tvm_stack_make_shape call has no arguments, it is still
    // valid and represents a scalar shape ().  Therefore, -1 is used
    // to represent "no shape arguments exist", while 0 represents
    // "shape arguments exist, all of which are size 0".
    int64_t shape_stack{-1};
    uint64_t array_stack{0};
    uint64_t arg_stack{0};
  };

  static StackSizes Check(Stmt stmt) {
    StackSizeChecker visitor;
    visitor.VisitStmt(stmt);
    return visitor.max_stack_;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      // Parallel for loops have their own stack and allocations, so
      // stop the recursion here.
      return;
    } else {
      this->VisitStmt(op->body);
    }
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_call_packed())) {
      return MakeCallPacked(op, /* use_string_lookup */ true);
    } else if (op->op.same_as(builtin::tvm_call_cpacked())) {
      return MakeCallPacked(op, /* use_string_lookup */ false);
    } else if (op->op.same_as(builtin::tvm_call_trace_packed())) {
      return MakeCallTracePacked(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_shape())) {
      return MakeShape(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_array())) {
      return MakeArray(op);
    } else {
      return StmtExprVisitor::VisitExpr_(op);
    }
  }
  // call shape
  void MakeShape(const CallNode* op) {
    // if args.size() == 0, it is still valid and represents a scalar
    // shape ().  Therefore, -1 is used to represent "no shape
    // arguments exist", while 0 represents "shape arguments exist,
    // all of which are size 0".
    if (current_stack_.shape_stack == -1) {
      current_stack_.shape_stack = 0;
    }
    current_stack_.shape_stack += op->args.size();
    StmtExprVisitor::VisitExpr_(op);
  }
  // make array
  void MakeArray(const CallNode* op) {
    current_stack_.array_stack += 1;
    StmtExprVisitor::VisitExpr_(op);
  }
  // call packed.
  void MakeCallPacked(const CallNode* op, bool use_string_lookup) {
    StackSizes restore_stack = current_stack_;

    size_t arg_count = op->args.size();

    // cpacked expects a resource_handle parameter
    if (!use_string_lookup) {
      arg_count--;
    }

    current_stack_.arg_stack += arg_count;
    // Specially handle the buffer packed intrinsic
    StmtExprVisitor::VisitExpr_(op);
    // Record the amount of stack space needed, then reset the stack
    // position to its previous location.
    UpdateMaxStack();
    current_stack_ = restore_stack;
  }

  void MakeCallTracePacked(const CallNode* op) {
    StackSizes restore_stack = current_stack_;

    size_t args_size = op->args.size();
    ICHECK_GT(args_size, 0);
    current_stack_.arg_stack += args_size;

    StmtExprVisitor::VisitExpr_(op);
    // Record the amount of stack space needed, then reset the stack
    // position to its previous location.
    UpdateMaxStack();
    current_stack_ = restore_stack;

    // However, the arguments to this CallNode remain on top of the
    // stack, so we can use more than one packed function's arguments
    // with the one stack.
    current_stack_.arg_stack = restore_stack.arg_stack + args_size - 1;
  }

  void UpdateMaxStack() {
    max_stack_.arg_stack = std::max(current_stack_.arg_stack, max_stack_.arg_stack);
    max_stack_.shape_stack = std::max(current_stack_.shape_stack, max_stack_.shape_stack);
    max_stack_.array_stack = std::max(current_stack_.array_stack, max_stack_.array_stack);
  }

  StackSizes current_stack_;
  StackSizes max_stack_;
};

// Calculate the statistics of packed function.
// These information are needed during codegen.
class BuiltinLower : public StmtExprMutator {
 public:
  // Record stack frame for existing scope.
  struct AllocaScope {
    Buffer stack_shape;
    Var stack_array = Var("stack_array", DataType::Handle());
    Var stack_value = Var("stack_value", DataType::Handle());
    Buffer stack_tcode;

    int64_t max_shape_stack{-1};
    uint64_t max_array_stack{0};
    uint64_t max_arg_stack{0};

    int64_t run_shape_stack{-1};
    uint64_t run_array_stack{0};
    uint64_t run_arg_stack{0};
  };

  Stmt Build(Stmt stmt) { return this->VisitBodyAndRealizeAlloca(stmt); }

  // Allcoate stack frames, only at parallel-for or root.
  Stmt VisitBodyAndRealizeAlloca(Stmt stmt) {
    // Initial check to identify maximum stack sizes.  These are used
    // to construct Buffer objects to hold the stack, which are then
    // used when mutating.
    auto max_sizes = StackSizeChecker::Check(stmt);

    alloca_scope_.emplace_back();
    auto& scope = alloca_scope_.back();

    if (max_sizes.shape_stack != -1) {
      scope.stack_shape = decl_buffer({IntImm(DataType::Int(64), max_sizes.shape_stack)},
                                      DataType::Int(64), "stack_shape");
      stmt = LetStmt(scope.stack_shape->data, StackAlloca("shape", max_sizes.shape_stack), stmt);
    }

    if (max_sizes.array_stack != 0) {
      stmt = LetStmt(scope.stack_array, StackAlloca("array", max_sizes.array_stack), stmt);
    }

    if (max_sizes.arg_stack != 0) {
      scope.stack_tcode = decl_buffer({IntImm(DataType::UInt(64), max_sizes.arg_stack)},
                                      DataType::Int(32), "stack_tcode");
      stmt = LetStmt(scope.stack_value, StackAlloca("arg_value", max_sizes.arg_stack), stmt);

      stmt = LetStmt(scope.stack_tcode->data, StackAlloca("arg_tcode", max_sizes.arg_stack), stmt);
    }

    // Copy these values from the earlier search, for use in bounds
    // checks.
    scope.max_shape_stack = max_sizes.shape_stack;
    scope.max_array_stack = max_sizes.array_stack;
    scope.max_arg_stack = max_sizes.arg_stack;

    stmt = this->VisitStmt(stmt);

    ICHECK(!alloca_scope_.empty());
    alloca_scope_.pop_back();

    return stmt;
  }

  Stmt VisitStmt(const Stmt& s) final {
    // allocate space to hold prepare stmts before s
    prep_seq_stack_.emplace_back(std::vector<Stmt>());

    auto stmt = StmtExprMutator::VisitStmt(s);
    auto& scope = alloca_scope_.back();
    ICHECK_EQ(scope.run_shape_stack, -1);
    ICHECK_EQ(scope.run_array_stack, 0);

    auto prep_seq = std::move(prep_seq_stack_.back());
    prep_seq_stack_.pop_back();

    if (prep_seq.size() != 0) {
      Stmt ret = SeqStmt::Flatten(prep_seq, stmt);
      return ret;
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (const CallNode* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::texture2d_alloca())) {
        return StmtExprMutator::VisitStmt(MakeTextureAlloc(op, call));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    // Lower allocate to device allocate when needed.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    // Get constant allocation bound.
    int64_t nbytes = GetVectorBytes(op->dtype);
    // If the buffers are for CPU and have global scope,
    // and less than runtime::kMaxStackAlloca heuristic
    // they are not serviced with TVMBackendWorkspaceAlloc calls
    // to be placed on stack.
    if (op->annotations.count(transform::kDisableLowerTVMBuiltin)) {
      if (Downcast<Bool>(op->annotations[transform::kDisableLowerTVMBuiltin])) {
        return stmt;
      }
    }
    if (device_type_.defined()) {
      if (const auto* dev_type = device_type_.as<IntImmNode>()) {
        auto storage_scope = Downcast<PointerType>(op->buffer_var->type_annotation)->storage_scope;
        if (dev_type->value == kDLCPU && storage_scope == "global") {
          size_t constant_size = op->ConstantAllocationSize();
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
    if (op->attr_key == attr::device_id) {
      ICHECK(!device_id_.defined());
      device_id_ = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::device_type) {
      ICHECK(!device_type_.defined());
      device_type_ = op->value;
      return this->VisitStmt(op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    Stmt body;

    if (op->kind == ForKind::kParallel) {
      body = this->VisitBodyAndRealizeAlloca(op->body);
    } else {
      body = this->VisitStmt(op->body);
    }

    if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->body = std::move(body);
      return Stmt(n);
    }
  }
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_call_packed())) {
      return MakeCallPacked(op, /* use_string_lookup */ true);
    } else if (op->op.same_as(builtin::tvm_call_cpacked())) {
      return MakeCallPacked(op, /* use_string_lookup */ false);
    } else if (op->op.same_as(builtin::tvm_call_trace_packed())) {
      return MakeCallTracePacked(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_shape())) {
      return MakeShape(op);
    } else if (op->op.same_as(builtin::tvm_stack_make_array())) {
      return MakeArray(op);
    } else if (op->op.same_as(builtin::tvm_context_id())) {
      return make_zero(op->dtype);
    } else if (op->op.same_as(builtin::mem_copy())) {
      return MakeMemCopy(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr MakeMemCopy(const CallNode* op) {
    PrimExpr dst = op->args[0];
    PrimExpr src = op->args[1];
    PrimExpr size = op->args[2];

    std::string fdevapi_prefix =
        "device_api." + std::string(runtime::DeviceName(device_type_.as<IntImmNode>()->value));

    Call call_packed = Call(DataType::Int(32), builtin::tvm_call_packed(),
                            {StringImm(fdevapi_prefix + ".mem_copy"), dst, src, size});
    return VisitExpr(call_packed);
  }

  // call shape
  PrimExpr MakeShape(const CallNode* op) {
    // if args.size() == 0, it represents a scalar shape ()
    ICHECK(!alloca_scope_.empty());
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();
    if (scope.run_shape_stack == -1) {
      scope.run_shape_stack = 0;
    }
    int64_t stack_begin = scope.run_shape_stack;
    scope.run_shape_stack += op->args.size();
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    // no need to perform any store for a scalar shape
    for (size_t i = 0; i < op->args.size(); ++i) {
      prep_seq.emplace_back(BufferStore(scope.stack_shape, cast(DataType::Int(64), op->args[i]),
                                        {ConstInt32(stack_begin + i)}));
    }
    return AddressOffset(scope.stack_shape->data, DataType::Int(64), stack_begin);
  }
  // make array
  PrimExpr MakeArray(const CallNode* op) {
    ICHECK(!alloca_scope_.empty());
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();

    size_t idx = scope.run_array_stack;
    scope.run_array_stack += 1;
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrData, op->args[0]));
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrShape, op->args[1]));
    PrimExpr strides = op->args[2];
    if (!strides.defined() || is_zero(strides)) {
      strides = make_zero(DataType::Handle());
    }
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrStrides, strides));
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrNDim, op->args[3]));
    DataType dtype = op->args[4].dtype();
    prep_seq.emplace_back(
        TVMStructSet(scope.stack_array, idx, builtin::kArrTypeCode,
                     make_const(DataType::UInt(8), static_cast<int>(dtype.code()))));
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrTypeBits,
                                       make_const(DataType::UInt(8), dtype.bits())));
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrTypeLanes,
                                       make_const(DataType::UInt(16), dtype.lanes())));
    // set byte offset
    int data_bytes = GetVectorBytes(dtype);
    PrimExpr byte_offset = op->args[5];
    if (!is_zero(byte_offset)) {
      byte_offset = byte_offset * make_const(byte_offset.dtype(), data_bytes);
    }
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrByteOffset,
                                       cast(DataType::UInt(64), byte_offset)));
    ICHECK(device_type_.defined()) << "Unknown device type in current IR";
    ICHECK(device_id_.defined()) << "Unknown device id in current IR";
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrDeviceId,
                                       cast(DataType::Int(32), device_id_)));
    prep_seq.emplace_back(TVMStructSet(scope.stack_array, idx, builtin::kArrDeviceType,
                                       cast(DataType::Int(32), device_type_)));
    return TVMStructGet(DataType::Handle(), scope.stack_array, idx, builtin::kArrAddr);
  }
  // call packed.
  PrimExpr MakeCallPacked(const CallNode* op, bool use_string_lookup) {
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();

    int64_t restore_shape_stack = scope.run_shape_stack;
    size_t restore_array_stack = scope.run_array_stack;
    size_t arg_stack_begin = scope.run_arg_stack;

    size_t arg_count = op->args.size();

    // cpacked expects a resource_handle parameter
    if (!use_string_lookup) {
      arg_count--;
    }

    scope.run_arg_stack += arg_count;
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < arg_count; ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = Cast(api_type, arg);
      }
      prep_seq.emplace_back(TVMStructSet(scope.stack_value,
                                         static_cast<int>(arg_stack_begin + i - 1),
                                         builtin::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      if (api_type.is_handle() && arg.as<StringImmNode>()) {
        arg_tcode = kTVMStr;
      }
      if (IsArrayHandle(arg)) arg_tcode = kTVMDLTensorHandle;
      prep_seq.emplace_back(BufferStore(scope.stack_tcode, ConstInt32(arg_tcode), {stack_index}));
    }
    // Verify stack size matches earlier value.
    ICHECK_LE(scope.run_arg_stack, scope.max_arg_stack);
    ICHECK_LE(scope.run_shape_stack, scope.max_shape_stack);
    ICHECK_LE(scope.run_array_stack, scope.max_array_stack);
    scope.run_shape_stack = restore_shape_stack;
    scope.run_array_stack = restore_array_stack;
    scope.run_arg_stack = arg_stack_begin;
    Array<PrimExpr> packed_args = {op->args[0], scope.stack_value, scope.stack_tcode->data,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1)};

    // cpacked call resource_handle
    if (!use_string_lookup) {
      tir::Var resource_handle = Downcast<Var>(op->args[arg_count]);
      packed_args.push_back(StringImm(resource_handle->name_hint));
    }

    auto builtin_call = use_string_lookup ? builtin::tvm_call_packed_lowered()
                                          : builtin::tvm_call_cpacked_lowered();
    return Call(op->dtype, builtin_call, packed_args);
  }

  PrimExpr MakeCallTracePacked(const CallNode* op) {
    ICHECK(!alloca_scope_.empty());
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();

    int64_t restore_shape_stack = scope.run_shape_stack;
    size_t restore_array_stack = scope.run_array_stack;
    size_t arg_stack_begin = scope.run_arg_stack;
    scope.run_arg_stack += op->args.size();
    size_t args_size = op->args.size();
    ICHECK_GT(args_size, 0);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = Cast(api_type, arg);
      }
      prep_seq.emplace_back(TVMStructSet(scope.stack_value,
                                         static_cast<int>(arg_stack_begin + i - 1),
                                         builtin::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      ICHECK(!IsArrayHandle(arg)) << "Trace does not support Buffers";
      prep_seq.emplace_back(BufferStore(scope.stack_tcode, ConstInt32(arg_tcode), {stack_index}));
    }
    // Verify stack size matches earlier value.
    ICHECK_LE(scope.run_arg_stack, scope.max_arg_stack);
    ICHECK_LE(scope.run_shape_stack, scope.max_shape_stack);
    ICHECK_LE(scope.run_array_stack, scope.max_array_stack);
    scope.run_shape_stack = restore_shape_stack;
    scope.run_array_stack = restore_array_stack;
    // Update the top of the stack, so we can use more than one
    // packed function's arguments with the one stack.
    scope.run_arg_stack = arg_stack_begin + args_size - 1;
    Array<PrimExpr> packed_args = {op->args[0], scope.stack_value, scope.stack_tcode->data,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1),
                                   // Pass traced value.
                                   op->args[args_size - 1]};
    return Call(op->dtype, builtin::tvm_call_trace_packed_lowered(), packed_args);
  }

  Stmt MakeTextureAlloc(const LetStmtNode* let, const CallNode* call) {
    ICHECK(device_type_.defined()) << "Unknown device type in current IR";
    ICHECK(device_id_.defined()) << "Unknown device id in current IR";
    Stmt throw_last_error = Evaluate(Call(DataType::Int(32), builtin::tvm_throw_last_error(), {}));

    Stmt body = SeqStmt(
        {IfThenElse(Call(DataType::Bool(1), builtin::isnullptr(), {let->var}), throw_last_error),
         let->body});
    DataType dtype =
        let->var->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>()->dtype;

    std::string fdevapi_prefix = "device_api.";
    fdevapi_prefix += runtime::DeviceName(device_type_.as<IntImmNode>()->value);
    Call call_packed =
        Call(let->var.dtype(), builtin::tvm_call_packed(),
             {StringImm(fdevapi_prefix + ".AllocTexture"), cast(DataType::Int(32), device_type_),
              cast(DataType::Int(32), device_id_), cast(DataType::UInt(64), call->args[0]),
              cast(DataType::UInt(64), call->args[1]), IntImm(DataType::Int(32), dtype.code()),
              IntImm(DataType::Int(32), dtype.bits())});

    Stmt alloca = LetStmt(let->var, call_packed, body);

    Call free_op =
        Call(DataType::Int(32), builtin::tvm_call_packed(),
             {StringImm(fdevapi_prefix + ".FreeTexture"), cast(DataType::Int(32), device_type_),
              cast(DataType::Int(32), device_id_), let->var});

    Stmt free_stmt = IfThenElse(free_op != make_zero(DataType::Int(32)), throw_last_error);
    body = SeqStmt({alloca, free_stmt});
    return body;
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

  // The prepration sequence to be emitted before the current statement.
  std::vector<std::vector<Stmt>> prep_seq_stack_;
  PrimExpr device_type_;
  PrimExpr device_id_;

  // Record all stack frames.
  std::vector<AllocaScope> alloca_scope_;
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
