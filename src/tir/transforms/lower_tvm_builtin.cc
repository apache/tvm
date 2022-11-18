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

// Calculate the statistics of packed function.
// These information are needed during codegen.
class BuiltinLower : public StmtExprMutator {
 public:
  // NOTE: Right now, we make the following scoping requirement
  // for memory allocated by the following primitives
  // - tvm_stack_make_array
  // - tvm_stack_make_shape
  // - arg stack
  //
  // Scoping and liveness rules:
  // - Every call_packed introduce a new scope.
  // - The memory allocated by tvm_stack_make_array/make_shape will
  //   no longer become valid outside the scope (and may be reused by
  //   subsequent call_packed.
  // - TODO(tvm-team): we might consider a root scope so stack_make_shape
  //   can be called out-side call_packed.
  //
  //  Example:
  //  {
  //    call_packed(make_shape1(...),
  //                call_packed(make_shape2(...))
  //    call_packed(make_shape3(...))
  //  }
  //
  //  In this case, make_shape1 and make_shape2 should not share memory,
  //  but they can share memory with make_shape3.
  //
  //  Rationale: most of the packed calls needs their own internal
  //  argument stack, and those stack can be shared across calls.
  //  Scoping is a quick way to enable sharing without having
  //  to do full-scale liveness analysis and it does its job.
  //  Alternative approaches can also be used.
  struct StackSizes {
    // If a tvm_stack_make_shape call has no arguments, it is still
    // valid and represents a scalar shape ().  Therefore, -1 is used
    // to represent "no shape arguments exist", while 0 represents
    // "shape arguments exist, all of which are size 0".
    int64_t shape_stack{-1};
    uint64_t array_stack{0};
    uint64_t arg_stack{0};
  };

  // Record stack frame for existing scope.
  struct AllocaScope {
    Buffer stack_shape;
    Var stack_array = Var("stack_array", DataType::Handle());
    Var stack_value = Var("stack_value", DataType::Handle());
    Buffer stack_tcode;

    StackSizes max_sizes;
    StackSizes run_sizes;

    void UpdateMax() {
      max_sizes.shape_stack = std::max(max_sizes.shape_stack, run_sizes.shape_stack);
      max_sizes.array_stack = std::max(max_sizes.array_stack, run_sizes.array_stack);
      max_sizes.arg_stack = std::max(max_sizes.arg_stack, run_sizes.arg_stack);
    }

    void AssertMaxIsValid() const {
      ICHECK((max_sizes.shape_stack >= run_sizes.shape_stack) ||
             (max_sizes.array_stack >= run_sizes.array_stack) ||
             (max_sizes.arg_stack >= run_sizes.arg_stack));
    }
  };

  Stmt Build(Stmt stmt) { return this->VisitBodyAndRealizeAlloca(stmt); }

  StackSizes GetMaxStack(Stmt stmt) {
    BuiltinLower precheck;
    precheck.is_precheck_ = true;
    precheck.device_id_ = this->device_id_;
    precheck.device_type_ = this->device_type_;

    precheck.alloca_scope_.emplace_back();
    {
      // NOTE: this scope reference is invalid after any mutation is applied to alloca_scope_.
      auto& scope = precheck.alloca_scope_.back();
      scope.stack_shape =
          decl_buffer({IntImm(DataType::Int(64), 0)}, DataType::Int(64), "stack_shape");
      scope.stack_tcode =
          decl_buffer({IntImm(DataType::UInt(64), 0)}, DataType::Int(32), "stack_tcode");
    }

    precheck.VisitStmt(stmt);

    ICHECK_EQ(precheck.alloca_scope_.size(), 1);
    return precheck.alloca_scope_[0].max_sizes;
  }

  // Allcoate stack frames, only at parallel-for or root.
  Stmt VisitBodyAndRealizeAlloca(Stmt stmt) {
    // Only perform the precheck up to the point where we would add a
    // new scope.
    if (is_precheck_) {
      return stmt;
    }

    alloca_scope_.emplace_back();
    {
      // NOTE: this scope reference is invalid after any mutation is applied to alloca_scope_.
      auto& scope = alloca_scope_.back();

      // Initial check to identify maximum stack sizes.  These are used
      // to construct Buffer objects to hold the stack, which are then
      // used when mutating.
      scope.max_sizes = GetMaxStack(stmt);

      if (scope.max_sizes.shape_stack != -1) {
        scope.stack_shape = decl_buffer({IntImm(DataType::Int(64), scope.max_sizes.shape_stack)},
                                        DataType::Int(64), "stack_shape");
        stmt = LetStmt(scope.stack_shape->data, StackAlloca("shape", scope.max_sizes.shape_stack),
                       stmt);
      }

      if (scope.max_sizes.array_stack != 0) {
        stmt = LetStmt(scope.stack_array, StackAlloca("array", scope.max_sizes.array_stack), stmt);
      }

      if (scope.max_sizes.arg_stack != 0) {
        scope.stack_tcode = decl_buffer({IntImm(DataType::UInt(64), scope.max_sizes.arg_stack)},
                                        DataType::Int(32), "stack_tcode");
        stmt =
            LetStmt(scope.stack_value, StackAlloca("arg_value", scope.max_sizes.arg_stack), stmt);

        stmt = LetStmt(scope.stack_tcode->data, StackAlloca("arg_tcode", scope.max_sizes.arg_stack),
                       stmt);
      }
    }

    stmt = this->VisitStmt(stmt);

    ICHECK(!alloca_scope_.empty());
    alloca_scope_.pop_back();

    return stmt;
  }

  Stmt VisitStmt(const Stmt& s) final {
    // allocate space to hold prepare stmts before s
    prep_seq_stack_.emplace_back(std::vector<Stmt>());

    auto scope_size = alloca_scope_.size();
    auto stmt = StmtExprMutator::VisitStmt(s);
    {
      // NOTE: this scope reference is invalid after any mutation is applied to alloca_scope_.
      auto& scope = alloca_scope_.back();
      // This invariant asserts the assumption that
      // make_stack_shape only happens within a call_packed.
      // We could relax this in the future if we want to
      // introduce root scope as a separate scope
      ICHECK_EQ(alloca_scope_.size(), scope_size)
          << "alloca_scope_ length is different before and after recursion";
      ICHECK_EQ(scope.run_sizes.shape_stack, -1)
          << "Expect no tvm_stack_make_shape outside of CallNodes";
      ICHECK_EQ(scope.run_sizes.array_stack, 0)
          << "Expect no tvm_stack_make_array outside of CallNodes";
    }

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
      if (call->op.same_as(builtin::nd_mem_alloc_with_scope())) {
        return StmtExprMutator::VisitStmt(MakeNdMemAllocWithScope(op, call));
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
    } else if (op->op.same_as(builtin::dma_copy())) {
      return MakeDMACopy(op);
    } else if (op->op.same_as(builtin::dma_wait())) {
      return MakeDMAWait(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr MakeDMACopy(const CallNode* op) {
    PrimExpr queue_id = op->args[0];
    PrimExpr dst = op->args[1];
    PrimExpr src = op->args[2];
    PrimExpr size = op->args[3];
    PrimExpr bypass_cache = op->args[4];

    std::string fdevapi_prefix =
        "device_api." + std::string(runtime::DeviceName(device_type_.as<IntImmNode>()->value));

    Call call_packed =
        Call(DataType::Int(32), builtin::tvm_call_packed(),
             {StringImm(fdevapi_prefix + ".dma_copy"), queue_id, dst, src, size, bypass_cache});
    return VisitExpr(call_packed);
  }

  PrimExpr MakeDMAWait(const CallNode* op) {
    PrimExpr queue_id = op->args[0];
    PrimExpr inflight = op->args[1];

    std::string fdevapi_prefix =
        "device_api." + std::string(runtime::DeviceName(device_type_.as<IntImmNode>()->value));

    Call call_packed = Call(DataType::Int(32), builtin::tvm_call_packed(),
                            {StringImm(fdevapi_prefix + ".dma_wait"), queue_id, inflight});
    return VisitExpr(call_packed);
  }

  // call shape
  PrimExpr MakeShape(const CallNode* op) {
    // if args.size() == 0, it represents a scalar shape ()
    ICHECK(!alloca_scope_.empty());
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();
    if (scope.run_sizes.shape_stack == -1) {
      scope.run_sizes.shape_stack = 0;
    }
    int64_t stack_begin = scope.run_sizes.shape_stack;
    scope.run_sizes.shape_stack += op->args.size();
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

    size_t idx = scope.run_sizes.array_stack;
    scope.run_sizes.array_stack += 1;
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
    PrimExpr elem_offset = op->args[5];
    PrimExpr byte_offset;
    if (!is_zero(elem_offset)) {
      byte_offset = elem_offset * make_const(elem_offset.dtype(), data_bytes);
    } else {
      byte_offset = elem_offset;
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

    int64_t restore_shape_stack = scope.run_sizes.shape_stack;
    size_t restore_array_stack = scope.run_sizes.array_stack;
    size_t arg_stack_begin = scope.run_sizes.arg_stack;

    size_t arg_count = op->args.size();

    // cpacked expects a resource_handle parameter
    if (!use_string_lookup) {
      arg_count--;
    }

    scope.run_sizes.arg_stack += arg_count;
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
    if (is_precheck_) {
      scope.UpdateMax();
    } else {
      scope.AssertMaxIsValid();
    }
    scope.run_sizes.shape_stack = restore_shape_stack;
    scope.run_sizes.array_stack = restore_array_stack;
    scope.run_sizes.arg_stack = arg_stack_begin;
    Array<PrimExpr> packed_args = {op->args[0], scope.stack_value, scope.stack_tcode->data,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1)};

    // cpacked call resource_handle
    if (!use_string_lookup) {
      PrimExpr last_arg = op->args[arg_count];
      const VarNode* var_node = last_arg.as<VarNode>();
      if (var_node != nullptr) {
        tir::Var resource_handle = GetRef<Var>(var_node);
        packed_args.push_back(StringImm(resource_handle->name_hint));
      } else {
        packed_args.push_back(last_arg);
      }
    }

    auto builtin_call = use_string_lookup ? builtin::tvm_call_packed_lowered()
                                          : builtin::tvm_call_cpacked_lowered();
    return Call(op->dtype, builtin_call, packed_args);
  }

  PrimExpr MakeCallTracePacked(const CallNode* op) {
    ICHECK(!alloca_scope_.empty());
    auto& scope = alloca_scope_.back();
    auto& prep_seq = prep_seq_stack_.back();

    int64_t restore_shape_stack = scope.run_sizes.shape_stack;
    size_t restore_array_stack = scope.run_sizes.array_stack;
    size_t arg_stack_begin = scope.run_sizes.arg_stack;
    scope.run_sizes.arg_stack += op->args.size();
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
    if (is_precheck_) {
      scope.UpdateMax();
    } else {
      scope.AssertMaxIsValid();
    }
    scope.run_sizes.shape_stack = restore_shape_stack;
    scope.run_sizes.array_stack = restore_array_stack;
    // Update the top of the stack, so we can use more than one
    // packed function's arguments with the one stack.
    scope.run_sizes.arg_stack = arg_stack_begin + args_size - 1;
    Array<PrimExpr> packed_args = {op->args[0], scope.stack_value, scope.stack_tcode->data,
                                   ConstInt32(arg_stack_begin),
                                   ConstInt32(arg_stack_begin + op->args.size() - 1),
                                   // Pass traced value.
                                   op->args[args_size - 1]};
    return Call(op->dtype, builtin::tvm_call_trace_packed_lowered(), packed_args);
  }

  Stmt MakeNdMemAllocWithScope(const LetStmtNode* let, const CallNode* call) {
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

    Array<PrimExpr> args = {
        StringImm(fdevapi_prefix + ".alloc_nd"),
        device_type_,
        device_id_,
        IntImm(DataType::Int(32), dtype.code()),
        IntImm(DataType::Int(32), dtype.bits()),
    };

    for (size_t i = 0; i < call->args.size(); ++i) {
      args.push_back(call->args[i]);
    }

    Call call_packed = Call(let->var.dtype(), builtin::tvm_call_packed(), args);
    Stmt alloca = LetStmt(let->var, call_packed, body);

    PrimExpr storage_scope = call->args[0];
    Call free_op = Call(DataType::Int(32), builtin::tvm_call_packed(),
                        {StringImm(fdevapi_prefix + ".free_nd"), device_type_, device_id_,
                         storage_scope, let->var});

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

  bool is_precheck_{false};

  // Record all stack frames.
  std::vector<AllocaScope> alloca_scope_;
};

namespace transform {

Pass LowerTVMBuiltin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BuiltinLower().Build(n->body);
    VLOG(2) << "LowerTVMBuiltin: " << f;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerTVMBuiltin", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerTVMBuiltin").set_body_typed(LowerTVMBuiltin);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
