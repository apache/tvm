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
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>

#include "ir_util.h"

namespace tvm {
namespace tir {

inline PrimExpr ConstInt32(size_t index) {
  CHECK_LE(index, std::numeric_limits<int>::max());
  return make_const(DataType::Int(32), static_cast<int>(index));
}

inline PrimExpr StackAlloca(std::string type, size_t num) {
  Array<PrimExpr> args = {StringImmNode::make(type), ConstInt32(num)};
  return CallNode::make(
      DataType::Handle(),
      intrinsic::tvm_stack_alloca,
      args, CallNode::Intrinsic);
}

// Calculate the statistics of packed function.
// These information are needed during codegen.
class BuiltinLower : public StmtExprMutator {
 public:
  Stmt Build(Stmt stmt) {
    stack_shape_ = Var("stack_shape", DataType::Handle());
    stack_array_ = Var("stack_array", DataType::Handle());
    stack_value_ = Var("stack_value", DataType::Handle());
    stack_tcode_ = Var("stack_tcode", DataType::Handle());
    stmt = this->VisitStmt(stmt);
    if (max_shape_stack_ != 0) {
      stmt = LetStmtNode::make(
          stack_shape_, StackAlloca("shape", max_shape_stack_), stmt);
    }
    if (max_array_stack_ != 0) {
      stmt = LetStmtNode::make(
          stack_array_, StackAlloca("array", max_array_stack_), stmt);
    }
    if (max_arg_stack_ != 0) {
      stmt = LetStmtNode::make(
          stack_value_, StackAlloca("arg_value", max_arg_stack_), stmt);
      stmt = LetStmtNode::make(
          stack_tcode_, StackAlloca("arg_tcode", max_arg_stack_), stmt);
    }
    return stmt;
  }

  Stmt VisitStmt(const Stmt& s) final {
    auto stmt = StmtExprMutator::VisitStmt(s);
    CHECK_EQ(run_shape_stack_, 0);
    CHECK_EQ(run_array_stack_, 0);

    if (prep_seq_.size() != 0) {
      Stmt ret = SeqStmt::Flatten(prep_seq_, stmt);
      prep_seq_.clear();
      return ret;
    } else {
      return stmt;
    }
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
    CHECK(device_type_.defined()) << "Unknown device type in current IR";
    CHECK(device_id_.defined()) << "Unknown device id in current IR";
    Stmt throw_last_error = EvaluateNode::make(
        CallNode::make(DataType::Int(32),
                       intrinsic::tvm_throw_last_error, {},
                       CallNode::Intrinsic));

    Stmt body = SeqStmt({
        IfThenElseNode::make(
            CallNode::make(DataType::Bool(1),
                           intrinsic::tvm_handle_is_null,
                           {op->buffer_var}, CallNode::PureIntrinsic),
            throw_last_error),
        op->body});

    Stmt alloca = LetStmtNode::make(
        op->buffer_var,
        CallNode::make(op->buffer_var.dtype(),
                       "TVMBackendAllocWorkspace",
                       {cast(DataType::Int(32), device_type_),
                        cast(DataType::Int(32), device_id_),
                        cast(DataType::UInt(64), total_bytes),
                        IntImm(DataType::Int(32), op->dtype.code()),
                        IntImm(DataType::Int(32), op->dtype.bits())},
                       CallNode::Extern),
        body);

    PrimExpr free_op = CallNode::make(DataType::Int(32),
                                  "TVMBackendFreeWorkspace",
                                  {cast(DataType::Int(32), device_type_),
                                   cast(DataType::Int(32), device_id_),
                                   op->buffer_var},
                                  CallNode::Extern);
    Stmt free_stmt = IfThenElseNode::make(
        free_op != make_zero(DataType::Int(32)), throw_last_error);
    body = SeqStmt({alloca, free_stmt});
    body = AttrStmtNode::make(
        op->buffer_var, attr::storage_alignment,
        make_const(DataType::Int(32), runtime::kTempAllocaAlignment),
        body);
    return body;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_id) {
      CHECK(!device_id_.defined());
      device_id_ = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::device_context_type) {
      CHECK(!device_type_.defined());
      device_type_ = op->value;
      return this->VisitStmt(op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_call_packed)) {
      return MakeCallPacked(op);
    } else if (op->is_intrinsic(intrinsic::tvm_call_trace_packed)) {
      return MakeCallTracePacked(op);
    } else if (op->is_intrinsic(intrinsic::tvm_stack_make_shape)) {
      return MakeShape(op);
    } else if (op->is_intrinsic(intrinsic::tvm_stack_make_array)) {
      return MakeArray(op);
    } else if (op->is_intrinsic(intrinsic::tvm_context_id)) {
      return make_zero(op->dtype);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  // call shape
  PrimExpr MakeShape(const CallNode* op) {
    size_t stack_begin = run_shape_stack_;
    run_shape_stack_ += op->args.size();
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 0; i < op->args.size(); ++i) {
      prep_seq_.emplace_back(
          StoreNode::make(stack_shape_, cast(DataType::Int(64), op->args[i]),
                      ConstInt32(stack_begin +i), const_true(1)));
    }
    return AddressOffset(stack_shape_, DataType::Int(64), stack_begin);
  }
  // make array
  PrimExpr MakeArray(const CallNode* op) {
    size_t idx = run_array_stack_;
    run_array_stack_ += 1;
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrData, op->args[0]));
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrShape, op->args[1]));
    PrimExpr strides = op->args[2];
    if (!strides.defined() || is_zero(strides)) {
      strides = make_zero(DataType::Handle());
    }
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrStrides, strides));
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrNDim, op->args[3]));
    DataType dtype = op->args[4].dtype();
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrTypeCode,
                     make_const(DataType::UInt(8), static_cast<int>(dtype.code()))));
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrTypeBits,
                     make_const(DataType::UInt(8), dtype.bits())));
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrTypeLanes,
                     make_const(DataType::UInt(16), dtype.lanes())));
    // set byte offset
    int data_bytes = GetVectorBytes(dtype);
    PrimExpr byte_offset = op->args[5];
    if (!is_zero(byte_offset)) {
      byte_offset = byte_offset * make_const(byte_offset.dtype(), data_bytes);
    }
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrByteOffset,
                     cast(DataType::UInt(64), byte_offset)));
    CHECK(device_type_.defined()) << "Unknown device type in current IR";
    CHECK(device_id_.defined()) << "Unknown device id in current IR";
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrDeviceId,
                     cast(DataType::Int(32), device_id_)));
    prep_seq_.emplace_back(
        TVMStructSet(stack_array_, idx, intrinsic::kArrDeviceType,
                     cast(DataType::Int(32), device_type_)));
    return TVMStructGet(DataType::Handle(), stack_array_, idx, intrinsic::kArrAddr);
  }
  // call packed.
  PrimExpr MakeCallPacked(const CallNode* op) {
    size_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = CastNode::make(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(
          stack_value_, static_cast<int>(arg_stack_begin + i - 1),
          intrinsic::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      if (api_type.is_handle() && arg.as<StringImmNode>()) {
        arg_tcode = kTVMStr;
      }
      if (IsArrayHandle(arg)) arg_tcode = kTVMDLTensorHandle;
      prep_seq_.emplace_back(
          StoreNode::make(stack_tcode_,
                      ConstInt32(arg_tcode),
                      stack_index, const_true(1)));
    }
    // UPDATE stack value
    max_arg_stack_ = std::max(run_arg_stack_, max_arg_stack_);
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    run_arg_stack_ = arg_stack_begin;
    Array<PrimExpr> packed_args = {
      op->args[0],
      stack_value_,
      stack_tcode_,
      ConstInt32(arg_stack_begin),
      ConstInt32(arg_stack_begin + op->args.size() - 1)
    };
    return CallNode::make(
        DataType::Int(32), intrinsic::tvm_call_packed_lowered,
        packed_args, CallNode::Intrinsic);
  }

  PrimExpr MakeCallTracePacked(const CallNode *op) {
    size_t restore_shape_stack = run_shape_stack_;
    size_t restore_array_stack = run_array_stack_;
    size_t arg_stack_begin = run_arg_stack_;
    run_arg_stack_ += op->args.size();
    size_t args_size = op->args.size();
    CHECK_GT(args_size, 0);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    for (size_t i = 1; i < op->args.size(); ++i) {
      PrimExpr stack_index = ConstInt32(arg_stack_begin + i - 1);
      PrimExpr arg = op->args[i];
      DataType t = arg.dtype();
      DataType api_type = APIType(t);
      if (t != api_type) {
        arg = CastNode::make(api_type, arg);
      }
      prep_seq_.emplace_back(TVMStructSet(
          stack_value_, static_cast<int>(arg_stack_begin + i - 1),
          intrinsic::kTVMValueContent, arg));
      int arg_tcode = api_type.code();
      CHECK(!IsArrayHandle(arg)) << "Trace does not support Buffers";
      prep_seq_.emplace_back(
          StoreNode::make(stack_tcode_,
                      ConstInt32(arg_tcode),
                      stack_index, const_true(1)));
    }
    // UPDATE stack value
    max_arg_stack_ = std::max(run_arg_stack_, max_arg_stack_);
    max_shape_stack_ = std::max(run_shape_stack_, max_shape_stack_);
    max_array_stack_ = std::max(run_array_stack_, max_array_stack_);
    run_shape_stack_ = restore_shape_stack;
    run_array_stack_ = restore_array_stack;
    // Update the top of the stack, so we can use more than one
    // packed function's arguments with the one stack.
    run_arg_stack_ = arg_stack_begin + args_size - 1;
    Array<PrimExpr> packed_args = {
      op->args[0],
      stack_value_,
      stack_tcode_,
      ConstInt32(arg_stack_begin),
      ConstInt32(arg_stack_begin + op->args.size() - 1),
      // Pass traced value.
      op->args[args_size - 1]
    };
    return CallNode::make(
        op->dtype, intrinsic::tvm_call_trace_packed_lowered,
        packed_args, CallNode::Intrinsic);
  }

 private:
  bool IsArrayHandle(const PrimExpr& arg) {
    // specially set array handle.
    if (const CallNode* buf = arg.as<CallNode>()) {
      if (buf->is_intrinsic(intrinsic::tvm_struct_get) &&
          buf->args[2].as<IntImmNode>()->value == intrinsic::kArrAddr) {
        return true;
      }
    }
    return false;
  }

  // The prepration sequence to be emitted.
  std::vector<Stmt> prep_seq_;
  PrimExpr device_type_;
  PrimExpr device_id_;
  // Var handle for each stack.
  Var stack_shape_;
  Var stack_array_;
  Var stack_tcode_;
  Var stack_value_;
  // The running statistics
  uint64_t run_shape_stack_{0};
  uint64_t run_array_stack_{0};
  uint64_t run_arg_stack_{0};
  // statistics of stacks
  uint64_t max_shape_stack_{0};
  uint64_t max_array_stack_{0};
  uint64_t max_arg_stack_{0};
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

TVM_REGISTER_GLOBAL("tir.transform.LowerTVMBuiltin")
.set_body_typed(LowerTVMBuiltin);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
