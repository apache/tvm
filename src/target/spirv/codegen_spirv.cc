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
 * \file codegen_spirv.cc
 * \brief Generate SPIRV block
 */
#include <tvm/tir/expr.h>
#include <tvm/runtime/container.h>
#include <string>
#include "codegen_spirv.h"
#include "../../arith/compute_expr.h"

namespace tvm {
namespace codegen {

std::vector<uint32_t> CodeGenSPIRV::BuildFunction(const PrimFunc& f) {
  this->InitFuncState();
  CHECK(f->HasNonzeroAttr(tir::attr::kNoAlias))
      << "SPIRV only takes restricted memory model";
  std::vector<Var> pod_args;
  uint32_t num_buffer = 0;

  for (Var arg : f->params) {
    DataType t = arg.dtype();
    if (t.is_handle()) {
      if (auto* ptr = arg->type_annotation.as<PointerTypeNode>()) {
        auto* prim = ptr->element_type.as<PrimTypeNode>();
        CHECK(prim);
        DataType value_type = prim->dtype;
        spirv::Value arg_value = builder_->BufferArgument(
            builder_->GetSType(value_type), 0, num_buffer);
        storage_info_[arg.get()].UpdateContentType(value_type);
        var_map_[arg.get()] = arg_value;
      } else {
        LOG(FATAL) << "require all handles to be typed";
      }
      ++num_buffer;
    } else {
      pod_args.push_back(arg);
    }
  }
  spirv::Value func_ptr = builder_->NewFunction();
  builder_->StartFunction(func_ptr);

  // All the POD arguments are passed in through PushConstant
  if (pod_args.size() != 0) {
    std::vector<spirv::SType> value_types;
    for (size_t i = 0; i < pod_args.size(); ++i) {
      value_types.push_back(builder_->GetSType(pod_args[i].dtype()));
    }
    spirv::Value ptr = builder_->DeclarePushConstant(value_types);
    for (size_t i = 0; i < pod_args.size(); ++i) {
      spirv::Value value = builder_->GetPushConstant(
          ptr, value_types[i], static_cast<uint32_t>(i));
      var_map_[pod_args[i].get()] = value;
    }
  }
  this->VisitStmt(f->body);
  builder_->SetLocalSize(func_ptr, workgroup_size_);
  builder_->MakeInst(spv::OpReturn);
  builder_->MakeInst(spv::OpFunctionEnd);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol.defined())
      << "CodeGenSPIRV: Expect PrimFunc to have the global_symbol attribute";

  builder_->CommitKernelFunction(
    func_ptr, static_cast<std::string>(global_symbol.value()));

  return builder_->Finalize();
}

void CodeGenSPIRV::InitFuncState() {
  std::fill(workgroup_size_, workgroup_size_ + 3, 1);
  var_map_.clear();
  storage_info_.clear();
  analyzer_.reset(new arith::Analyzer());
  builder_.reset(new spirv::IRBuilder());
  builder_->InitHeader();
}

spirv::Value CodeGenSPIRV::GetThreadIndex(
    const IterVar& iv, const PrimExpr& extent) {
  runtime::ThreadScope ts = runtime::ThreadScope::make(iv->thread_tag);
  spirv::Value v;
  if (ts.rank == 1) {
    v = builder_->GetLocalID(ts.dim_index);
    auto* sizeptr = extent.as<tir::IntImmNode>();
    CHECK(sizeptr)
        << "SPIRV only allows constant thread group size " << " get " << extent;
    CHECK_LT(ts.dim_index, 3);
    workgroup_size_[ts.dim_index] = static_cast<uint32_t>(sizeptr->value);
  } else {
    v = builder_->GetWorkgroupID(ts.dim_index);
  }
  return builder_->Cast(builder_->GetSType(iv->var.dtype()), v);
}

spirv::Value CodeGenSPIRV::CreateStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  spirv::Value value;
  if (sync == "warp") {
    return value;
  } else if (sync == "shared") {
    auto type_int = builder_->GetSType(DataType::Int(32));
    builder_->MakeInst(
      spv::OpControlBarrier,
      builder_->IntImm(type_int, static_cast<int64_t>(spv::ScopeWorkgroup)),
      builder_->IntImm(type_int, static_cast<int64_t>(spv::ScopeWorkgroup)),
      builder_->IntImm(type_int, static_cast<int64_t>(
        spv::MemorySemanticsSequentiallyConsistentMask |
        spv::MemorySemanticsWorkgroupMemoryMask)));
  } else {
    LOG(FATAL) << "Do not support sync " << sync;
  }
  return value;
}

spirv::Value CodeGenSPIRV::VisitExpr_(const VarNode* op) {
  auto it = var_map_.find(op);
  CHECK(it != var_map_.end()) << "cannot find variable " << op->name_hint;
  return it->second;
}

spirv::Value CodeGenSPIRV::VisitExpr_(const IntImmNode* op) {
  return builder_->IntImm(builder_->GetSType(op->dtype), op->value);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const FloatImmNode* op) {
  return builder_->FloatImm(builder_->GetSType(op->dtype), op->value);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const StringImmNode* op) {
  LOG(FATAL) << "StringImm is not supported in Device code";
  return spirv::Value();
}

spirv::Value CodeGenSPIRV::VisitExpr_(const CastNode* op) {
  return builder_->Cast(builder_->GetSType(op->dtype), MakeValue(op->value));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const AddNode* op) {
  return builder_->Add(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const SubNode* op) {
  return builder_->Sub(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const MulNode* op) {
  return builder_->Mul(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const DivNode* op) {
  return builder_->Div(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const ModNode* op) {
  return builder_->Mod(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const MinNode* op) {
  spirv::Value a = MakeValue(op->a);
  spirv::Value b = MakeValue(op->b);
  return builder_->Select(builder_->LT(a, b), a, b);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const MaxNode* op) {
  spirv::Value a = MakeValue(op->a);
  spirv::Value b = MakeValue(op->b);
  return builder_->Select(builder_->GT(a, b), a, b);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const LTNode* op) {
  return builder_->LT(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const LENode* op) {
  return builder_->LE(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const GTNode* op) {
  return builder_->GT(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const GENode* op) {
  return builder_->GE(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const EQNode* op) {
  return builder_->EQ(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const NENode* op) {
  return builder_->NE(MakeValue(op->a), MakeValue(op->b));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const AndNode* op) {
  spirv::Value a = MakeValue(op->a);
  spirv::Value b = MakeValue(op->b);
  return builder_->MakeValue(spv::OpLogicalAnd, a.stype, a, b);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const OrNode* op) {
  spirv::Value a = MakeValue(op->a);
  spirv::Value b = MakeValue(op->b);
  return builder_->MakeValue(spv::OpLogicalOr, a.stype, a, b);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const NotNode* op) {
  spirv::Value a = MakeValue(op->a);
  return builder_->MakeValue(spv::OpLogicalNot, a.stype, a);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const SelectNode* op) {
  return builder_->Select(MakeValue(op->condition),
                          MakeValue(op->true_value),
                          MakeValue(op->false_value));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const LetNode* op) {
  CHECK(!var_map_.count(op->var.get()));
  var_map_[op->var.get()] = MakeValue(op->value);
  analyzer_->Bind(op->var, op->value);
  return MakeValue(op->body);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const CallNode* op) {
  if (op->is_intrinsic("spirv_glsl450")) {
    CHECK_GE(op->args.size(), 2U);
    uint32_t inst_id = static_cast<uint32_t>(
        op->args[0].as<IntImmNode>()->value);
    std::vector<spirv::Value> values;
    for (size_t i = 1; i < op->args.size(); ++i) {
      values.push_back(MakeValue(op->args[i]));
    }
    return builder_->CallGLSL450(
        builder_->GetSType(op->dtype), inst_id, values);
  } else if (op->is_intrinsic(CallNode::bitwise_and)) {
    CHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseAnd, a.stype, a, b);
  } else if (op->is_intrinsic(CallNode::bitwise_xor)) {
    CHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseXor, a.stype, a, b);
  } else if (op->is_intrinsic(CallNode::bitwise_or)) {
    CHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseOr, a.stype, a, b);
  } else if (op->is_intrinsic(CallNode::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    spirv::Value a = MakeValue(op->args[0]);
    return builder_->MakeValue(spv::OpNot, a.stype, a);
  } else if (op->is_intrinsic(CallNode::shift_left)) {
    CHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpShiftLeftLogical, a.stype, a, b);
  } else if (op->is_intrinsic(CallNode::shift_right)) {
    CHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    if (op->args[0].dtype().is_int()) {
      return builder_->MakeValue(spv::OpShiftRightArithmetic, a.stype, a, b);
    } else {
      return builder_->MakeValue(spv::OpShiftRightLogical, a.stype, a, b);
    }
  } else if (op->is_intrinsic(CallNode::reinterpret)) {
    return builder_->MakeValue(spv::OpBitcast, builder_->GetSType(op->dtype),
                               MakeValue(op->args[0]));
  } else if (op->is_intrinsic(intrinsic::tvm_large_uint_imm)) {
    CHECK_EQ(op->args.size(), 2U);
    uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
    uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
    uint64_t val = (high << 32U) | low;
    return builder_->UIntImm(builder_->GetSType(op->dtype), val);
  } else if (op->is_intrinsic(intrinsic::tvm_storage_sync)) {
    return this->CreateStorageSync(op);
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    CHECK_EQ(op->args.size(), 3U);
    spirv::Value cond = MakeValue(op->args[0]);
    spirv::Label then_label = builder_->NewLabel();
    spirv::Label else_label = builder_->NewLabel();
    spirv::Label merge_label = builder_->NewLabel();
    builder_->MakeInst(
        spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(
        spv::OpBranchConditional, cond, then_label, else_label);
    // then block, must get label after we see the value
    builder_->StartLabel(then_label);
    spirv::Value then_value = MakeValue(op->args[1]);
    spirv::Label then_value_label = builder_->CurrentLabel();
    builder_->MakeInst(spv::OpBranch, merge_label);
    // else block
    builder_->StartLabel(else_label);
    spirv::Value else_value = MakeValue(op->args[2]);
    spirv::Label else_value_label = builder_->CurrentLabel();
    builder_->MakeInst(spv::OpBranch, merge_label);
    // merge block
    builder_->StartLabel(merge_label);
    spirv::PhiValue phi = builder_->MakePhi(then_value.stype, 2);
    phi.SetIncoming(0, then_value, then_value_label);
    phi.SetIncoming(1, else_value, else_value_label);
    return phi;
  } else if (op->is_intrinsic("popcount")) {
    return builder_->MakeValue(
        spv::OpBitCount,
        builder_->GetSType(op->dtype),
        MakeValue(op->args[0]));
  } else {
    if (op->call_type == CallNode::Intrinsic ||
        op->call_type == CallNode::PureIntrinsic) {
      LOG(FATAL) << "Unresolved intrinsic " << op->name
                 << " with return type " << op->dtype;
    } else if (op->call_type == CallNode::Extern ||
               op->call_type == CallNode::PureExtern) {
      LOG(FATAL) << "Unresolved extern " << op->name
                 << " with return type " << op->dtype;
    } else {
      LOG(FATAL) << "Unresolved call type " << op->call_type;
    }
    return spirv::Value();
  }
}

spirv::Value CodeGenSPIRV::VisitExpr_(const RampNode* op) {
  std::vector<spirv::Value> values;
  spirv::Value base = MakeValue(op->base);
  for (int i = 0; i < op->lanes; ++i) {
    spirv::Value v = base;
    if (i != 0) {
      spirv::Value offset = MakeValue(
          make_const(op->stride.dtype(), i) * op->stride);
      v = builder_->Add(v, offset);
    }
    values.push_back(v);
  }
  return builder_->Concat(values);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const BroadcastNode* op) {
  std::vector<spirv::Value> values;
  spirv::Value v = MakeValue(op->value);
  for (int i = 0; i < op->lanes; i++) {
    values.push_back(v);
  }
  return builder_->Concat(values);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const LoadNode* op) {
  CHECK(is_one(op->predicate));
  auto it = storage_info_.find(op->buffer_var.get());
  CHECK(it != storage_info_.end());
  StorageInfo& info = it->second;
  if (!info.content_fixed) {
    info.UpdateContentType(op->dtype);
  }

  spirv::SType content_type = builder_->GetSType(info.content_type);
  spirv::Value buffer = MakeValue(op->buffer_var);
  spirv::SType ptr_type = builder_->GetPointerType(
      content_type, buffer.stype.storage_class);

  uint32_t mask = spv::MemoryAccessMaskNone;
  if (info.is_volatile) {
    mask |= spv::MemoryAccessVolatileMask;
  }
  if (op->dtype.lanes() == 1) {
    CHECK_EQ(info.content_type, op->dtype)
        << "Vulkan only allow one type access to the same buffer";
    spirv::Value index = MakeValue(op->index);
    spirv::Value ptr = builder_->StructArrayAccess(
        ptr_type, buffer, index);
    return builder_->MakeValue(spv::OpLoad, content_type, ptr, mask);
  } else {
    if (op->dtype.element_of() == info.content_type) {
      // because content type is element type, we can only do scalarize load.
      std::vector<spirv::Value> values;
      auto f = [&](int i, spirv::Value index) {
        spirv::Value ptr = builder_->StructArrayAccess(
            ptr_type, buffer, index);
        values.emplace_back(
            builder_->MakeValue(spv::OpLoad, content_type, ptr, mask));
      };
      this->Scalarize(op->index, f);
      return builder_->Concat(values);
    } else {
      if (const RampNode* ramp = op->index.as<RampNode>()) {
        if (is_one(ramp->stride)) {
          CHECK_EQ(ramp->lanes, op->dtype.lanes());
          arith::ModularSet me = analyzer_->modular_set(ramp->base);
          CHECK((me->coeff % ramp->lanes) == 0 &&
                (me->base % ramp->lanes)  == 0)
              << "Only aligned vector access is allowed in SPIRV";
          PrimExpr vec_index = analyzer_->Simplify(
              ramp->base / make_const(ramp->base.dtype(), ramp->lanes));
          spirv::Value ptr = builder_->StructArrayAccess(
              ptr_type, buffer, MakeValue(vec_index));
          return builder_->MakeValue(spv::OpLoad, content_type, ptr, mask);
        }
      }
    }
    LOG(FATAL) << "Only aligned continuous vector access is allowed in SPIRV";
  }
  LOG(FATAL) << "Only aligned continuous vector access is allowed in SPIRV";
  return spirv::Value();
}

void CodeGenSPIRV::Scalarize(const PrimExpr& e,
                             std::function<void(int i, spirv::Value v)> f) {
  if (const RampNode* ramp = e.as<RampNode>()) {
    for (int i = 0; i < ramp->dtype.lanes(); ++i) {
      PrimExpr offset = ramp->base + ramp->stride * i;
      f(i, MakeValue(offset));
    }
  } else {
    spirv::SType etype = builder_->GetSType(e.dtype().element_of());
    spirv::Value value = MakeValue(e);
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      f(i, builder_->MakeValue(
          spv::OpCompositeExtract, etype, value, i));
    }
  }
}

void CodeGenSPIRV::VisitStmt_(const StoreNode* op) {
  CHECK(is_one(op->predicate));
  auto it = storage_info_.find(op->buffer_var.get());
  CHECK(it != storage_info_.end());
  StorageInfo& info = it->second;

  if (!info.content_fixed) {
    info.UpdateContentType(op->value.dtype());
  }

  spirv::SType content_type = builder_->GetSType(info.content_type);
  spirv::Value buffer = MakeValue(op->buffer_var);
  spirv::Value value = MakeValue(op->value);
  spirv::SType ptr_type = builder_->GetPointerType(
      content_type, buffer.stype.storage_class);

  uint32_t mask = spv::MemoryAccessMaskNone;
  if (info.is_volatile) {
    mask |= spv::MemoryAccessVolatileMask;
  }

  if (op->value.dtype().lanes() == 1) {
    CHECK_EQ(info.content_type, op->value.dtype())
        << "Vulkan only allow one type access to the same buffer";
    spirv::Value index = MakeValue(op->index);
    spirv::Value ptr = builder_->StructArrayAccess(
        ptr_type, buffer, index);
    builder_->MakeInst(spv::OpStore, ptr, value, mask);
  } else {
    if (op->value.dtype().element_of() == info.content_type) {
      // because content type is element type, we can only do scalarize load.
      auto f = [&](int i, spirv::Value index) {
        spirv::Value elem = builder_->MakeValue(
            spv::OpCompositeExtract, content_type, value, i);
        spirv::Value ptr = builder_->StructArrayAccess(
            ptr_type, buffer, index);
        builder_->MakeInst(spv::OpStore, ptr, elem, mask);
      };
      this->Scalarize(op->index, f);
    } else {
      if (const RampNode* ramp = op->index.as<RampNode>()) {
        if (is_one(ramp->stride)) {
          CHECK_EQ(ramp->lanes, op->value.dtype().lanes());
          arith::ModularSet me = analyzer_->modular_set(ramp->base);
          CHECK((me->coeff % ramp->lanes) == 0 &&
                (me->base % ramp->lanes)  == 0)
              << "Only aligned vector access is allowed in SPIRV";
          PrimExpr vec_index = analyzer_->Simplify(
              ramp->base / make_const(ramp->base.dtype(), ramp->lanes));
          spirv::Value ptr = builder_->StructArrayAccess(
              ptr_type, buffer, MakeValue(vec_index));
          builder_->MakeInst(spv::OpStore, ptr, value, mask);
          return;
        }
      }
      LOG(FATAL) << "Only aligned continuous vector access is allowed in SPIRV";
    }
  }
}

void CodeGenSPIRV::VisitStmt_(const ForNode* op) {
  CHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::make_by_min_extent(op->min, op->extent));
  spirv::Value init_value = MakeValue(op->min);
  spirv::Value extent_value = MakeValue(op->extent);
  // Must get init label after making value(to make sure they are correct)
  spirv::Label init_label = builder_->CurrentLabel();
  spirv::Label head_label = builder_->NewLabel();
  spirv::Label body_label = builder_->NewLabel();
  spirv::Label continue_label = builder_->NewLabel();
  spirv::Label merge_label = builder_->NewLabel();
  builder_->MakeInst(spv::OpBranch, head_label);

  // Loop head
  builder_->StartLabel(head_label);
  spirv::PhiValue loop_var = builder_->MakePhi(init_value.stype, 2);
  loop_var.SetIncoming(0, init_value, init_label);
  spirv::Value loop_cond = builder_->LT(loop_var, extent_value);
  uint32_t control = (
      op->for_type == ForType::Unrolled ?
      spv::LoopControlUnrollMask : spv::LoopControlMaskNone);
  builder_->MakeInst(
      spv::OpLoopMerge, merge_label, continue_label, control);
  builder_->MakeInst(
      spv::OpBranchConditional, loop_cond, body_label, merge_label,
      weight_likely_branch_, 1);

  // loop body
  builder_->StartLabel(body_label);
  var_map_[op->loop_var.get()] = spirv::Value(loop_var);
  this->VisitStmt(op->body);
  builder_->MakeInst(spv::OpBranch, continue_label);

  // loop continue
  builder_->StartLabel(continue_label);
  spirv::Value one =
      op->loop_var.dtype().is_int() ?
      builder_->IntImm(loop_var.stype, 1) :
      builder_->UIntImm(loop_var.stype, 1);
  spirv::Value next_value = builder_->Add(loop_var, one);
  loop_var.SetIncoming(1, next_value, builder_->CurrentLabel());
  builder_->MakeInst(spv::OpBranch, head_label);
  // loop merge
  builder_->StartLabel(merge_label);
}

void CodeGenSPIRV::VisitStmt_(const IfThenElseNode* op) {
  spirv::Value cond = MakeValue(op->condition);
  spirv::Label then_label = builder_->NewLabel();
  spirv::Label merge_label = builder_->NewLabel();
  if (op->else_case.defined()) {
    spirv::Label else_label = builder_->NewLabel();
    builder_->MakeInst(
        spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(
        spv::OpBranchConditional, cond, then_label, else_label);
    // then block
    builder_->StartLabel(then_label);
    this->VisitStmt(op->then_case);
    builder_->MakeInst(spv::OpBranch, merge_label);
    // else block
    builder_->StartLabel(else_label);
    this->VisitStmt(op->else_case);
    builder_->MakeInst(spv::OpBranch, merge_label);
  } else {
    builder_->MakeInst(
        spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(
        spv::OpBranchConditional, cond, then_label, merge_label,
        weight_likely_branch_, 1);
    // then block
    builder_->StartLabel(then_label);
    this->VisitStmt(op->then_case);
    builder_->MakeInst(spv::OpBranch, merge_label);
  }
  // start merge label;
  builder_->StartLabel(merge_label);
}

void CodeGenSPIRV::VisitStmt_(const AllocateNode* op) {
  CHECK(!is_zero(op->condition));
  CHECK(!op->dtype.is_handle());
  int32_t constant_size = op->constant_allocation_size();
  CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation in GPU";
  spirv::Value buf;
  StorageInfo& info = storage_info_[op->buffer_var.get()];
  spirv::SType etype = builder_->GetSType(op->dtype);
  if (info.scope.rank == runtime::StorageRank::kLocal) {
    buf = builder_->Allocate(
        etype, static_cast<uint32_t>(constant_size),
        spv::StorageClassFunction);
  } else {
    // shared memory
    CHECK(info.scope.rank == runtime::StorageRank::kShared)
        << "Can only allocate shared or local memory inside kernel";
    // Shared memory
    buf = builder_->Allocate(
        etype, static_cast<uint32_t>(constant_size),
        spv::StorageClassWorkgroup);
  }
  CHECK(!info.content_fixed);
  info.UpdateContentType(op->dtype);
  CHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_map_.count(iv->var.get())) {
        var_map_[iv->var.get()] = GetThreadIndex(iv, op->value);
        analyzer_->Bind(iv->var, Range::make_by_min_extent(0, op->value));
      }
    }
  } else if (op->attr_key == tir::attr::storage_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    storage_info_[v].scope =
        runtime::StorageScope::make(op->value.as<StringImmNode>()->value);
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    storage_info_[v].is_volatile = true;
  }
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const AssertStmtNode* op) {
  With<arith::ConstraintContext> cctx(analyzer_.get(), op->condition);
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const LetStmtNode* op) {
  CHECK(!var_map_.count(op->var.get()));
  CHECK(!op->var.dtype().is_handle());
  var_map_[op->var.get()] = MakeValue(op->value);
  analyzer_->Bind(op->var, op->value);
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenSPIRV::VisitStmt_(const EvaluateNode* op) {
  MakeValue(op->value);
}

}  // namespace codegen
}  // namespace tvm
