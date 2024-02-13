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
#include "codegen_spirv.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>

#include "../../runtime/pack_args.h"
#include "../../runtime/vulkan/vulkan_common.h"
#include "../../tir/transforms/ir_utils.h"

namespace tvm {
namespace codegen {

CodeGenSPIRV::CodeGenSPIRV(Target target) : spirv_support_(target) {}

runtime::SPIRVShader CodeGenSPIRV::BuildFunction(const PrimFunc& f, const std::string& name) {
  this->InitFuncState();
  ICHECK(f->HasNonzeroAttr(tir::attr::kNoAlias)) << "SPIRV only takes restricted memory model";
  std::vector<Var> pod_args;
  uint32_t i_buffer = 0;

  // Currently, all storage and uniform buffer arguments are passed as
  // a single descriptor set at index 0.  If ever non-zero, must
  // ensure it is less than maxBoundDescriptorSets.
  const uint32_t descriptor_set = 0;

  for (Var arg : f->params) {
    DataType t = arg.dtype();
    if (t.is_handle()) {
      auto* ptr = arg->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr) << "All handles passed to the Vulkan codegen must have a type_annotation as a "
                     "PointerType, "
                  << "and must point to a PrimType";
      auto* prim = ptr->element_type.as<PrimTypeNode>();
      ICHECK(prim) << "All handles passed to the Vulkan codegen must have a type_annotation as a "
                      "PointerType, "
                   << "and must point to a PrimType";
      DataType value_storage_type = prim->dtype;
      if (value_storage_type == DataType::Bool()) {
        // We need a physically addressable buffer type to support boolean tensors.
        // The loaded byte is cast to bool inside the LoadNode visitor below.
        value_storage_type = boolean_storage_type_.with_lanes(value_storage_type.lanes());
      }
      spirv::Value arg_value = builder_->BufferArgument(builder_->GetSType(value_storage_type),
                                                        descriptor_set, i_buffer++);
      builder_->SetName(arg_value, arg->name_hint);
      storage_info_[arg.get()].SetContentType(value_storage_type, arg->name_hint);
      var_map_[arg.get()] = arg_value;
    } else {
      pod_args.push_back(arg);
    }
  }
  spirv::Value func_ptr = builder_->NewFunction();
  builder_->StartFunction(func_ptr);

  runtime::SPIRVShader shader;

  if (pod_args.size() != 0) {
    std::vector<spirv::SType> value_types;
    for (size_t i = 0; i < pod_args.size(); ++i) {
      value_types.push_back(builder_->GetSType(pod_args[i].dtype()));
    }
    if (pod_args.size() * sizeof(runtime::ArgUnion64) <= runtime::vulkan::kMaxPushConstantsBytes) {
      spirv::Value ptr = builder_->DeclarePushConstant(value_types);
      for (size_t i = 0; i < pod_args.size(); ++i) {
        spirv::Value value =
            builder_->GetPushConstant(ptr, value_types[i], static_cast<uint32_t>(i));
        var_map_[pod_args[i].get()] = value;
      }
    } else {
      shader.flag |= 1 << runtime::vulkan::ShaderMetaDataFlagMask::kUseUBO;
      // If we need to pass more arguments than push constants could handle, we use UBO.
      spirv::Value ptr = builder_->DeclareUniformBuffer(value_types, descriptor_set, i_buffer++);
      for (size_t i = 0; i < pod_args.size(); ++i) {
        spirv::Value value = builder_->GetUniform(ptr, value_types[i], static_cast<uint32_t>(i));
        var_map_[pod_args[i].get()] = value;
      }
    }
  }
  this->VisitStmt(f->body);
  builder_->SetLocalSize(func_ptr, workgroup_size_);
  builder_->MakeInst(spv::OpReturn);
  builder_->MakeInst(spv::OpFunctionEnd);

  builder_->CommitKernelFunction(func_ptr, name);

  ICHECK_LE(shared_memory_bytes_used_, spirv_support_.max_shared_memory_per_block)
      << "Vulkan shader " << name << " uses " << shared_memory_bytes_used_
      << " bytes of shared memory, "
      << "but target supports only " << spirv_support_.max_shared_memory_per_block << " bytes.  "
      << "If the device supports this allocation, "
      << "please add -max_shared_memory_per_block=NBYTES to the target, "
      << "or query all device parameters by adding -from_device=0.";

  shader.data = builder_->Finalize();
  return shader;
}

void CodeGenSPIRV::InitFuncState() {
  std::fill(workgroup_size_, workgroup_size_ + 3, 1);
  var_map_.clear();
  storage_info_.clear();
  analyzer_.reset(new arith::Analyzer());
  builder_.reset(new spirv::IRBuilder(spirv_support_));
  builder_->InitHeader();
  shared_memory_bytes_used_ = 0;
  fragment_info_.clear();
}

spirv::Value CodeGenSPIRV::GetThreadIndex(const IterVar& iv, const PrimExpr& extent) {
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  spirv::Value v;
  if (ts.rank == 1) {
    v = builder_->GetLocalID(ts.dim_index);
    auto* sizeptr = extent.as<tir::IntImmNode>();
    ICHECK(sizeptr) << "SPIRV only allows constant thread group size "
                    << " get " << extent;
    ICHECK_GE(ts.dim_index, 0) << "vthread should have been optimized out by here";
    ICHECK_LT(ts.dim_index, 3);
    workgroup_size_[ts.dim_index] = static_cast<uint32_t>(sizeptr->value);
  } else {
    v = builder_->GetWorkgroupID(ts.dim_index);
  }
  return builder_->Cast(builder_->GetSType(iv->var.dtype()), v);
}

spirv::Value CodeGenSPIRV::CreateStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  spirv::Value value;

  uint32_t vulkan_api_version = spirv_support_.vulkan_api_version;

  int64_t sync_scope;
  int64_t memory_semantics = spv::MemorySemanticsSequentiallyConsistentMask;
  if ((sync == "warp") && (vulkan_api_version >= VK_API_VERSION_1_1)) {
    // Synchronize control at the Subgroup level, but memory at the
    // Workgroup level.  This is because different invocations in a
    // subgroup may have each modified memory that exists at the
    // workgroup scope.  This should be changed if/when tir exposes
    // more information as to which memory access needs to be
    // synchronized.
    sync_scope = spv::ScopeSubgroup;
    memory_semantics |=
        spv::MemorySemanticsSubgroupMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask;

  } else if ((sync == "shared") || (sync == "warp")) {
    sync_scope = spv::ScopeWorkgroup;
    memory_semantics |= spv::MemorySemanticsWorkgroupMemoryMask;
  } else {
    LOG(FATAL) << "Do not support sync " << sync;
  }

  auto type_int = builder_->GetSType(DataType::Int(32));
  builder_->MakeInst(spv::OpControlBarrier, builder_->IntImm(type_int, sync_scope),
                     builder_->IntImm(type_int, sync_scope),
                     builder_->IntImm(type_int, memory_semantics));

  return value;
}

spirv::Value CodeGenSPIRV::VisitExpr_(const VarNode* op) {
  auto it = var_map_.find(op);
  ICHECK(it != var_map_.end()) << "cannot find variable " << op->name_hint;
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
  return builder_->Select(MakeValue(op->condition), MakeValue(op->true_value),
                          MakeValue(op->false_value));
}

spirv::Value CodeGenSPIRV::VisitExpr_(const LetNode* op) {
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  var_map_[op->var.get()] = MakeValue(op->value);
  analyzer_->Bind(op->var, op->value);
  return MakeValue(op->body);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_spirv_pure_glsl450())) {
    ICHECK_GE(op->args.size(), 2U);
    uint32_t inst_id = static_cast<uint32_t>(op->args[0].as<IntImmNode>()->value);
    std::vector<spirv::Value> values;
    for (size_t i = 1; i < op->args.size(); ++i) {
      values.push_back(MakeValue(op->args[i]));
    }
    return builder_->CallGLSL450(builder_->GetSType(op->dtype), inst_id, values);
  } else if (op->op.same_as(builtin::bitwise_and())) {
    ICHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseAnd, a.stype, a, b);
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    ICHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseXor, a.stype, a, b);
  } else if (op->op.same_as(builtin::bitwise_or())) {
    ICHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpBitwiseOr, a.stype, a, b);
  } else if (op->op.same_as(builtin::bitwise_not())) {
    ICHECK_EQ(op->args.size(), 1U);
    spirv::Value a = MakeValue(op->args[0]);
    return builder_->MakeValue(spv::OpNot, a.stype, a);
  } else if (op->op.same_as(builtin::shift_left())) {
    ICHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    return builder_->MakeValue(spv::OpShiftLeftLogical, a.stype, a, b);
  } else if (op->op.same_as(builtin::shift_right())) {
    ICHECK_EQ(op->args.size(), 2U);
    spirv::Value a = MakeValue(op->args[0]);
    spirv::Value b = MakeValue(op->args[1]);
    if (op->args[0].dtype().is_int()) {
      return builder_->MakeValue(spv::OpShiftRightArithmetic, a.stype, a, b);
    } else {
      return builder_->MakeValue(spv::OpShiftRightLogical, a.stype, a, b);
    }
  } else if (op->op.same_as(builtin::reinterpret())) {
    return builder_->MakeValue(spv::OpBitcast, builder_->GetSType(op->dtype),
                               MakeValue(op->args[0]));
  } else if (op->op.same_as(builtin::large_uint_imm())) {
    ICHECK_EQ(op->args.size(), 2U);
    uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
    uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
    uint64_t val = (high << 32U) | low;
    return builder_->UIntImm(builder_->GetSType(op->dtype), val);
  } else if (op->op.same_as(builtin::tvm_storage_sync())) {
    return this->CreateStorageSync(op);
  } else if (op->op.same_as(builtin::if_then_else())) {
    ICHECK_EQ(op->args.size(), 3U);
    spirv::Value cond = MakeValue(op->args[0]);
    spirv::Label then_label = builder_->NewLabel();
    spirv::Label else_label = builder_->NewLabel();
    spirv::Label merge_label = builder_->NewLabel();
    builder_->MakeInst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(spv::OpBranchConditional, cond, then_label, else_label);
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
  } else if (op->op.same_as(builtin::popcount())) {
    return builder_->MakeValue(spv::OpBitCount, builder_->GetSType(op->dtype),
                               MakeValue(op->args[0]));
  } else if (op->op.same_as(builtin::call_pure_extern())) {
    ICHECK_GE(op->args.size(), 1U);
    const std::string& func_name = op->args[0].as<StringImmNode>()->value;
    if (func_name == "__dp4a") {
      std::vector<spirv::Value> values;
      for (size_t i = 1; i < op->args.size(); ++i) {
        values.push_back(MakeValue(op->args[i]));
      }
      return builder_->CallKHRIntegerDotProduct(builder_->GetSType(op->dtype), values, op->dtype);
    } else {
      LOG(FATAL) << "SPIR-V shader cannot make extern calls.  Graph contains extern \""
                 << Downcast<StringImm>(op->args[0]) << "\"";
      return spirv::Value();
    }
  } else if (op->op.same_as(builtin::call_extern())) {
    ICHECK_GE(op->args.size(), 1U);
    LOG(FATAL) << "SPIR-V shader cannot make extern calls.  Graph contains extern \""
               << Downcast<StringImm>(op->args[0]) << "\"";
    return spirv::Value();
  } else if (op->op.same_as(builtin::tvm_fill_fragment())) {
    ICHECK_EQ(op->args.size(), 6U);
    const VarNode* buffer_node = op->args[0].as<VarNode>();
    ICHECK(buffer_node && fragment_info_.count(buffer_node));
    DataType ele_dtype = GetElementDataType(buffer_node);
    ICHECK(ele_dtype.is_float()) << "Only floating point fragment accumulator is supported";
    spirv::SType ele_stype = builder_->GetSType(ele_dtype);
    spirv::SType& fragment_type = fragment_info_[buffer_node].stype;
    double init = static_cast<uint64_t>(Downcast<FloatImm>(op->args[5])->value);
    PrimExpr prim_index = op->args[4];
    spirv::Value init_val = builder_->GetCompositeConst(ele_stype, fragment_type, init);
    spirv::SType ptr_type =
        builder_->GetPointerType(fragment_type, fragment_info_[buffer_node].sclass);
    spirv::Value index = MakeValue(prim_index);
    ICHECK(var_map_.count(buffer_node));
    spirv::Value ptr = builder_->StructArrayAccess(ptr_type, var_map_[buffer_node], index);
    builder_->MakeInst(spv::OpStore, ptr, init_val, spv::MemoryAccessMaskNone);
    return spirv::Value();

  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    ICHECK_EQ(op->args.size(), 8U);
    const VarNode* buffer_node = op->args[0].as<VarNode>();
    ICHECK(buffer_node && fragment_info_.count(buffer_node));
    spirv::SType& fragment_type = fragment_info_[buffer_node].stype;
    PrimExpr dst_index = op->args[4];
    PrimExpr src_ptr_expr = op->args[5];
    int stride = static_cast<int>(Downcast<IntImm>(op->args[6])->value);
    auto type_int = builder_->GetSType(DataType::Int(32));
    spirv::Value stride_val = builder_->IntImm(type_int, stride);
    std::string layout = (op->args[7].as<StringImmNode>())->value;
    spirv::SType dst_ptr_type =
        builder_->GetPointerType(fragment_type, fragment_info_[buffer_node].sclass);
    spirv::Value dst_ptr =
        builder_->StructArrayAccess(dst_ptr_type, var_map_[buffer_node], MakeValue(dst_index));
    spirv::Value src_ptr = VisitExpr(op->args[5]);
    spirv::SType type_bool = builder_->GetSType(DataType::UInt(1));
    spirv::Value t_val = builder_->UIntImm(type_bool, 1);
    spirv::Value f_val = builder_->UIntImm(type_bool, 0);
    spirv::Value loaded =
        builder_->MakeValue(spv::OpCooperativeMatrixLoadNV, fragment_type, src_ptr, stride_val,
                            (layout != "row_major") ? t_val : f_val);
    builder_->MakeInst(spv::OpStore, dst_ptr, loaded, spv::MemoryAccessMaskNone);
    return spirv::Value();
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    const VarNode* buffer_d = op->args[0].as<VarNode>();
    const VarNode* buffer_a = op->args[2].as<VarNode>();
    const VarNode* buffer_b = op->args[4].as<VarNode>();
    const VarNode* buffer_c = op->args[6].as<VarNode>();
    PrimExpr index_d = op->args[1];
    PrimExpr index_a = op->args[3];
    PrimExpr index_b = op->args[5];
    tvm::tir::ExprDeepEqual expr_equal;
    PrimExpr index_c = op->args[7];
    bool is_equal = ((buffer_d == buffer_c) && expr_equal(index_d, index_c));
    spirv::SType& fragment_type_d = fragment_info_[buffer_d].stype;
    spirv::SType& fragment_type_a = fragment_info_[buffer_a].stype;
    spirv::SType& fragment_type_b = fragment_info_[buffer_b].stype;
    spirv::SType& fragment_type_c = fragment_info_[buffer_c].stype;
    spv::StorageClass storage = fragment_info_[buffer_d].sclass;
    spirv::SType ptr_type_d = builder_->GetPointerType(fragment_type_d, storage);
    spirv::SType ptr_type_a = builder_->GetPointerType(fragment_type_a, storage);
    spirv::SType ptr_type_b = builder_->GetPointerType(fragment_type_b, storage);
    spirv::SType ptr_type_c = builder_->GetPointerType(fragment_type_c, storage);
    spirv::Value ptr_d =
        builder_->StructArrayAccess(ptr_type_d, var_map_[buffer_d], MakeValue(index_d));
    spirv::Value ptr_a =
        builder_->StructArrayAccess(ptr_type_a, var_map_[buffer_a], MakeValue(index_a));
    spirv::Value ptr_b =
        builder_->StructArrayAccess(ptr_type_b, var_map_[buffer_b], MakeValue(index_b));
    spirv::Value ptr_c =
        is_equal ? ptr_d
                 : builder_->StructArrayAccess(ptr_type_c, var_map_[buffer_c], MakeValue(index_c));
    uint32_t mask = spv::MemoryAccessMaskNone;
    spirv::Value loaded_a = builder_->MakeValue(spv::OpLoad, fragment_type_a, ptr_a, mask);
    spirv::Value loaded_b = builder_->MakeValue(spv::OpLoad, fragment_type_b, ptr_b, mask);
    spirv::Value loaded_c = builder_->MakeValue(spv::OpLoad, fragment_type_c, ptr_c, mask);
    spirv::Value result = builder_->MakeValue(spv::OpCooperativeMatrixMulAddNV, fragment_type_d,
                                              loaded_a, loaded_b, loaded_c);
    builder_->MakeInst(spv::OpStore, ptr_d, result, spv::MemoryAccessMaskNone);
    return spirv::Value();
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    ICHECK_EQ(op->args.size(), 8U);
    const VarNode* buffer_node = op->args[0].as<VarNode>();
    PrimExpr index = op->args[4];
    PrimExpr buffer_ptr = op->args[5];
    int stride = static_cast<int>(Downcast<IntImm>(op->args[6])->value);
    auto type_int = builder_->GetSType(DataType::Int(32));
    spirv::Value stride_val = builder_->IntImm(type_int, stride);
    std::string layout = (op->args[7].as<StringImmNode>())->value;
    spirv::Value dst_ptr = VisitExpr(op->args[5]);
    spirv::SType& fragment_type = fragment_info_[buffer_node].stype;
    spv::StorageClass storage = fragment_info_[buffer_node].sclass;
    spirv::SType ptr_type = builder_->GetPointerType(fragment_type, storage);
    spirv::Value ptr =
        builder_->StructArrayAccess(ptr_type, var_map_[buffer_node], MakeValue(index));
    uint32_t mask = spv::MemoryAccessMaskNone;
    spirv::Value loaded = builder_->MakeValue(spv::OpLoad, fragment_type, ptr, mask);
    spirv::SType type_bool = builder_->GetSType(DataType::UInt(1));
    spirv::Value t_val = builder_->UIntImm(type_bool, 1);
    spirv::Value f_val = builder_->UIntImm(type_bool, 0);
    builder_->MakeInst(spv::OpCooperativeMatrixStoreNV, dst_ptr, loaded, stride_val,
                       (layout != "row_major") ? t_val : f_val);
    return spirv::Value();
  } else if (op->op.same_as(builtin::address_of())) {
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    Var buffer_var = load->buffer->data;
    const VarNode* buffer_node = buffer_var.get();
    PrimExpr index = load->indices[0];
    DataType ele_dtype = GetElementDataType(buffer_node);
    spirv::SType ele_stype = builder_->GetSType(ele_dtype);
    spirv::Value buffer_val = MakeValue(buffer_var);
    spirv::SType ptr_type = builder_->GetPointerType(ele_stype, buffer_val.stype.storage_class);
    ICHECK(var_map_.count(buffer_node));
    return builder_->StructArrayAccess(ptr_type, var_map_[buffer_node], MakeValue(index));
  } else if (op->op.same_as(builtin::tvm_thread_invariant())) {
    return MakeValue(op->args[0]);
  } else {
    LOG(FATAL) << "Unresolved call  " << op->op;
  }
}

spirv::Value CodeGenSPIRV::VisitExpr_(const RampNode* op) {
  std::vector<spirv::Value> values;
  spirv::Value base = MakeValue(op->base);
  int lanes = op->dtype.lanes();
  for (int i = 0; i < lanes; ++i) {
    spirv::Value v = base;
    if (i != 0) {
      spirv::Value offset = MakeValue(make_const(op->stride.dtype(), i) * op->stride);
      v = builder_->Add(v, offset);
    }
    values.push_back(v);
  }
  return builder_->Concat(values);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const BroadcastNode* op) {
  std::vector<spirv::Value> values;
  spirv::Value v = MakeValue(op->value);
  int lanes = op->dtype.lanes();
  for (int i = 0; i < lanes; i++) {
    values.push_back(v);
  }
  return builder_->Concat(values);
}

spirv::Value CodeGenSPIRV::VisitExpr_(const BufferLoadNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "SPIR-V codegen expects flat memory buffers";
  Var buffer_var = op->buffer->data;
  PrimExpr prim_index = op->indices[0];

  DataType desired_read_type = op->dtype;
  if (desired_read_type == DataType::Bool()) {
    desired_read_type = boolean_storage_type_.with_lanes(desired_read_type.lanes());
  }

  auto it = storage_info_.find(buffer_var.get());
  ICHECK(it != storage_info_.end());
  StorageInfo& info = it->second;
  info.CheckContentType(desired_read_type, prim_index.dtype().lanes());

  spirv::SType content_type = builder_->GetSType(info.element_type);
  spirv::Value buffer = MakeValue(buffer_var);
  spirv::SType ptr_type = builder_->GetPointerType(content_type, buffer.stype.storage_class);

  uint32_t mask = spv::MemoryAccessMaskNone;
  if (info.is_volatile) {
    mask |= spv::MemoryAccessVolatileMask;
  }

  if (desired_read_type == info.element_type) {
    // Requested a single value from an array.  This may be a scalar load
    // or a vectorized load, based on the array element type.
    PrimExpr vec_index = analyzer_->Simplify(prim_index);
    spirv::Value index = MakeValue(vec_index);
    spirv::Value ptr = builder_->StructArrayAccess(ptr_type, buffer, index);
    spirv::Value loaded = builder_->MakeValue(spv::OpLoad, content_type, ptr, mask);
    // OpTypeBool have no physical address/storage.  Here, cast from
    // the storage type to an OpTypeBool.
    if (op->dtype == DataType::Bool()) {
      auto spirv_bool = builder_->GetSType(DataType::Bool());
      loaded = builder_->Cast(spirv_bool, loaded);
    }
    return loaded;

  } else if (desired_read_type.element_of() == info.element_type) {
    // Requested several elements returned as an array.  Read out each
    // element and concatenate into the result.
    std::vector<spirv::Value> values;
    auto f = [&](int i, spirv::Value index) {
      spirv::Value ptr = builder_->StructArrayAccess(ptr_type, buffer, index);
      values.emplace_back(builder_->MakeValue(spv::OpLoad, content_type, ptr, mask));
    };
    this->Scalarize(prim_index, f);
    return builder_->Concat(values);

  } else {
    LOG(FATAL) << "Cannot perform buffer access of buffer variable '" << buffer_var->name_hint
               << "' with element type " << info.element_type << " using index of type "
               << prim_index->dtype << " to produce output of type " << op->dtype;
    return spirv::Value();
  }
}

void CodeGenSPIRV::Scalarize(const PrimExpr& e, std::function<void(int i, spirv::Value v)> f) {
  if (const RampNode* ramp = e.as<RampNode>()) {
    for (int i = 0; i < ramp->dtype.lanes(); ++i) {
      PrimExpr offset = ramp->base + ramp->stride * i;
      f(i, MakeValue(offset));
    }
  } else {
    spirv::SType etype = builder_->GetSType(e.dtype().element_of());
    spirv::Value value = MakeValue(e);
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      f(i, builder_->MakeValue(spv::OpCompositeExtract, etype, value, i));
    }
  }
}

spirv::Value CodeGenSPIRV::VisitExpr_(const ShuffleNode* op) {
  ICHECK(op->vectors.size() == 1 && op->indices.size() == 1)
      << "SPIR-V codegen only supports shuffle "
      << "of one vector with one index";
  spirv::Value vector = MakeValue(op->vectors[0]);
  int index = Downcast<Integer>(op->indices[0])->value;
  spirv::SType etype = builder_->GetSType(op->dtype);
  spirv::Value element = builder_->MakeValue(spv::OpCompositeExtract, etype, vector, index);
  return element;
}

void CodeGenSPIRV::VisitStmt_(const BufferStoreNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "SPIR-V codegen expects flat memory buffers";
  Var buffer_var = op->buffer->data;
  PrimExpr prim_index = op->indices[0];

  auto it = storage_info_.find(buffer_var.get());
  ICHECK(it != storage_info_.end());
  StorageInfo& info = it->second;
  info.CheckContentType(op->value.dtype(), prim_index.dtype().lanes());

  spirv::SType content_type = builder_->GetSType(info.element_type);
  spirv::Value buffer = MakeValue(buffer_var);
  spirv::Value value = MakeValue(op->value);
  spirv::SType ptr_type = builder_->GetPointerType(content_type, buffer.stype.storage_class);

  uint32_t mask = spv::MemoryAccessMaskNone;
  if (info.is_volatile) {
    mask |= spv::MemoryAccessVolatileMask;
  }

  if (op->value.dtype() == info.element_type) {
    // Requested store of a single value.  This may be a scalar store
    // or a vectorized store, based on the array element type.
    ICHECK_EQ(info.element_type, op->value.dtype())
        << "Vulkan only allow one type access to the same buffer";
    spirv::Value index = MakeValue(prim_index);
    spirv::Value ptr = builder_->StructArrayAccess(ptr_type, buffer, index);
    builder_->MakeInst(spv::OpStore, ptr, value, mask);

  } else if (op->value.dtype().element_of() == info.element_type) {
    // Requested store of several arbitrarily located values.  Extract
    // each value from the composite, then assign to the buffer.
    auto f = [&](int i, spirv::Value index) {
      spirv::Value elem = builder_->MakeValue(spv::OpCompositeExtract, content_type, value, i);
      spirv::Value ptr = builder_->StructArrayAccess(ptr_type, buffer, index);
      builder_->MakeInst(spv::OpStore, ptr, elem, mask);
    };
    this->Scalarize(prim_index, f);

  } else {
    LOG(FATAL) << "Cannot store value of type " << op->value.dtype() << " into buffer variable '"
               << buffer_var->name_hint << "' with element type " << info.element_type
               << " using index of type " << prim_index->dtype;
  }
}

void CodeGenSPIRV::VisitStmt_(const ForNode* op) {
  ICHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  spirv::Value init_value = MakeValue(op->min);
  spirv::Value extent_value = MakeValue(op->extent);
  // Must get init label after making value(to make sure they are correct)
  spirv::Label init_label = builder_->CurrentLabel();
  spirv::Label head_label = builder_->NewLabel();
  builder_->SetName(head_label, "for_loop_head");
  spirv::Label body_label = builder_->NewLabel();
  builder_->SetName(body_label, "for_loop_body");
  spirv::Label continue_label = builder_->NewLabel();
  builder_->SetName(continue_label, "for_loop_continue");
  spirv::Label merge_label = builder_->NewLabel();
  builder_->SetName(merge_label, "for_loop_merge");
  builder_->MakeInst(spv::OpBranch, head_label);

  // Loop head
  builder_->StartLabel(head_label);
  spirv::PhiValue loop_var = builder_->MakePhi(init_value.stype, 2);
  loop_var.SetIncoming(0, init_value, init_label);
  spirv::Value loop_cond = builder_->LT(loop_var, extent_value);
  uint32_t control =
      (op->kind == ForKind::kUnrolled ? spv::LoopControlUnrollMask : spv::LoopControlMaskNone);
  builder_->MakeInst(spv::OpLoopMerge, merge_label, continue_label, control);
  builder_->MakeInst(spv::OpBranchConditional, loop_cond, body_label, merge_label,
                     weight_likely_branch_, 1);

  // loop body
  builder_->StartLabel(body_label);
  var_map_[op->loop_var.get()] = spirv::Value(loop_var);
  this->VisitStmt(op->body);
  builder_->MakeInst(spv::OpBranch, continue_label);

  // loop continue
  builder_->StartLabel(continue_label);
  spirv::Value one = op->loop_var.dtype().is_int() ? builder_->IntImm(loop_var.stype, 1)
                                                   : builder_->UIntImm(loop_var.stype, 1);
  spirv::Value next_value = builder_->Add(loop_var, one);
  loop_var.SetIncoming(1, next_value, builder_->CurrentLabel());
  builder_->MakeInst(spv::OpBranch, head_label);
  // loop merge
  builder_->StartLabel(merge_label);
}

void CodeGenSPIRV::VisitStmt_(const WhileNode* op) {
  spirv::Label head_label = builder_->NewLabel();
  spirv::Label condition_label = builder_->NewLabel();
  spirv::Label body_label = builder_->NewLabel();
  spirv::Label continue_label = builder_->NewLabel();
  spirv::Label merge_label = builder_->NewLabel();
  builder_->MakeInst(spv::OpBranch, head_label);

  // Loop head
  builder_->StartLabel(head_label);
  uint32_t control = spv::LoopControlMaskNone;
  builder_->MakeInst(spv::OpLoopMerge, merge_label, continue_label, control);
  builder_->MakeInst(spv::OpBranch, condition_label);

  // Loop condition evaluation.  The condition could contain if/else
  // blocks that introduce additional labels, so the condition cannot
  // be in the loop head's block.
  builder_->StartLabel(condition_label);
  spirv::Value loop_cond = MakeValue(op->condition);
  builder_->MakeInst(spv::OpBranchConditional, loop_cond, body_label, merge_label,
                     weight_likely_branch_, 1);

  // loop body
  builder_->StartLabel(body_label);
  this->VisitStmt(op->body);
  builder_->MakeInst(spv::OpBranch, continue_label);

  // loop continue
  builder_->StartLabel(continue_label);
  builder_->MakeInst(spv::OpBranch, head_label);

  // loop merge
  builder_->StartLabel(merge_label);
}

void CodeGenSPIRV::VisitStmt_(const IfThenElseNode* op) {
  spirv::Value cond = MakeValue(op->condition);
  spirv::Label then_label = builder_->NewLabel();
  spirv::Label merge_label = builder_->NewLabel();
  if (op->else_case) {
    spirv::Label else_label = builder_->NewLabel();
    builder_->MakeInst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(spv::OpBranchConditional, cond, then_label, else_label);
    // then block
    builder_->StartLabel(then_label);
    this->VisitStmt(op->then_case);
    builder_->MakeInst(spv::OpBranch, merge_label);
    // else block
    builder_->StartLabel(else_label);
    this->VisitStmt(op->else_case.value());
    builder_->MakeInst(spv::OpBranch, merge_label);
  } else {
    builder_->MakeInst(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
    builder_->MakeInst(spv::OpBranchConditional, cond, then_label, merge_label,
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
  ICHECK(!is_zero(op->condition));
  ICHECK(!op->dtype.is_handle());
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation in GPU";

  spirv::Value buf;
  const std::string scope = GetPtrStorageScope(op->buffer_var);
  auto storage_scope = runtime::StorageScope::Create(scope);
  spirv::SType etype = builder_->GetSType(op->dtype);
  runtime::StorageRank rank = storage_scope.rank;
  spv::StorageClass storage_class;
  const VarNode* var_node = (op->buffer_var).get();

  switch (rank) {
    case runtime::StorageRank::kWMMAMatrixA:
    case runtime::StorageRank::kWMMAMatrixB:
    case runtime::StorageRank::kWMMAAccumulator: {
      ICHECK(fragment_info_.count(var_node));
      fragment_info_[var_node].scope = scope;
      etype = GetFragmentSType(var_node, op->dtype);
      storage_class = spv::StorageClassFunction;
      fragment_info_[var_node].sclass = storage_class;
      ICHECK(fragment_info_.count(var_node));
      const std::string& scope = fragment_info_[var_node].scope;
      const std::string& shape_str = fragment_info_.at(var_node).shape;
      std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
      int64_t size = dim.first * dim.second;
      buf = builder_->Allocate(etype, static_cast<uint32_t>(constant_size) / size, storage_class);
    } break;
    case runtime::StorageRank::kLocal: {
      storage_class = spv::StorageClassFunction;
      buf = builder_->Allocate(etype, static_cast<uint32_t>(constant_size), storage_class);
    } break;
    case runtime::StorageRank::kShared: {
      storage_class = spv::StorageClassWorkgroup;
      // Shared memory
      // Aligned on 4-byte boundary
      int32_t aligned_constant_size = ((constant_size + 3) & ~0x3);
      buf = builder_->Allocate(etype, static_cast<uint32_t>(aligned_constant_size), storage_class);

      size_t num_bytes =
          op->dtype.bytes() * op->dtype.lanes() * static_cast<uint32_t>(aligned_constant_size);
      shared_memory_bytes_used_ += num_bytes;
    } break;
    default:
      LOG(FATAL) << "Can only allocate shared or local memory inside kernel";
  }

  builder_->SetName(buf, op->buffer_var->name_hint);

  StorageInfo& info = storage_info_[op->buffer_var.get()];
  ICHECK(!info.element_type_known);
  info.SetContentType(op->dtype, op->buffer_var->name_hint);

  ICHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const DeclBufferNode* op) { this->VisitStmt(op->body); }

void CodeGenSPIRV::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      // Will throw error if rebinding same local variable to a different extent.
      analyzer_->Bind(iv->var, Range::FromMinExtent(0, op->value));
      if (!var_map_.count(iv->var.get())) {
        var_map_[iv->var.get()] = GetThreadIndex(iv, op->value);
      }
    }
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    storage_info_[v].is_volatile = true;
  } else if (op->attr_key == tir::attr::buffer_bind_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
  } else if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_info_[buffer] = {shape_str->value};
  }
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const AssertStmtNode* op) {
  With<arith::ConstraintContext> cctx(analyzer_.get(), op->condition);
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const LetStmtNode* op) {
  ICHECK(!var_map_.count(op->var.get()));
  ICHECK(!op->var.dtype().is_handle());
  var_map_[op->var.get()] = MakeValue(op->value);
  analyzer_->Bind(op->var, op->value);
  this->VisitStmt(op->body);
}

void CodeGenSPIRV::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenSPIRV::VisitStmt_(const EvaluateNode* op) { MakeValue(op->value); }

spirv::SType CodeGenSPIRV::GetFragmentSType(const VarNode* buffer, const DataType& dtype) {
  ICHECK(fragment_info_.count(buffer));
  const std::string& scope = fragment_info_[buffer].scope;
  const std::string& shape_str = fragment_info_.at(buffer).shape;
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  int64_t size = dim.first * dim.second;
  spirv::SType stype = builder_->GetSType(dtype.with_lanes(size), dim.first, dim.second);
  fragment_info_[buffer].stype = stype;
  return stype;
}

DataType CodeGenSPIRV::GetElementDataType(const VarNode* buffer) {
  auto it = storage_info_.find(buffer);
  ICHECK(it != storage_info_.end());
  return it->second.element_type;
}

}  // namespace codegen
}  // namespace tvm
