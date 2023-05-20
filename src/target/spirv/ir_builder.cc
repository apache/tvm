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
 * \file ir_builder.cc
 * \brief IRBuilder for SPIRV block
 */
#include "ir_builder.h"

#include <spirv.hpp>

namespace tvm {
namespace codegen {
namespace spirv {

// implementations

IRBuilder::IRBuilder(const SPIRVSupport& support) : spirv_support_(support) {}

void IRBuilder::InitHeader() {
  ICHECK_EQ(header_.size(), 0U);
  header_.push_back(spv::MagicNumber);

  // Target SPIR-V version 1.0.  Additional functionality will be
  // enabled through extensions.
  header_.push_back(0x10000);

  // generator: set to 0, unknown
  header_.push_back(0U);
  // Bound: set during Finalize
  header_.push_back(0U);
  // Schema: reserved
  header_.push_back(0U);

  // Declare CapabilityShader by default.  All other capabilities are
  // determined by the types declared.
  capabilities_used_.insert(spv::CapabilityShader);

#ifdef TVM_SPIRV_KHR_INTEGER_DOT_PRODUCT
  if (spirv_support_.supports_integer_dot_product) {
    capabilities_used_.insert(spv::CapabilityDotProductKHR);
    capabilities_used_.insert(spv::CapabilityDotProductInput4x8BitPackedKHR);
    extensions_used_.insert("SPV_KHR_integer_dot_product");
  }
#endif

  if (spirv_support_.supports_cooperative_matrix) {
    capabilities_used_.insert(spv::CapabilityCooperativeMatrixNV);
    extensions_used_.insert("SPV_NV_cooperative_matrix");
  }

  // memory model
  ib_.Begin(spv::OpMemoryModel)
      .AddSeq(spv::AddressingModelLogical, spv::MemoryModelGLSL450)
      .Commit(&entry_);
  this->InitPreDefs();
}

void IRBuilder::InitPreDefs() {
  ext_glsl450_ = ExtInstImport("GLSL.std.450");
  t_int32_ = DeclareType(DataType::Int(32));
  t_uint32_ = DeclareType(DataType::UInt(32));
  t_bool_ = DeclareType(DataType::UInt(1));
  t_fp32_ = DeclareType(DataType::Float(32));
  const_i32_zero_ = IntImm(t_int32_, 0);

  // declare void, and void functions
  t_void_.id = id_counter_++;
  ib_.Begin(spv::OpTypeVoid).Add(t_void_).Commit(&global_);
  t_void_func_.id = id_counter_++;
  ib_.Begin(spv::OpTypeFunction).AddSeq(t_void_func_, t_void_).Commit(&global_);
}

std::vector<uint32_t> IRBuilder::Finalize() {
  std::vector<uint32_t> data;
  // Index for upper bound of id numbers.
  const int kBoundLoc = 3;
  header_[kBoundLoc] = id_counter_;
  data.insert(data.end(), header_.begin(), header_.end());
  for (const auto& capability : capabilities_used_) {
    ib_.Begin(spv::OpCapability).Add(capability).Commit(&data);
  }
  for (const auto& ext_name : extensions_used_) {
    ib_.Begin(spv::OpExtension).Add(ext_name).Commit(&data);
  }
  data.insert(data.end(), extended_instruction_section_.begin(),
              extended_instruction_section_.end());
  data.insert(data.end(), entry_.begin(), entry_.end());
  data.insert(data.end(), exec_mode_.begin(), exec_mode_.end());
  data.insert(data.end(), debug_.begin(), debug_.end());
  data.insert(data.end(), decorate_.begin(), decorate_.end());
  data.insert(data.end(), global_.begin(), global_.end());
  data.insert(data.end(), func_header_.begin(), func_header_.end());
  data.insert(data.end(), function_scope_vars_.begin(), function_scope_vars_.end());
  data.insert(data.end(), function_.begin(), function_.end());
  return data;
}

SType IRBuilder::GetSType(const DataType& dtype, uint32_t row, uint32_t col) {
  if (dtype == DataType::Int(32)) {
    return t_int32_;
  } else if (dtype == DataType::UInt(1)) {
    return t_bool_;
  } else if (dtype == DataType::Float(32)) {
    return t_fp32_;
  } else if (dtype == DataType::UInt(32)) {
    return t_uint32_;
  }
  uint64_t type_key;
  type_key = static_cast<uint32_t>(dtype.code());
  type_key |= static_cast<uint32_t>(dtype.bits()) << 8U;
  if (row * col == 0) {
    ICHECK((row == 0) && (col == 0));
    type_key |= static_cast<uint32_t>(dtype.lanes()) << 16U;
  } else {
    type_key |= static_cast<uint64_t>(row) << 32U;
    type_key |= static_cast<uint64_t>(col) << 40U;
  }

  auto it = pod_type_tbl_.find(type_key);
  if (it != pod_type_tbl_.end()) {
    return it->second;
  }
  SType t = DeclareType(dtype, row, col);
  pod_type_tbl_[type_key] = t;
  return t;
}

SType IRBuilder::GetPointerType(const SType& value_type, spv::StorageClass storage_class) {
  ICHECK_NE(storage_class, spv::StorageClassMax);
  auto key = std::make_pair(value_type.id, storage_class);
  auto it = pointer_type_tbl_.find(key);
  if (it != pointer_type_tbl_.end()) {
    return it->second;
  }
  SType t;
  t.id = id_counter_++;
  t.type = DataType::Handle();
  t.element_type_id = value_type.id;
  t.storage_class = storage_class;
  ib_.Begin(spv::OpTypePointer).AddSeq(t, storage_class, value_type).Commit(&global_);
  pointer_type_tbl_[key] = t;
  return t;
}

SType IRBuilder::GetStructArrayType(const SType& value_type, uint32_t num_elems,
                                    bool interface_block) {
  auto key = std::make_tuple(value_type.id, num_elems, interface_block);
  auto it = struct_array_type_tbl_.find(key);
  if (it != struct_array_type_tbl_.end()) {
    return it->second;
  }

  SType arr_type;
  arr_type.id = id_counter_++;
  arr_type.type = DataType::Handle();
  arr_type.element_type_id = value_type.id;

  if (num_elems != 0) {
    Value length = UIntImm(GetSType(DataType::UInt(32)), num_elems);
    ib_.Begin(spv::OpTypeArray).AddSeq(arr_type, value_type, length).Commit(&global_);
  } else {
    ib_.Begin(spv::OpTypeRuntimeArray).AddSeq(arr_type, value_type).Commit(&global_);
  }
  int nbits = value_type.type.bits() * value_type.type.lanes();
  ICHECK_EQ(nbits % 8, 0);
  uint32_t nbytes = static_cast<uint32_t>(nbits) / 8;
  // decorate the array type.
  this->Decorate(spv::OpDecorate, arr_type, spv::DecorationArrayStride, nbytes);
  // declare struct of array
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.type = DataType::Handle();
  struct_type.element_type_id = value_type.id;
  ib_.Begin(spv::OpTypeStruct).AddSeq(struct_type, arr_type).Commit(&global_);
  // decorate the array type.
  ib_.Begin(spv::OpMemberDecorate)
      .AddSeq(struct_type, 0, spv::DecorationOffset, 0)
      .Commit(&decorate_);

  if (interface_block) {
    // Runtime array are always decorated as Block or BufferBlock
    // (shader storage buffer)
    if (spirv_support_.supports_storage_buffer_storage_class) {
      // If SPIRV 1.3+, or with extension
      // SPV_KHR_storage_buffer_storage_class, BufferBlock is
      // deprecated.
      extensions_used_.insert("SPV_KHR_storage_buffer_storage_class");
      this->Decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);
    } else {
      if (num_elems == 0) {
        this->Decorate(spv::OpDecorate, struct_type, spv::DecorationBufferBlock);
      }
    }
  }
  struct_array_type_tbl_[key] = struct_type;
  return struct_type;
}

Value IRBuilder::StructArrayAccess(const SType& res_type, Value buffer, Value index) {
  ICHECK(buffer.flag == kStructArrayPtr);
  return MakeValue(spv::OpInBoundsAccessChain, res_type, buffer, const_i32_zero_, index);
}

Value IRBuilder::IntImm(const SType& dtype, int64_t value) {
  return GetConst_(dtype, reinterpret_cast<uint64_t*>(&value));
}

Value IRBuilder::UIntImm(const SType& dtype, uint64_t value) { return GetConst_(dtype, &value); }

Value IRBuilder::FloatImm(const SType& dtype, double value) {
  if (dtype.type.bits() == 64) {
    return GetConst_(dtype, reinterpret_cast<uint64_t*>(&value));
  } else if (dtype.type.bits() == 32) {
    float fvalue = static_cast<float>(value);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(&fvalue);
    uint64_t data = ptr[0];
    return GetConst_(dtype, &data);
  } else {
    ICHECK_EQ(dtype.type.bits(), 16);
    float fvalue = static_cast<float>(value);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(&fvalue);
    uint64_t data = ptr[0];
    if (data == 0)
      return GetConst_(dtype, &data);
    else
      return Cast(dtype, FloatImm(GetSType(DataType::Float(32)), value));
  }
}

Value IRBuilder::BufferArgument(const SType& value_type, uint32_t descriptor_set,
                                uint32_t binding) {
  // If SPIRV 1.3+, or with extension SPV_KHR_storage_buffer_storage_class, BufferBlock is
  // deprecated.
  spv::StorageClass storage_class;
  if (spirv_support_.supports_storage_buffer_storage_class) {
    storage_class = spv::StorageClassStorageBuffer;
  } else {
    storage_class = spv::StorageClassUniform;
  }

  SType sarr_type = GetStructArrayType(value_type, 0, true);
  SType ptr_type = GetPointerType(sarr_type, storage_class);
  Value val = NewValue(ptr_type, kStructArrayPtr);

  ib_.Begin(spv::OpVariable).AddSeq(ptr_type, val, storage_class).Commit(&global_);

  this->DecorateBufferArgument(val, descriptor_set, binding);
  return val;
}

Value IRBuilder::DeclareStorageVariable(const std::vector<SType>& value_types,
                                        spv::StorageClass storage_class, ValueKind kind) {
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.type = DataType::Handle();
  ib_.Begin(spv::OpTypeStruct).Add(struct_type);
  for (const SType& vtype : value_types) {
    ib_.Add(vtype);
  }
  ib_.Commit(&global_);

  uint32_t offset = 0;
  for (uint32_t i = 0; i < value_types.size(); ++i) {
    ib_.Begin(spv::OpMemberDecorate)
        .AddSeq(struct_type, i, spv::DecorationOffset, offset)
        .Commit(&decorate_);
    DataType t = value_types[i].type;
    uint32_t nbits = t.bits() * t.lanes();
    ICHECK_EQ(nbits % 8, 0);
    uint32_t bytes = (nbits / 8);
    if (t.bits() == 32) {
      // In our Vulkan runtime, each scalar argument always occupies 64 bit.
      offset += bytes * 2;
    } else {
      ICHECK_EQ(t.bits(), 64);
      offset += bytes;
    }
  }
  this->Decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);

  SType ptr_type = GetPointerType(struct_type, storage_class);
  Value val = NewValue(ptr_type, kind);
  ib_.Begin(spv::OpVariable).AddSeq(ptr_type, val, storage_class).Commit(&global_);
  return val;
}

Value IRBuilder::DeclarePushConstant(const std::vector<SType>& value_types) {
  ICHECK_EQ(push_const_.id, 0);
  return DeclareStorageVariable(value_types, spv::StorageClassPushConstant, kPushConstantPtr);
}

Value IRBuilder::GetPushConstant(Value ptr_push_const, const SType& v_type, uint32_t index) {
  SType ptr_vtype = this->GetPointerType(v_type, spv::StorageClassPushConstant);
  Value ptr = this->MakeValue(spv::OpAccessChain, ptr_vtype, ptr_push_const,
                              IntImm(t_int32_, static_cast<int64_t>(index)));
  return this->MakeValue(spv::OpLoad, v_type, ptr);
}

Value IRBuilder::DeclareUniformBuffer(const std::vector<SType>& value_types,
                                      uint32_t descriptor_set, uint32_t binding) {
  Value val = DeclareStorageVariable(value_types, spv::StorageClassUniform, kUniformPtr);
  this->DecorateBufferArgument(val, descriptor_set, binding);
  return val;
}

void IRBuilder::DecorateBufferArgument(Value val, uint32_t descriptor_set, uint32_t binding) {
  this->Decorate(spv::OpDecorate, val, spv::DecorationDescriptorSet, descriptor_set);
  this->Decorate(spv::OpDecorate, val, spv::DecorationBinding, binding);
}

Value IRBuilder::GetUniform(Value ptr_push_const, const SType& v_type, uint32_t index) {
  SType ptr_vtype = this->GetPointerType(v_type, spv::StorageClassUniform);
  Value ptr = this->MakeValue(spv::OpAccessChain, ptr_vtype, ptr_push_const,
                              IntImm(t_int32_, static_cast<int64_t>(index)));
  return this->MakeValue(spv::OpLoad, v_type, ptr);
}

Value IRBuilder::NewFunction() { return NewValue(t_void_func_, kFunction); }

void IRBuilder::CommitKernelFunction(const Value& func, const std::string& name) {
  ICHECK_EQ(func.flag, kFunction);
  ib_.Begin(spv::OpEntryPoint).AddSeq(spv::ExecutionModelGLCompute, func, name);
  for (auto& it : built_in_tbl_) {
    ib_.Add(it.second);
  }
  ib_.Commit(&entry_);
}

void IRBuilder::StartFunction(const Value& func) {
  ICHECK_EQ(func.flag, kFunction);
  // add function declaration to the header.
  ib_.Begin(spv::OpFunction).AddSeq(t_void_, func, 0, t_void_func_).Commit(&func_header_);

  spirv::Label start_label = this->NewLabel();
  ib_.Begin(spv::OpLabel).AddSeq(start_label).Commit(&func_header_);
  curr_label_ = start_label;
}

void IRBuilder::SetLocalSize(const Value& func, uint32_t local_size[3]) {
  ICHECK_EQ(func.flag, kFunction);
  ib_.Begin(spv::OpExecutionMode)
      .AddSeq(func, spv::ExecutionModeLocalSize, local_size[0], local_size[1], local_size[2])
      .Commit(&exec_mode_);
}

Value IRBuilder::Allocate(const SType& value_type, uint32_t num_elems,
                          spv::StorageClass storage_class) {
  ICHECK_NE(num_elems, 0U);
  SType sarr_type = GetStructArrayType(value_type, num_elems, false);
  SType ptr_type = GetPointerType(sarr_type, storage_class);
  Value val = NewValue(ptr_type, kStructArrayPtr);
  if (storage_class == spv::StorageClassFunction) {
    ib_.Begin(spv::OpVariable).AddSeq(ptr_type, val, storage_class).Commit(&func_header_);
  } else {
    ib_.Begin(spv::OpVariable).AddSeq(ptr_type, val, storage_class).Commit(&global_);
  }
  return val;
}

Value IRBuilder::GetWorkgroupID(uint32_t dim_index) {
  std::string name = "blockIdx." + std::string(1, 'x' + dim_index);
  return GetBuiltInValue(spv::BuiltInWorkgroupId, dim_index, name);
}

Value IRBuilder::GetLocalID(uint32_t dim_index) {
  std::string name = "threadIdx." + std::string(1, 'x' + dim_index);
  return GetBuiltInValue(spv::BuiltInLocalInvocationId, dim_index, name);
}

Value IRBuilder::GetBuiltInValue(spv::BuiltIn built_in, uint32_t index, const std::string& name) {
  // Returned cached value if it exists
  {
    auto it = built_in_values_tbl_.find({built_in, index});
    if (it != built_in_values_tbl_.end()) {
      return it->second;
    }
  }

  DataType data_type;
  DataType global_arr_type;
  switch (built_in) {
    case spv::BuiltInLocalInvocationId:
    case spv::BuiltInWorkgroupId:
      data_type = DataType::Int(32);
      global_arr_type = data_type.with_lanes(3);
      break;

    default:
      LOG(FATAL) << "No data type defined for SPIR-V Built-In " << built_in;
  }

  // Look up the decorated array value at global scope.  If it doesn't
  // exist already, declare it.
  Value global_array;
  {
    auto it = built_in_tbl_.find(built_in);
    if (it != built_in_tbl_.end()) {
      global_array = it->second;
    } else {
      SType ptr_arr_type = this->GetPointerType(GetSType(global_arr_type), spv::StorageClassInput);
      global_array = NewValue(ptr_arr_type, kVectorPtr);

      ib_.Begin(spv::OpVariable)
          .AddSeq(ptr_arr_type, global_array, spv::StorageClassInput)
          .Commit(&global_);
      this->Decorate(spv::OpDecorate, global_array, spv::DecorationBuiltIn, built_in);

      switch (built_in) {
        case spv::BuiltInLocalInvocationId:
          SetName(global_array, "BuiltInLocalInvocationId");
          break;
        case spv::BuiltInWorkgroupId:
          SetName(global_array, "BuiltInWorkgroupId");
          break;

        default:
          break;
      }

      built_in_tbl_[built_in] = global_array;
    }
  }

  // Declare the dereferenced value
  SType data_stype = GetSType(data_type);
  SType ptr_type = this->GetPointerType(data_stype, spv::StorageClassInput);
  Value global_const_index = UIntImm(t_int32_, static_cast<int64_t>(index));

  Value ptr = NewValue(ptr_type, kNormal);
  ib_.Begin(spv::OpAccessChain)
      .AddSeq(ptr_type, ptr, global_array, global_const_index)
      .Commit(&function_scope_vars_);

  Value output = NewValue(data_stype, kNormal);
  ib_.Begin(spv::OpLoad).AddSeq(data_stype, output, ptr).Commit(&function_scope_vars_);
  if (name.size()) {
    SetName(output, name);
  }

  // Store to cache and return
  built_in_values_tbl_[{built_in, index}] = output;
  return output;
}

Value IRBuilder::GetConst_(const SType& dtype, const uint64_t* pvalue) {
  auto key = std::make_pair(dtype.id, pvalue[0]);
  auto it = const_tbl_.find(key);
  if (it != const_tbl_.end()) {
    return it->second;
  }
  ICHECK_LE(dtype.type.bits(), 64);
  Value ret = NewValue(dtype, kConstant);
  if (dtype.type == DataType::UInt(1)) {
    // bool types.
    if (*pvalue) {
      ib_.Begin(spv::OpConstantTrue).AddSeq(dtype, ret);
    } else {
      ib_.Begin(spv::OpConstantFalse).AddSeq(dtype, ret);
    }
  } else {
    // Integral/floating-point types.
    ib_.Begin(spv::OpConstant).AddSeq(dtype, ret);
    uint64_t mask = 0xFFFFFFFFUL;
    ib_.Add(static_cast<uint32_t>(pvalue[0] & mask));
    if (dtype.type.bits() > 32) {
      if (dtype.type.is_int()) {
        int64_t sign_mask = 0xFFFFFFFFL;
        const int64_t* sign_ptr = reinterpret_cast<const int64_t*>(pvalue);
        ib_.Add(static_cast<uint32_t>((sign_ptr[0] >> 32L) & sign_mask));
      } else {
        ib_.Add(static_cast<uint32_t>((pvalue[0] >> 32UL) & mask));
      }
    }
  }
  ib_.Commit(&global_);
  const_tbl_[key] = ret;
  return ret;
}

SType IRBuilder::DeclareType(const DataType& dtype, uint32_t row, uint32_t col) {
  AddCapabilityFor(dtype);

  if (dtype.lanes() == 1) {
    SType t;
    t.id = id_counter_++;
    t.type = dtype;
    if (dtype.bits() == 1) {
      ICHECK(dtype.is_uint());
      ib_.Begin(spv::OpTypeBool).Add(t).Commit(&global_);
    } else if (dtype.is_int()) {
      ib_.Begin(spv::OpTypeInt).AddSeq(t, dtype.bits(), 1).Commit(&global_);
    } else if (dtype.is_uint()) {
      ib_.Begin(spv::OpTypeInt).AddSeq(t, dtype.bits(), 0).Commit(&global_);
    } else if (dtype.is_float()) {
      ib_.Begin(spv::OpTypeFloat).AddSeq(t, dtype.bits()).Commit(&global_);
    } else {
      LOG(FATAL) << "declare type do not support handle";
    }
    return t;
  } else {
    SType t;
    t.id = id_counter_++;
    t.type = dtype;
    SType base_type = GetSType(dtype.element_of());

    if (row * col == 0) {
      ICHECK((row == 0) && (col == 0));
      ib_.Begin(spv::OpTypeVector).AddSeq(t, base_type, dtype.lanes()).Commit(&global_);
    } else {
      Value v_row = GetSpecConst(GetSType(DataType::UInt(32)), row);
      Value v_col = GetSpecConst(GetSType(DataType::UInt(32)), col);
      Value scope = UIntImm(GetSType(DataType::UInt(32)), spv::ScopeSubgroup);
      ib_.Begin(spv::OpTypeCooperativeMatrixNV)
          .AddSeq(t, base_type, scope, v_row, v_col)
          .Commit(&global_);
    }
    return t;
  }
}

void IRBuilder::AddCapabilityFor(const DataType& dtype) {
  // Declare appropriate capabilities for int/float types
  if (dtype.is_int() || dtype.is_uint()) {
    if (dtype.bits() == 8) {
      ICHECK(spirv_support_.supports_int8)
          << "Vulkan target does not support Int8 capability.  "
          << "If your device supports 8-bit int operations, "
          << "please either add -supports_int8=1 to the target, "
          << "or query all device parameters by adding -from_device=0.";
      capabilities_used_.insert(spv::CapabilityInt8);
    } else if (dtype.bits() == 16) {
      ICHECK(spirv_support_.supports_int16)
          << "Vulkan target does not support Int16 capability.  "
          << "If your device supports 16-bit int operations, "
          << "please either add -supports_int16=1 to the target, "
          << "or query all device parameters by adding -from_device=0.";
      capabilities_used_.insert(spv::CapabilityInt16);
    } else if (dtype.bits() == 64) {
      ICHECK(spirv_support_.supports_int64)
          << "Vulkan target does not support Int64 capability.  "
          << "If your device supports 64-bit int operations, "
          << "please either add -supports_int64=1 to the target, "
          << "or query all device parameters by adding -from_device=0.";
      capabilities_used_.insert(spv::CapabilityInt64);
    }

  } else if (dtype.is_float()) {
    if (dtype.bits() == 16) {
      ICHECK(spirv_support_.supports_float16)
          << "Vulkan target does not support Float16 capability.  "
          << "If your device supports 16-bit float operations, "
          << "please either add -supports_float16=1 to the target, "
          << "or query all device parameters by adding -from_device=0.";
      capabilities_used_.insert(spv::CapabilityFloat16);
    } else if (dtype.bits() == 64) {
      ICHECK(spirv_support_.supports_float64)
          << "Vulkan target does not support Float64 capability.  "
          << "If your device supports 64-bit float operations, "
          << "please either add -supports_float64=1 to the target, "
          << "or query all device parameters by adding -from_device=0.";
      capabilities_used_.insert(spv::CapabilityFloat64);
    }
  }

  // Declare ability to read type to/from storage buffers.  Doing so
  // here is a little bit overzealous, should be relaxed in the
  // future.  Requiring StorageBuffer8BitAccess in order to declare an
  // Int8 prevents use of an 8-bit loop iterator on a device that
  // supports Int8 but doesn't support 8-bit buffer access.
  if (dtype.bits() == 8) {
    ICHECK(spirv_support_.supports_storage_buffer_8bit_access)
        << "Vulkan target does not support StorageBuffer8BitAccess.  "
        << "If your device supports 8-bit buffer access, "
        << "please either add -supports_8bit_buffer=1 to the target, "
        << "or query all device parameters by adding -from_device=0.";
    capabilities_used_.insert(spv::CapabilityStorageBuffer8BitAccess);
    extensions_used_.insert("SPV_KHR_8bit_storage");

    ICHECK(spirv_support_.supports_storage_buffer_storage_class)
        << "Illegal Vulkan target description.  "
        << "Vulkan spec requires extension VK_KHR_storage_buffer_storage_class "
        << "if VK_KHR_8bit_storage is supported.  "
        << "Please either add -supports_storage_buffer_storage_class=1 to the target, "
        << "or query all device parameters by adding -from_device=0.";
  } else if (dtype.bits() == 16) {
    ICHECK(spirv_support_.supports_storage_buffer_16bit_access)
        << "Vulkan target does not support StorageBuffer16BitAccess.  "
        << "If your device supports 16-bit buffer access, "
        << "please either add -supports_16bit_buffer=1 to the target, "
        << "or query all device parameters by adding -from_device=0.";

    extensions_used_.insert("SPV_KHR_16bit_storage");
    if (spirv_support_.supports_storage_buffer_storage_class) {
      capabilities_used_.insert(spv::CapabilityStorageBuffer16BitAccess);
    } else {
      capabilities_used_.insert(spv::CapabilityStorageUniformBufferBlock16);
    }
  }
}

PhiValue IRBuilder::MakePhi(const SType& out_type, uint32_t num_incoming) {
  Value val = NewValue(out_type, kNormal);
  ib_.Begin(spv::OpPhi).AddSeq(out_type, val);
  for (uint32_t i = 0; i < 2 * num_incoming; ++i) {
    ib_.Add(0);
  }
  PhiValue phi;
  phi.id = val.id;
  phi.stype = out_type;
  phi.flag = kNormal;
  phi.instr = ib_.Commit(&function_);
  ICHECK_EQ(phi.instr.WordCount(), 2 * num_incoming + 3);
  return phi;
}

Value IRBuilder::CallGLSL450(const SType& ret_type, uint32_t inst_id,
                             const std::vector<Value>& args) {
  Value val = NewValue(ret_type, kNormal);
  ib_.Begin(spv::OpExtInst).AddSeq(ret_type, val, ext_glsl450_, inst_id);
  for (const Value& v : args) {
    ib_.Add(v);
  }
  ib_.Commit(&function_);
  return val;
}

Value IRBuilder::CallKHRIntegerDotProduct(const SType& ret_type, const std::vector<Value>& args,
                                          const DataType& dtype) {
  if (args.size() != 3) {
    LOG(FATAL) << "Unresolved arguments in SPIRV_KHR_integer_dot_product";
  }
  Value val = NewValue(ret_type, kNormal);
#ifdef TVM_SPIRV_KHR_INTEGER_DOT_PRODUCT
  ICHECK(spirv_support_.supports_integer_dot_product)
      << "Vulkan target does not support integer dot product capability.  "
      << "If your device supports integer dot product operations, "
      << "please either add -mattr=+dotprod to the target, "
      << "or query all device parameters by adding -from_device=0.";
  if (dtype.is_int()) {
    ib_.Begin(spv::OpSDotAccSatKHR).AddSeq(ret_type, val);
  } else if (dtype.is_uint()) {
    ib_.Begin(spv::OpUDotAccSatKHR).AddSeq(ret_type, val);
  } else {
    LOG(FATAL) << "Unsupported type";
  }
#else
  LOG(FATAL) << "Please turn on USE_SPIRV_KHR_INTEGER_DOT_PRODUCT in config.cmake";
#endif

  for (const Value& v : args) {
    ib_.Add(v);
  }
  ib_.Commit(&function_);
  return val;
}

Value IRBuilder::Concat(const std::vector<Value>& vec) {
  bool is_const = vec[0].flag == kConstant;
  DataType etype = vec[0].stype.type;
  int lanes = etype.lanes();
  for (size_t i = 1; i < vec.size(); ++i) {
    ICHECK_EQ(etype, vec[i].stype.type.element_of())
        << "Cannot concat vector of different element type";
    lanes += vec[i].stype.type.lanes();
    is_const = is_const && (vec[i].flag == kConstant);
  }
  Value ret = NewValue(GetSType(etype.with_lanes(lanes)), kNormal);
  if (is_const && vec.size() == static_cast<size_t>(lanes)) {
    ib_.Begin(spv::OpConstantComposite);
    ib_.AddSeq(ret.stype, ret);
    for (const Value& v : vec) {
      ib_.Add(v);
    }
    ib_.Commit(&global_);
  } else {
    ib_.Begin(spv::OpCompositeConstruct);
    ib_.AddSeq(ret.stype, ret);
    for (const Value& v : vec) {
      ib_.Add(v);
    }
    ib_.Commit(&function_);
  }
  return ret;
}

Value IRBuilder::Cast(const SType& dst_type, spirv::Value value) {
  ICHECK_NE(value.stype.id, 0U);
  if (value.stype.id == dst_type.id) return value;
  const tvm::DataType& from = value.stype.type;
  const tvm::DataType& to = dst_type.type;
  ICHECK_EQ(from.lanes(), to.lanes());
  if (from == DataType::Bool()) {
    if (to.is_int()) {
      return Select(value, IntImm(dst_type, 1), IntImm(dst_type, 0));
    } else if (to.is_uint()) {
      return Select(value, UIntImm(dst_type, 1), UIntImm(dst_type, 0));
    } else if (to.is_float()) {
      return MakeValue(spv::OpConvertUToF, dst_type,
                       Select(value, UIntImm(t_uint32_, 1), UIntImm(t_uint32_, 0)));
    } else {
      LOG(FATAL) << "cannot cast from " << from << " to " << to;
      return Value();
    }
  } else if (to == DataType::Bool()) {
    if (from.is_int()) {
      return NE(value, IntImm(value.stype, 0));
    } else if (to.is_uint()) {
      return NE(value, UIntImm(value.stype, 0));
    } else {
      LOG(FATAL) << "cannot cast from " << from << " to " << to;
      return Value();
    }
  } else if (from.is_int() && to.is_int()) {
    return MakeValue(spv::OpSConvert, dst_type, value);
  } else if (from.is_uint() && to.is_uint()) {
    return MakeValue(spv::OpUConvert, dst_type, value);
  } else if (from.is_uint() && to.is_int()) {
    if (from.bits() != to.bits()) {
      value = MakeValue(spv::OpUConvert, GetSType(from.with_bits(to.bits())), value);
    }
    return MakeValue(spv::OpBitcast, dst_type, value);
  } else if (from.is_int() && to.is_uint()) {
    if (from.bits() != to.bits()) {
      value = MakeValue(spv::OpSConvert, GetSType(from.with_bits(to.bits())), value);
    }
    return MakeValue(spv::OpBitcast, dst_type, value);
  } else if (from.is_float() && to.is_int()) {
    return MakeValue(spv::OpConvertFToS, dst_type, value);
  } else if (from.is_float() && to.is_uint()) {
    return MakeValue(spv::OpConvertFToU, dst_type, value);
  } else if (from.is_int() && to.is_float()) {
    return MakeValue(spv::OpConvertSToF, dst_type, value);
  } else if (from.is_uint() && to.is_float()) {
    return MakeValue(spv::OpConvertUToF, dst_type, value);
  } else if (from.is_float() && to.is_float()) {
    return MakeValue(spv::OpFConvert, dst_type, value);
  } else {
    LOG(FATAL) << "do not support type cast from " << from << " to " << to;
    return Value();
  }
}

Value IRBuilder::GetCompositeConst(const SType& ele_stype, const SType& composite_stype,
                                   const double dval) {
  auto key = std::make_pair(composite_stype.id, dval);
  auto it = composite_const_tbl_.find(key);
  if (it != composite_const_tbl_.end()) {
    return it->second;
  }
  spirv::Value const_val = FloatImm(ele_stype, dval);
  Value new_val = NewValue(composite_stype, kNormal);
  ib_.Begin(spv::OpConstantComposite).AddSeq(composite_stype, new_val, const_val);
  ib_.Commit(&global_);
  composite_const_tbl_[key] = new_val;
  return new_val;
}

Value IRBuilder::GetSpecConst(const SType& dtype, uint64_t value) {
  ICHECK_LE(dtype.type.bits(), 32);
  Value ret = NewValue(dtype, kSpecConst);
  ib_.Begin(spv::OpSpecConstant).AddSeq(dtype, ret);
  ib_.Add(static_cast<uint32_t>(value));
  ib_.Commit(&global_);
  return ret;
}

#define DEFINE_BUILDER_BINARY_USIGN_OP(_OpName, _Op)       \
  Value IRBuilder::_OpName(Value a, Value b) {             \
    ICHECK_EQ(a.stype.id, b.stype.id);                     \
    if (a.stype.type.is_int() || a.stype.type.is_uint()) { \
      return MakeValue(spv::OpI##_Op, a.stype, a, b);      \
    } else {                                               \
      ICHECK(a.stype.type.is_float());                     \
      return MakeValue(spv::OpF##_Op, a.stype, a, b);      \
    }                                                      \
  }

#define DEFINE_BUILDER_BINARY_SIGN_OP(_OpName, _Op)   \
  Value IRBuilder::_OpName(Value a, Value b) {        \
    ICHECK_EQ(a.stype.id, b.stype.id);                \
    if (a.stype.type.is_int()) {                      \
      return MakeValue(spv::OpS##_Op, a.stype, a, b); \
    } else if (a.stype.type.is_uint()) {              \
      return MakeValue(spv::OpU##_Op, a.stype, a, b); \
    } else {                                          \
      ICHECK(a.stype.type.is_float());                \
      return MakeValue(spv::OpF##_Op, a.stype, a, b); \
    }                                                 \
  }

DEFINE_BUILDER_BINARY_USIGN_OP(Add, Add);
DEFINE_BUILDER_BINARY_USIGN_OP(Sub, Sub);
DEFINE_BUILDER_BINARY_USIGN_OP(Mul, Mul);
DEFINE_BUILDER_BINARY_SIGN_OP(Div, Div);

Value IRBuilder::Mod(Value a, Value b) {
  ICHECK_EQ(a.stype.id, b.stype.id);
  if (a.stype.type.is_int()) {
    return MakeValue(spv::OpSRem, a.stype, a, b);
  } else if (a.stype.type.is_uint()) {
    return MakeValue(spv::OpUMod, a.stype, a, b);
  } else {
    ICHECK(a.stype.type.is_float());
    return MakeValue(spv::OpFRem, a.stype, a, b);
  }
}

#define DEFINE_BUILDER_CMP_OP(_OpName, _Op)                                                     \
  Value IRBuilder::_OpName(Value a, Value b) {                                                  \
    ICHECK_EQ(a.stype.id, b.stype.id);                                                          \
    ICHECK_EQ(a.stype.type.lanes(), b.stype.type.lanes());                                      \
    const auto& bool_type = this->GetSType(DataType::UInt(1).with_lanes(a.stype.type.lanes())); \
    if (a.stype.type.is_int()) {                                                                \
      return MakeValue(spv::OpS##_Op, bool_type, a, b);                                         \
    } else if (a.stype.type.is_uint()) {                                                        \
      return MakeValue(spv::OpU##_Op, bool_type, a, b);                                         \
    } else {                                                                                    \
      ICHECK(a.stype.type.is_float());                                                          \
      return MakeValue(spv::OpFOrd##_Op, bool_type, a, b);                                      \
    }                                                                                           \
  }

DEFINE_BUILDER_CMP_OP(LT, LessThan);
DEFINE_BUILDER_CMP_OP(LE, LessThanEqual);
DEFINE_BUILDER_CMP_OP(GT, GreaterThan);
DEFINE_BUILDER_CMP_OP(GE, GreaterThanEqual);

#define DEFINE_BUILDER_CMP_UOP(_OpName, _Op)                                                    \
  Value IRBuilder::_OpName(Value a, Value b) {                                                  \
    ICHECK_EQ(a.stype.id, b.stype.id);                                                          \
    ICHECK_EQ(a.stype.type.lanes(), b.stype.type.lanes());                                      \
    const auto& bool_type = this->GetSType(DataType::UInt(1).with_lanes(a.stype.type.lanes())); \
    if (a.stype.type.is_int() || a.stype.type.is_uint()) {                                      \
      return MakeValue(spv::OpI##_Op, bool_type, a, b);                                         \
    } else {                                                                                    \
      ICHECK(a.stype.type.is_float());                                                          \
      return MakeValue(spv::OpFOrd##_Op, bool_type, a, b);                                      \
    }                                                                                           \
  }

DEFINE_BUILDER_CMP_UOP(EQ, Equal);
DEFINE_BUILDER_CMP_UOP(NE, NotEqual);

Value IRBuilder::Select(Value cond, Value a, Value b) {
  ICHECK_EQ(a.stype.id, b.stype.id);
  ICHECK_EQ(cond.stype.type.element_of(), DataType::UInt(1));
  return MakeValue(spv::OpSelect, a.stype, cond, a, b);
}

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
