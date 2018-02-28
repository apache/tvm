/*!
 *  Copyright (c) 2018 by Contributors
 * \file ir_builder.cc
 * \brief IRBuilder for SPIRV block
 */

#if TVM_VULKAN_RUNTIME

#include "./ir_builder.h"

namespace tvm {
namespace codegen {
namespace spirv {

// implementations

void IRBuilder::InitHeader() {
  CHECK_EQ(header_.size(), 0U);
  header_.push_back(spv::MagicNumber);
  header_.push_back(spv::Version);
  // generator: set to 0, unknown
  header_.push_back(0U);
  // Bound: set during Finalize
  header_.push_back(0U);
  // Schema: reserved
  header_.push_back(0U);
  // shader
  ib_.Begin(spv::OpCapability).Add(spv::CapabilityShader).Commit(&header_);
  // memory model
  ib_.Begin(spv::OpMemoryModel).AddSeq(
        spv::AddressingModelLogical,
        spv::MemoryModelGLSL450).Commit(&entry_);
  this->InitPreDefs();
}

void IRBuilder::InitPreDefs() {
  ext_glsl450_ = ExtInstImport("GLSL.std.450");
  t_int32_ = DeclareType(Int(32));
  t_uint32_ = DeclareType(UInt(32));
  t_bool_ = DeclareType(UInt(1));
  t_fp32_ = DeclareType(Float(32));
  const_i32_zero_ = IntImm(t_int32_, 0);
  // declare void, and void functions
  t_void_.id = id_counter_++;
  ib_.Begin(spv::OpTypeVoid).Add(t_void_).Commit(&global_);
  t_void_func_.id = id_counter_++;
  ib_.Begin(spv::OpTypeFunction)
      .AddSeq(t_void_func_, t_void_).Commit(&global_);
}

SType IRBuilder::GetSType(const Type& dtype) {
  if (dtype == Int(32)) {
    return t_int32_;
  } else if (dtype == UInt(1)) {
    return t_bool_;
  } else if (dtype == Float(32)) {
    return t_fp32_;
  } else if (dtype == UInt(32)) {
    return t_uint32_;
  }
  uint32_t type_key;
  type_key = static_cast<uint32_t>(dtype.code());
  type_key |= static_cast<uint32_t>(dtype.bits()) << 8U;
  type_key |= static_cast<uint32_t>(dtype.lanes()) << 16U;
  auto it = pod_type_tbl_.find(type_key);
  if (it != pod_type_tbl_.end()) {
    return it->second;
  }
  SType t = DeclareType(dtype);
  pod_type_tbl_[type_key] = t;
  return t;
}

SType IRBuilder::GetPointerType(const SType& value_type,
                                spv::StorageClass storage_class) {
  CHECK_NE(storage_class, spv::StorageClassMax);
  auto key = std::make_pair(value_type.id, storage_class);
  auto it = pointer_type_tbl_.find(key);
  if (it != pointer_type_tbl_.end()) {
    return it->second;
  }
  SType t;
  t.id = id_counter_++;
  t.type = Handle();
  t.element_type_id = value_type.id;
  t.storage_class = storage_class;
  ib_.Begin(spv::OpTypePointer)
      .AddSeq(t, storage_class, value_type).Commit(&global_);
  pointer_type_tbl_[key] = t;
  return t;
}

SType IRBuilder::GetStructArrayType(const SType& value_type,
                                    uint32_t num_elems) {
  auto key = std::make_pair(value_type.id, num_elems);
  auto it = struct_array_type_tbl_.find(key);
  if (it != struct_array_type_tbl_.end()) {
    return it->second;
  }

  SType arr_type;
  arr_type.id = id_counter_++;
  arr_type.type = Handle();
  arr_type.element_type_id = value_type.id;

  if (num_elems != 0) {
    Value length = UIntImm(GetSType(UInt(32)), num_elems);
    ib_.Begin(spv::OpTypeArray)
        .AddSeq(arr_type, value_type, length).Commit(&global_);
  } else {
    ib_.Begin(spv::OpTypeRuntimeArray)
        .AddSeq(arr_type, value_type).Commit(&global_);
  }
  int nbits = value_type.type.bits() * value_type.type.lanes();
  CHECK_EQ(nbits % 8, 0);
  uint32_t nbytes = static_cast<uint32_t>(nbits) / 8;
  // decorate the array type.
  this->Decorate(spv::OpDecorate,
                 arr_type, spv::DecorationArrayStride, nbytes);
  // declare struct of array
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.type = Handle();
  struct_type.element_type_id = value_type.id;
  ib_.Begin(spv::OpTypeStruct)
      .AddSeq(struct_type, arr_type).Commit(&global_);
  // decorate the array type.
  ib_.Begin(spv::OpMemberDecorate)
      .AddSeq(struct_type, 0, spv::DecorationOffset, 0)
      .Commit(&decorate_);
  // runtime array are always decorated as BufferBlock(shader storage buffer)
  if (num_elems == 0) {
    this->Decorate(spv::OpDecorate,
                   struct_type, spv::DecorationBufferBlock);
  }
  struct_array_type_tbl_[key] = struct_type;
  return struct_type;
}

Value IRBuilder::StructArrayAccess(const SType& res_type,
                                   Value buffer,
                                   Value index) {
  CHECK(buffer.flag == kStructArrayPtr);
  return MakeValue(spv::OpInBoundsAccessChain,
                   res_type, buffer,
                   const_i32_zero_, index);
}

Value IRBuilder::IntImm(const SType& dtype, int64_t value) {
  return GetConst_(dtype, reinterpret_cast<uint64_t*>(&value));
}

Value IRBuilder::UIntImm(const SType& dtype, uint64_t value) {
  return GetConst_(dtype, &value);
}

Value IRBuilder::FloatImm(const SType& dtype, double value) {
  if (dtype.type.bits() == 64) {
    return GetConst_(dtype, reinterpret_cast<uint64_t*>(&value));
  } else if (dtype.type.bits() == 32) {
    float fvalue = static_cast<float>(value);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(&fvalue);
    uint64_t data = ptr[0];
    return GetConst_(dtype, &data);
  } else {
    CHECK_EQ(dtype.type.bits(), 16);
    return Cast(dtype,
                FloatImm(GetSType(Float(32)), value));
  }
}

Value IRBuilder::BufferArgument(const SType& value_type,
                                uint32_t descriptor_set,
                                uint32_t binding) {
  SType sarr_type = GetStructArrayType(value_type, 0);
  SType ptr_type = GetPointerType(sarr_type, spv::StorageClassUniform);
  Value val = NewValue(ptr_type, kStructArrayPtr);
  ib_.Begin(spv::OpVariable)
      .AddSeq(ptr_type, val, spv::StorageClassUniform).Commit(&global_);
  this->Decorate(spv::OpDecorate,
                 val, spv::DecorationDescriptorSet, descriptor_set);
  this->Decorate(spv::OpDecorate,
                 val, spv::DecorationBinding, binding);
  return val;
}

Value IRBuilder::DeclarePushConstant(const std::vector<SType>& value_types) {
  CHECK_EQ(push_const_.id, 0);
  SType struct_type;
  struct_type.id = id_counter_++;
  struct_type.type = Handle();
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
    Type t = value_types[i].type;
    uint32_t nbits = t.bits() * t.lanes();
    CHECK_EQ(nbits % 8 , 0);
    offset += nbits / 8;
  }
  // Decorate push constants as UBO
  this->Decorate(spv::OpDecorate, struct_type, spv::DecorationBlock);

  SType ptr_type = GetPointerType(
      struct_type, spv::StorageClassPushConstant);
  Value val = NewValue(ptr_type, kPushConstantPtr);
  ib_.Begin(spv::OpVariable)
      .AddSeq(ptr_type, val, spv::StorageClassPushConstant).Commit(&global_);
  return val;
}

Value IRBuilder::GetPushConstant(
    Value ptr_push_const, const SType& v_type, uint32_t index) {
  SType ptr_vtype = this->GetPointerType(v_type, spv::StorageClassPushConstant);
  Value ptr = this->MakeValue(
      spv::OpAccessChain, ptr_vtype, ptr_push_const,
      IntImm(t_int32_, static_cast<int64_t>(index)));
  return this->MakeValue(spv::OpLoad, v_type, ptr);
}

Value IRBuilder::DeclareKenrelFunction(const std::string& name) {
  Value val = NewValue(t_void_func_, kFunction);
  ib_.Begin(spv::OpEntryPoint)
      .AddSeq(spv::ExecutionModelGLCompute, val, name)
      .Commit(&entry_);
  return val;
}

void IRBuilder::StartFunction(const Value& func) {
  CHECK_EQ(func.flag, kFunction);
  this->MakeInst(
      spv::OpFunction, t_void_, func, 0, t_void_func_);
  spirv::Label start_label = this->NewLabel();
  this->StartLabel(start_label);
}

void IRBuilder::SetLocalSize(const Value& func,
                             uint32_t local_size[3]) {
  CHECK_EQ(func.flag, kFunction);
  ib_.Begin(spv::OpExecutionMode)
      .AddSeq(func, spv::ExecutionModeLocalSize,
              local_size[0], local_size[1], local_size[2])
      .Commit(&exec_mode_);
}

Value IRBuilder::Allocate(const SType& value_type,
                          uint32_t num_elems,
                          spv::StorageClass storage_class) {
  CHECK_NE(num_elems, 0U);
  SType sarr_type = GetStructArrayType(value_type, num_elems);
  SType ptr_type = GetPointerType(sarr_type, storage_class);
  Value val = NewValue(ptr_type, kStructArrayPtr);
  if (storage_class == spv::StorageClassFunction) {
    ib_.Begin(spv::OpVariable)
        .AddSeq(ptr_type, val, storage_class).Commit(&function_);
  } else {
    ib_.Begin(spv::OpVariable)
        .AddSeq(ptr_type, val, storage_class).Commit(&global_);
  }
  return val;
}

Value IRBuilder::GetWorkgroupID(uint32_t dim_index) {
  if (workgroup_id_.id == 0) {
    SType vec3_type = this->GetSType(Int(32).with_lanes(3));
    SType ptr_type = this->GetPointerType(
        vec3_type, spv::StorageClassInput);
    workgroup_id_ = NewValue(ptr_type, kVectorPtr);
    ib_.Begin(spv::OpVariable)
        .AddSeq(ptr_type, workgroup_id_, spv::StorageClassInput)
        .Commit(&global_);
    this->Decorate(spv::OpDecorate, workgroup_id_,
                   spv::DecorationBuiltIn, spv::BuiltInWorkgroupId);
  }
  SType pint_type = this->GetPointerType(t_int32_, spv::StorageClassInput);
  Value ptr = this->MakeValue(
      spv::OpAccessChain, pint_type, workgroup_id_,
      IntImm(t_int32_, static_cast<int64_t>(dim_index)));
  return this->MakeValue(spv::OpLoad, t_int32_, ptr);
}

Value IRBuilder::GetLocalID(uint32_t dim_index) {
  if (local_id_.id == 0) {
    SType vec3_type = this->GetSType(Int(32).with_lanes(3));
    SType ptr_type = this->GetPointerType(vec3_type, spv::StorageClassInput);
    local_id_ = NewValue(ptr_type, kVectorPtr);
    ib_.Begin(spv::OpVariable)
        .AddSeq(ptr_type, local_id_, spv::StorageClassInput)
        .Commit(&global_);
    this->Decorate(spv::OpDecorate, local_id_,
                   spv::DecorationBuiltIn, spv::BuiltInLocalInvocationId);
  }
  SType pint_type = this->GetPointerType(t_int32_, spv::StorageClassInput);
  Value ptr = this->MakeValue(
      spv::OpAccessChain, pint_type, local_id_,
      UIntImm(t_int32_, static_cast<int64_t>(dim_index)));
  return this->MakeValue(spv::OpLoad, t_int32_, ptr);
}

Value IRBuilder::GetConst_(const SType& dtype, const uint64_t* pvalue) {
  auto key = std::make_pair(dtype.id, pvalue[0]);
  auto it = const_tbl_.find(key);
  if (it != const_tbl_.end()) {
    return it->second;
  }
  CHECK_LE(dtype.type.bits(), 64);
  Value ret = NewValue(dtype, kConstant);
  if (dtype.type == UInt(1)) {
    // bool types.
    if (*pvalue) {
      ib_.Begin(spv::OpConstantTrue).AddSeq(ret);
    } else {
      ib_.Begin(spv::OpConstantFalse).AddSeq(ret);
    }
  } else {
    // Integral/floating-point types.
    ib_.Begin(spv::OpConstant).AddSeq(dtype, ret);
    uint64_t mask = 0xFFFFFFFFUL;
    ib_.Add(static_cast<uint32_t>(pvalue[0] & mask));
    if (dtype.type.bits() > 32) {
      if (dtype.type.is_int()) {
        int64_t sign_mask = 0xFFFFFFFFL;
        const int64_t* sign_ptr =
            reinterpret_cast<const int64_t*>(pvalue);
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

SType IRBuilder::DeclareType(const Type& dtype) {
  if (dtype.lanes() == 1) {
    SType t;
    t.id = id_counter_++;
    t.type = dtype;
    if (dtype.bits() == 1) {
      CHECK(dtype.is_uint());
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
    ib_.Begin(spv::OpTypeVector).AddSeq(
        t, base_type, dtype.lanes()).Commit(&global_);
    return t;
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
  CHECK_EQ(phi.instr.WordCount(), 2 * num_incoming + 3);
  return phi;
}

Value IRBuilder::CallGLSL450(const SType& ret_type,
                             uint32_t inst_id,
                             const std::vector<Value>& args) {
  Value val = NewValue(ret_type, kNormal);
  ib_.Begin(spv::OpExtInst)
      .AddSeq(ret_type, val, ext_glsl450_, inst_id);
  for (const Value& v : args) {
    ib_.Add(v);
  }
  ib_.Commit(&function_);
  return val;
}

Value IRBuilder::Concat(const std::vector<Value>& vec) {
  bool is_const = vec[0].flag == kConstant;
  Type etype = vec[0].stype.type;
  int lanes = etype.lanes();
  for (size_t i = 1; i < vec.size(); ++i) {
    CHECK_EQ(etype, vec[i].stype.type.element_of())
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
  CHECK_NE(value.stype.id, 0U);
  if (value.stype.id == dst_type.id) return value;
  const tvm::Type& from = value.stype.type;
  const tvm::Type& to = dst_type.type;
  CHECK_EQ(from.lanes(), to.lanes());

  if (from.is_int() && to.is_int()) {
    return MakeValue(spv::OpSConvert, dst_type, value);
  } else if (from.is_uint() && to.is_uint()) {
    return MakeValue(spv::OpUConvert, dst_type, value);
  } else if (from.is_uint() && to.is_int()) {
    if (from.bits() != to.bits()) {
      value = MakeValue(
          spv::OpUConvert, GetSType(from.with_bits(to.bits())), value);
    }
    return MakeValue(spv::OpBitcast, dst_type, value);
  } else if (from.is_int() && to.is_uint()) {
    if (from.bits() != to.bits()) {
      value = MakeValue(
          spv::OpSConvert, GetSType(from.with_bits(to.bits())), value);
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
    LOG(FATAL) << "do not support type cast from "
               << from << " to " << to;
    return Value();
  }
}

#define DEFINE_BUILDER_BINARY_USIGN_OP(_OpName, _Op)              \
  Value IRBuilder::_OpName(Value a, Value b) {                    \
    CHECK_EQ(a.stype.id, b.stype.id);                             \
    if (a.stype.type.is_int() || a.stype.type.is_uint()) {        \
      return MakeValue(spv::OpI ## _Op, a.stype, a, b);           \
    } else {                                                      \
      CHECK(a.stype.type.is_float());                             \
      return MakeValue(spv::OpF ## _Op, a.stype, a, b);           \
    }                                                             \
  }

#define DEFINE_BUILDER_BINARY_SIGN_OP(_OpName, _Op)               \
  Value IRBuilder::_OpName(Value a, Value b) {                    \
    CHECK_EQ(a.stype.id, b.stype.id);                             \
    if (a.stype.type.is_int()) {                                   \
      return MakeValue(spv::OpS ## _Op, a.stype, a, b);            \
    } else if (a.stype.type.is_uint()) {                           \
      return MakeValue(spv::OpU ## _Op, a.stype, a, b);            \
    } else {                                                       \
      CHECK(a.stype.type.is_float());                              \
      return MakeValue(spv::OpF ## _Op, a.stype, a, b);            \
    }                                                              \
  }

DEFINE_BUILDER_BINARY_USIGN_OP(Add, Add);
DEFINE_BUILDER_BINARY_USIGN_OP(Sub, Sub);
DEFINE_BUILDER_BINARY_USIGN_OP(Mul, Mul);
DEFINE_BUILDER_BINARY_SIGN_OP(Div, Div);

Value IRBuilder::Mod(Value a, Value b) {
  CHECK_EQ(a.stype.id, b.stype.id);
  if (a.stype.type.is_int()) {
    return MakeValue(spv::OpSRem, a.stype, a, b);
  } else if (a.stype.type.is_uint()) {
    return MakeValue(spv::OpUMod, a.stype, a, b);
  } else {
    CHECK(a.stype.type.is_float());
    return MakeValue(spv::OpFRem, a.stype, a, b);
  }
}


#define DEFINE_BUILDER_CMP_OP(_OpName, _Op)                        \
  Value IRBuilder:: _OpName(Value a, Value b) {                    \
    CHECK_EQ(a.stype.id, b.stype.id);                              \
    if (t_bool_.id == 0) {                                         \
      t_bool_ = DeclareType(UInt(1));                              \
    }                                                              \
    if (a.stype.type.is_int()) {                                   \
      return MakeValue(spv::OpS ## _Op, t_bool_, a, b);            \
    } else if (a.stype.type.is_uint()) {                           \
      return MakeValue(spv::OpU ## _Op, t_bool_, a, b);            \
    } else {                                                       \
      CHECK(a.stype.type.is_float());                              \
      return MakeValue(spv::OpFOrd ## _Op, t_bool_, a, b);         \
    }                                                              \
  }

DEFINE_BUILDER_CMP_OP(LT, LessThan);
DEFINE_BUILDER_CMP_OP(LE, LessThanEqual);
DEFINE_BUILDER_CMP_OP(GT, GreaterThan);
DEFINE_BUILDER_CMP_OP(GE, GreaterThanEqual);

#define DEFINE_BUILDER_CMP_UOP(_OpName, _Op)                       \
  Value IRBuilder:: _OpName(Value a, Value b) {                    \
    CHECK_EQ(a.stype.id, b.stype.id);                              \
    if (t_bool_.id == 0) {                                         \
      t_bool_ = DeclareType(UInt(1));                              \
    }                                                              \
    if (a.stype.type.is_int() || a.stype.type.is_uint()) {         \
      return MakeValue(spv::OpI ## _Op, t_bool_, a, b);            \
    } else {                                                       \
      CHECK(a.stype.type.is_float());                              \
      return MakeValue(spv::OpFOrd ## _Op, t_bool_, a, b);         \
    }                                                              \
  }

DEFINE_BUILDER_CMP_UOP(EQ, Equal);
DEFINE_BUILDER_CMP_UOP(NE, NotEqual);

Value IRBuilder::Select(Value cond, Value a, Value b) {
  CHECK_EQ(a.stype.id, b.stype.id);
  CHECK_EQ(cond.stype.type, UInt(1));
  return MakeValue(spv::OpSelect, a.stype, cond, a, b);
}

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_VULKAN_RUNTIME
