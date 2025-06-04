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
/*
 * \file tvm/runtime/data_type.h
 * \brief Primitive runtime data type.
 */
// Acknowledgement: DataType structure design originates from Halide.
#ifndef TVM_RUNTIME_DATA_TYPE_H_
#define TVM_RUNTIME_DATA_TYPE_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/logging.h>

#include <cstring>
#include <string>
#include <type_traits>

namespace tvm {
namespace runtime {

using tvm_index_t = ffi::Shape::index_type;

/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
 public:
  /*!
   * \brief Type code for the DataType.
   *
   * DLPack consistency:
   * 1) kInt is consistent with kDLInt
   * 2) kUInt is consistent with kDLUInt
   * 3) kFloat is consistent with kDLFloat
   */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = kDLOpaqueHandle,
    kBFloat = kDLBfloat,
    kFloat8_e3m4 = kDLFloat8_e3m4,
    kFloat8_e4m3 = kDLFloat8_e4m3,
    kFloat8_e4m3b11fnuz = kDLFloat8_e4m3b11fnuz,
    kFloat8_e4m3fn = kDLFloat8_e4m3fn,
    kFloat8_e4m3fnuz = kDLFloat8_e4m3fnuz,
    kFloat8_e5m2 = kDLFloat8_e5m2,
    kFloat8_e5m2fnuz = kDLFloat8_e5m2fnuz,
    kFloat8_e8m0fnu = kDLFloat8_e8m0fnu,
    kFloat6_e2m3fn = kDLFloat6_e2m3fn,
    kFloat6_e3m2fn = kDLFloat6_e3m2fn,
    kFloat4_e2m1fn = kDLFloat4_e2m1fn,
    kCustomBegin = 129
  };
  /*! \brief default constructor */
  DataType() { data_ = DataType::Void(); }
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit DataType(DLDataType dtype) : data_(dtype) {}
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   */
  DataType(int code, int bits, int lanes, bool is_scalable = false) {
    data_.code = static_cast<uint8_t>(code);
    data_.bits = static_cast<uint8_t>(bits);
    if (is_scalable) {
      ICHECK(lanes > 1) << "Invalid value for vscale factor" << lanes;
    }
    data_.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);
    if (code == kBFloat) {
      ICHECK_EQ(bits, 16);
    }
    if (code == kFloat8_e3m4 || code == kFloat8_e4m3 || code == kFloat8_e4m3b11fnuz ||
        code == kFloat8_e4m3fn || code == kFloat8_e4m3fnuz || code == kFloat8_e5m2 ||
        code == kFloat8_e5m2fnuz || code == kFloat8_e8m0fnu) {
      ICHECK_EQ(bits, 8);
    }
    if (code == kFloat6_e2m3fn || code == kFloat6_e3m2fn) {
      ICHECK_EQ(bits, 6);
    }
    if (code == kFloat4_e2m1fn) {
      ICHECK_EQ(bits, 4);
    }
  }
  /*! \return The type code. */
  int code() const { return static_cast<int>(data_.code); }
  /*! \return number of bits in the data. */
  int bits() const { return static_cast<int>(data_.bits); }
  /*! \return number of bytes to store each scalar. */
  int bytes() const { return (bits() + 7) / 8; }
  /*! \return number of lanes in the data. */
  int lanes() const {
    int lanes_as_int = static_cast<int16_t>(data_.lanes);
    if (lanes_as_int < 0) {
      LOG(FATAL) << "Can't fetch the lanes of a scalable vector at a compile time.";
    }
    return lanes_as_int;
  }
  /*! \return the integer multiplier of vscale in a scalable vector. */
  int vscale_factor() const {
    int lanes_as_int = static_cast<int16_t>(data_.lanes);
    if (lanes_as_int >= -1) {
      LOG(FATAL) << "A fixed length vector doesn't have a vscale factor.";
    }
    return -lanes_as_int;
  }
  /*! \return get vscale factor or lanes depending on scalability of the vector. */
  int get_lanes_or_vscale_factor() const {
    return is_scalable_vector() ? vscale_factor() : lanes();
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const { return !is_scalable_vector() && lanes() == 1; }
  /*! \return whether type is a scalar type. */
  bool is_bool() const { return code() == DataType::kUInt && bits() == 1; }
  /*! \return whether type is a float type. */
  bool is_float() const { return code() == DataType::kFloat; }
  /*! \return whether type is a bfloat type. */
  bool is_bfloat() const { return code() == DataType::kBFloat; }
  /*! \return whether type is any 8-bit custom Float8 variant. */
  bool is_float8() const {
    return bits() == 8 &&
           (code() == DataType::kFloat8_e3m4 || code() == DataType::kFloat8_e4m3 ||
            code() == DataType::kFloat8_e4m3b11fnuz || code() == DataType::kFloat8_e4m3fn ||
            code() == DataType::kFloat8_e4m3fnuz || code() == DataType::kFloat8_e5m2 ||
            code() == DataType::kFloat8_e5m2fnuz || code() == DataType::kFloat8_e8m0fnu);
  }
  /*! \return whether type is any 6-bit custom Float6 variant. */
  bool is_float6() const {
    return bits() == 6 &&
           (code() == DataType::kFloat6_e2m3fn || code() == DataType::kFloat6_e3m2fn);
  }
  /*! \return whether type is the 4-bit custom Float4_e2m1fn variant. */
  bool is_float4() const { return bits() == 4 && code() == DataType::kFloat4_e2m1fn; }
  /*! \return whether type is Float8E3M4. */
  bool is_float8_e3m4() const { return bits() == 8 && code() == DataType::kFloat8_e3m4; }
  /*! \return whether type is Float8E4M3. */
  bool is_float8_e4m3() const { return bits() == 8 && code() == DataType::kFloat8_e4m3; }
  /*! \return whether type is Float8E4M3B11FNUZ. */
  bool is_float8_e4m3b11fnuz() const {
    return bits() == 8 && code() == DataType::kFloat8_e4m3b11fnuz;
  }
  /*! \return whether type is Float8E4M3FN. */
  bool is_float8_e4m3fn() const { return bits() == 8 && code() == DataType::kFloat8_e4m3fn; }
  /*! \return whether type is Float8E4M3FNUZ. */
  bool is_float8_e4m3fnuz() const { return bits() == 8 && code() == DataType::kFloat8_e4m3fnuz; }
  /*! \return whether type is Float8E5M2. */
  bool is_float8_e5m2() const { return bits() == 8 && code() == DataType::kFloat8_e5m2; }
  /*! \return whether type is Float8E5M2FNUZ. */
  bool is_float8_e5m2fnuz() const { return bits() == 8 && code() == DataType::kFloat8_e5m2fnuz; }
  /*! \return whether type is Float8E8M0FNU. */
  bool is_float8_e8m0fnu() const { return bits() == 8 && code() == DataType::kFloat8_e8m0fnu; }
  /*! \return whether type is Float6E2M3FN. */
  bool is_float6_e2m3fn() const { return bits() == 6 && code() == DataType::kFloat6_e2m3fn; }
  /*! \return whether type is Float6E3M2FN. */
  bool is_float6_e3m2fn() const { return bits() == 6 && code() == DataType::kFloat6_e3m2fn; }
  /*! \return whether type is Float4E2M1FN. */
  bool is_float4_e2m1fn() const { return bits() == 4 && code() == DataType::kFloat4_e2m1fn; }
  /*! \return whether type is a float16 type. */
  bool is_float16() const { return is_float() && bits() == 16; }
  /*! \return whether type is a bfloat16 type. */
  bool is_bfloat16() const { return code() == DataType::kBFloat && bits() == 16; }
  /*! \return whether type is an int type. */
  bool is_int() const { return code() == DataType::kInt; }
  /*! \return whether type is an uint type. */
  bool is_uint() const { return code() == DataType::kUInt; }
  /*! \return whether type is a handle type. */
  bool is_handle() const { return code() == DataType::kHandle && !is_void(); }
  /*! \return whether type is a vector type. */
  bool is_scalable_or_fixed_length_vector() const {
    int encoded_lanes = static_cast<int16_t>(data_.lanes);
    return (encoded_lanes < -1) || (1 < encoded_lanes);
  }
  /*! \return Whether the type is a fixed length vector. */
  bool is_fixed_length_vector() const { return static_cast<int16_t>(data_.lanes) > 1; }
  /*! \return Whether the type is a scalable vector. */
  bool is_scalable_vector() const { return static_cast<int16_t>(data_.lanes) < -1; }
  /*! \return whether type is a vector type. */
  bool is_vector() const { return lanes() > 1; }
  /*! \return whether type is a bool vector type. */
  bool is_vector_bool() const { return is_scalable_or_fixed_length_vector() && bits() == 1; }
  /*! \return whether type is a Void type. */
  bool is_void() const { return code() == DataType::kHandle && bits() == 0 && lanes() == 0; }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  DataType with_lanes(int lanes) const { return DataType(data_.code, data_.bits, lanes); }
  /*!
   * \brief Create a new scalable vector data type by changing the vscale multiplier to a specified
   * value. We'll use the data_.lanes field for this value. \param vscale_factor The vscale
   * multiplier. \return A copy of the old DataType with the number of scalable lanes.
   */
  DataType with_scalable_vscale_factor(int vscale_factor) const {
    return DataType(data_.code, data_.bits, -vscale_factor);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  DataType with_bits(int bits) const { return DataType(data_.code, bits, data_.lanes); }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  DataType element_of() const { return with_lanes(1); }
  /*!
   * \brief Assignment operator.
   */
  DataType& operator=(const DataType& rhs) {
    if (this == &rhs) {
      return *this;
    }
    data_ = rhs.data_;
    return *this;
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
  bool operator==(const DataType& other) const {
    return data_.code == other.data_.code && data_.bits == other.data_.bits &&
           data_.lanes == other.data_.lanes;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
  bool operator!=(const DataType& other) const { return !operator==(other); }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType() const { return data_; }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static DataType Int(int bits, int lanes = 1) { return DataType(kDLInt, bits, lanes); }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
  static DataType UInt(int bits, int lanes = 1, bool is_scalable = false) {
    return DataType(kDLUInt, bits, lanes, is_scalable);
  }
  /*!
   * \brief Construct an float type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float(int bits, int lanes = 1) { return DataType(kDLFloat, bits, lanes); }
  /*!
   * \brief Construct an bfloat type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType BFloat(int bits, int lanes = 1) { return DataType(kDLBfloat, bits, lanes); }
  /*!
   * \brief Construct float8 e3m4 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E3M4(int lanes = 1) { return DataType(kFloat8_e3m4, 8, lanes); }

  /*!
   * \brief Construct float8 e4m3 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E4M3(int lanes = 1) { return DataType(kFloat8_e4m3, 8, lanes); }

  /*!
   * \brief Construct float8 e4m3b11fnuz datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E4M3B11FNUZ(int lanes = 1) {
    return DataType(kFloat8_e4m3b11fnuz, 8, lanes);
  }

  /*!
   * \brief Construct float8 e4m3fn datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E4M3FN(int lanes = 1) { return DataType(kFloat8_e4m3fn, 8, lanes); }

  /*!
   * \brief Construct float8 e4m3fnuz datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E4M3FNUZ(int lanes = 1) { return DataType(kFloat8_e4m3fnuz, 8, lanes); }

  /*!
   * \brief Construct float8 e5m2 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E5M2(int lanes = 1) { return DataType(kFloat8_e5m2, 8, lanes); }

  /*!
   * \brief Construct float8 e5m2fnuz datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E5M2FNUZ(int lanes = 1) { return DataType(kFloat8_e5m2fnuz, 8, lanes); }

  /*!
   * \brief Construct float8 e8m0fnu datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float8E8M0FNU(int lanes = 1) { return DataType(kFloat8_e8m0fnu, 8, lanes); }

  /*!
   * \brief Construct float6 e2m3fn datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float6E2M3FN(int lanes = 1) { return DataType(kFloat6_e2m3fn, 6, lanes); }

  /*!
   * \brief Construct float6 e3m2fn datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float6E3M2FN(int lanes = 1) { return DataType(kFloat6_e3m2fn, 6, lanes); }

  /*!
   * \brief Construct float4 e2m1fn datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float4E2M1FN(int lanes = 1) { return DataType(kFloat4_e2m1fn, 4, lanes); }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
  static DataType Bool(int lanes = 1, bool is_scalable = false) {
    return DataType::UInt(1, lanes, is_scalable);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Handle(int bits = 64, int lanes = 1) { return DataType(kHandle, bits, lanes); }
  /*!
   * \brief Construct a Void type.
   * \return The constructed data type.
   */
  static DataType Void() { return DataType(kHandle, 0, 0); }
  /*!
   * \brief Get the corresponding type of TVMShapeIndex.
   * \return The type of TVM shape index.
   */
  static DataType ShapeIndex() {
    if (std::is_signed<tvm_index_t>::value) {
      return DataType::Int(sizeof(tvm_index_t) * 8);
    } else {
      return DataType::UInt(sizeof(tvm_index_t) * 8);
    }
  }

 private:
  DLDataType data_;
};

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DataType dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  // allow bool to exist
  if (dtype == DataType::Bool() || dtype == DataType::Int(4) || dtype == DataType::UInt(4) ||
      dtype == DataType::Int(1) || dtype == DataType::Float4E2M1FN() ||
      dtype == DataType::Float6E2M3FN() || dtype == DataType::Float6E3M2FN()) {
    return 1;
  }
  ICHECK_EQ(data_bits % 8, 0U) << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
/*!
 * \brief Check whether two types are equal .
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

using ffi::DLDataTypeToString;
using ffi::StringToDLDataType;

inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {  // NOLINT(*)
  return os << dtype.operator DLDataType();
}
}  // namespace runtime

using DataType = runtime::DataType;

namespace ffi {

// runtime::DataType
template <>
struct TypeTraits<runtime::DataType> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDataType;

  static TVM_FFI_INLINE void CopyToAnyView(const runtime::DataType& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE void MoveToAny(runtime::DataType src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE std::optional<runtime::DataType> TryCastFromAnyView(const TVMFFIAny* src) {
    auto opt_dtype = TypeTraits<DLDataType>::TryCastFromAnyView(src);
    if (opt_dtype) {
      return runtime::DataType(opt_dtype.value());
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<DLDataType>::CheckAnyStrict(src);
  }

  static TVM_FFI_INLINE runtime::DataType CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return runtime::DataType(TypeTraits<DLDataType>::CopyFromAnyViewAfterCheck(src));
  }

  static TVM_FFI_INLINE std::string TypeStr() { return ffi::StaticTypeKey::kTVMFFIDataType; }
};

}  // namespace ffi
}  // namespace tvm

namespace std {
template <>
struct hash<tvm::DataType> {
  inline int cantor_pairing_function(int a, int b) const { return (a + b) * (a + b + 1) / 2 + b; }
  std::size_t operator()(tvm::DataType const& dtype) const {
    int a = dtype.code();
    int b = dtype.bits();
    int c = dtype.lanes();
    int d = cantor_pairing_function(a, b);
    return cantor_pairing_function(c, d);
  }
};
}  // namespace std

#endif  //  TVM_RUNTIME_DATA_TYPE_H_
