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

#include <tvm/runtime/c_runtime_api.h>
#include <dmlc/logging.h>
#include <type_traits>

namespace tvm {
namespace runtime {
/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
 public:
  /*! \brief Type code for the DataType. */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = TVMTypeCode::kTVMOpaqueHandle,
  };
  /*! \brief default constructor */
  DataType() {}
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit DataType(DLDataType dtype)
      : data_(dtype) {}
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   */
  DataType(int code, int bits, int lanes) {
    data_.code = static_cast<uint8_t>(code);
    data_.bits = static_cast<uint8_t>(bits);
    data_.lanes = static_cast<uint16_t>(lanes);
  }
  /*! \return The type code. */
  int code() const {
    return static_cast<int>(data_.code);
  }
  /*! \return number of bits in the data. */
  int bits() const {
    return static_cast<int>(data_.bits);
  }
  /*! \return number of bytes to store each scalar. */
  int bytes() const {
    return (bits() + 7) / 8;
  }
  /*! \return number of lanes in the data. */
  int lanes() const {
    return static_cast<int>(data_.lanes);
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const {
    return lanes() == 1;
  }
  /*! \return whether type is a scalar type. */
  bool is_bool() const {
    return code() == DataType::kUInt && bits() == 1;
  }
  /*! \return whether type is a float type. */
  bool is_float() const {
    return code() == DataType::kFloat;
  }
  /*! \return whether type is a float16 type. */
  bool is_float16() const {
    return is_float() && bits() == 16;
  }
  /*! \return whether type is an int type. */
  bool is_int() const {
    return code() == DataType::kInt;
  }
  /*! \return whether type is an uint type. */
  bool is_uint() const {
    return code() == DataType::kUInt;
  }
  /*! \return whether type is a handle type. */
  bool is_handle() const {
    return code() == DataType::kHandle;
  }
  /*! \return whether type is a vector type. */
  bool is_vector() const {
    return lanes() > 1;
  }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  DataType with_lanes(int lanes) const {
    return DataType(data_.code, data_.bits, lanes);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  DataType with_bits(int bits) const {
    return DataType(data_.code, bits, data_.lanes);
  }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  DataType element_of() const {
    return with_lanes(1);
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator==(const DataType& other) const {
    return
        data_.code == other.data_.code &&
        data_.bits == other.data_.bits &&
        data_.lanes == other.data_.lanes;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compre against.
   * \return The comparison resilt.
   */
  bool operator!=(const DataType& other) const {
    return !operator==(other);
  }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType () const {
    return data_;
  }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static DataType Int(int bits, int lanes = 1) {
    return DataType(kDLInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType UInt(int bits, int lanes = 1) {
    return DataType(kDLUInt, bits, lanes);
  }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float(int bits, int lanes = 1) {
    return DataType(kDLFloat, bits, lanes);
  }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Bool(int lanes = 1) {
    return DataType::UInt(1, lanes);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Handle(int bits = 64, int lanes = 1) {
    return DataType(kHandle, bits, lanes);
  }
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
  if (dtype == DataType::Bool()) return 1;
  CHECK_EQ(data_bits % 8, 0U)
      << "Need to load/store by multiple of bytes";
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
}  // namespace runtime

using DataType = runtime::DataType;

}  // namespace tvm
#endif  //  TVM_RUNTIME_DATA_TYPE_H_
