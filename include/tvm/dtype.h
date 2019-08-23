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
 * \file tvm/dtype.h
 * \brief Data type used in IR.
 */
// Acknowledgement: DataType structure design originates from Halide.
#ifndef TVM_DTYPE_H_
#define TVM_DTYPE_H_

#include "runtime/packed_func.h"

namespace tvm {
class Expr;

/*!
 * \brief Primitive data types in tvm.
 */
class DataType {
 public:
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
    return code() == kDLUInt && bits() == 1;
  }
  /*! \return whether type is a float type. */
  bool is_float() const {
    return code() == kDLFloat;
  }
  /*! \return whether type is an int type. */
  bool is_int() const {
    return code() == kDLInt;
  }
  /*! \return whether type is an uint type. */
  bool is_uint() const {
    return code() == kDLUInt;
  }
  /*! \return whether type is a handle type. */
  bool is_handle() const {
    return code() == kHandle;
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
  // operator overloadings
  bool operator==(const DataType& other) const {
    return
        data_.code == other.data_.code &&
        data_.bits == other.data_.bits &&
        data_.lanes == other.data_.lanes;
  }
  bool operator!=(const DataType& other) const {
    return !operator==(other);
  }
  operator DLDataType () const {
    return data_;
  }
  /*! \return the maximum possible value in this format. */
  TVM_DLL Expr max() const;
  /*! \return the minimum possible value in this format. */
  TVM_DLL Expr min() const;

 private:
  DLDataType data_;
};

/*!
 * \brief Construct an int type.
 * \param bits The number of bits in the type.
 * \param lanes The number of lanes.
 * \return The constructed data type.
 */
inline DataType Int(int bits, int lanes = 1) {
  return DataType(kDLInt, bits, lanes);
}

/*!
 * \brief Construct an uint type.
 * \param bits The number of bits in the type.
 * \param lanes The number of lanes
 * \return The constructed data type.
 */
inline DataType UInt(int bits, int lanes = 1) {
  return DataType(kDLUInt, bits, lanes);
}

/*!
 * \brief Construct a bool type.
 * \param lanes The number of lanes
 * \return The constructed data type.
 */
inline DataType Bool(int lanes = 1) {
  return UInt(1, lanes);
}

/*!
 * \brief Construct an uint type.
 * \param bits The number of bits in the type.
 * \param lanes The number of lanes
 * \return The constructed data type.
 */
inline DataType Float(int bits, int lanes = 1) {
  return DataType(kDLFloat, bits, lanes);
}

/*!
 * \brief Construct a handle type.
 * \param bits The number of bits in the type.
 * \param lanes The number of lanes
 * \return The constructed data type.
 */
inline DataType Handle(int bits = 64, int lanes = 1) {
  return DataType(kHandle, bits, lanes);
}

/*!
 * \brief Get the corresponding type of TVMShapeIndex.
 * \return The type of TVM shape index.
 */
inline DataType TVMShapeIndexType() {
  if (std::is_signed<tvm_index_t>::value) {
    return Int(sizeof(tvm_index_t) * 8);
  } else {
    return UInt(sizeof(tvm_index_t) * 8);
  }
}

/*!
 * \brief Convert DLDataType to DataType.
 * \param t The original type.
 * \return The conversion result.
 */
inline DataType TVMType2Type(DLDataType t) {
  return DataType(t.code, t.bits, t.lanes);
}

/*!
 * \brief Convert DataType to DataType.
 * \param t The original type.
 * \return The conversion result.
 */
inline DLDataType Type2TVMType(DataType t) {
  return t.operator DLDataType();
}

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DataType dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  // allow bool to exist
  if (dtype == Bool()) return 1;
  CHECK_EQ(data_bits % 8, 0U)
      << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

// Overload print function.
inline std::ostream& operator<<(std::ostream& os, DataType dtype) { // NOLINT(*)
  using namespace tvm::runtime;
  return os << dtype.operator DLDataType();
}

// Backward compatibility
using Type = DataType;
}  // namespace tvm
#endif  //  TVM_DTYPE_H_
