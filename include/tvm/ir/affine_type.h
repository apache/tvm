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
 * \file tvm/ir/affine_type.h
 * \brief Quantized Tensor Types.
 */
#ifndef TVM_IR_AFFINE_TYPE_H_
#define TVM_IR_AFFINE_TYPE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

namespace tvm {

/*!
 * \brief AffineType representation
 * \sa AffineType
 */
class AffineTypeNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "AffineType";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(AffineTypeNode, Object);
};

/*!
 * \brief Managed reference to AffineTypeNode.
 * \sa AffineTypeNode
 */
class AffineType : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AffineType, ObjectRef, AffineTypeNode);
};

/*!
 * \brief TensorAffineType representation
 * \sa TensorAffineType
 *
 *  This Type represents a quantized integer tensor that can be converted
 *  back to real space via the x_real = scale * (x_quant - zero_point)
 */
class TensorAffineTypeNode : public AffineTypeNode {
 public:
  /*! \brief The scale of this type */
  RelayExpr scale;
  /*! \brief The zero point of this type */
  RelayExpr zero_point;
  /*! \brief The data type of this type */
  DataType dtype;
  /*! \brief The axis for per-channel quantization */
  int axis;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("scale", &scale);
    v->Visit("zero_point", &zero_point);
    v->Visit("dtype", &dtype);
    v->Visit("axis", &axis);
  }

  bool SEqualReduce(const TensorAffineTypeNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(scale, other->scale) && equal(zero_point, other->zero_point) &&
           equal(dtype, other->dtype) && equal(axis, other->axis);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(scale);
    hash_reduce(zero_point);
    hash_reduce(dtype);
    hash_reduce(axis);
  }

  static constexpr const char* _type_key = "TensorAffineType";
  TVM_DECLARE_BASE_OBJECT_INFO(TensorAffineTypeNode, AffineTypeNode);
};

/*!
 * \brief Managed reference to AffineTypes.
 * \sa AffineTypeNode
 */
class TensorAffineType : public AffineType {
 public:
  TVM_DLL TensorAffineType(RelayExpr scale, RelayExpr zero_point, DataType dtype, int axis);

  TVM_DEFINE_OBJECT_REF_METHODS(TensorAffineType, AffineType, TensorAffineTypeNode);
};

/*!
 * \brief TupleAffineType representation
 * \sa TupleAffineType
 */
class TupleAffineTypeNode : public AffineTypeNode {
 public:
  /*! \brief The types of this tuple*/
  Array<TensorAffineType> types;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("types", &types); }

  bool SEqualReduce(const TupleAffineTypeNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(types, other->types);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(types);
  }

  static constexpr const char* _type_key = "TupleAffineType";
  TVM_DECLARE_BASE_OBJECT_INFO(TupleAffineTypeNode, AffineTypeNode);
};

/*!
 * \brief Managed reference to TupleAffineTypes.
 * \sa TupleAffineType
 */
class TupleAffineType : public AffineType {
 public:
  TVM_DLL TupleAffineType(Array<TensorAffineType> types);

  TVM_DEFINE_OBJECT_REF_METHODS(TupleAffineType, AffineType, TupleAffineTypeNode);
};

}  // namespace tvm
#endif  // TVM_IR_AFFINE_TYPE_H_
