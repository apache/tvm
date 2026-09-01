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
 * \file tvm/ir/prim/vector_expr.h
 * \brief Primitive vector expressions.
 */
#ifndef TVM_IR_PRIM_VECTOR_EXPR_H_
#define TVM_IR_PRIM_VECTOR_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace prim {

/*!
 * \brief Construct a vector with lanes elements
 *        where its i-th element equals base + i * stride.
 *  This is useful to construct a index for a continuous vector load.
 *
 *  Examples:
 *  - ramp(0, 1, 3) = [0, 1, 2]
 *  - ramp(1, 2, 4) = [1, 3, 5, 7]
 */
class RampNode : public ExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RampNode>()
        .def_ro("base", &RampNode::base)
        .def_ro("stride", &RampNode::stride)
        .def_ro("lanes", &RampNode::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Ramp", RampNode, ExprNode);
};

/*!
 * \brief Managed reference to RampNode
 * \sa RampNode
 */
class Ramp : public PrimExpr {
 public:
  TVM_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Ramp, PrimExpr, RampNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RampNode);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastNode : public ExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr value;
  /*! \brief The number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastNode>()
        .def_ro("value", &BroadcastNode::value)
        .def_ro("lanes", &BroadcastNode::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Broadcast", BroadcastNode, ExprNode);
};

/*!
 * \brief Managed reference to BroadcastNode
 * \sa BroadcastNode
 */
class Broadcast : public PrimExpr {
 public:
  TVM_DLL Broadcast(PrimExpr value, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Broadcast, PrimExpr, BroadcastNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastNode);
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleNode : public ExprNode {
 public:
  /*! \brief the input vectors. */
  ffi::Array<PrimExpr> vectors;
  /*! \brief The indices of each element. */
  ffi::Array<PrimExpr> indices;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShuffleNode>()
        .def_ro("vectors", &ShuffleNode::vectors)
        .def_ro("indices", &ShuffleNode::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Shuffle", ShuffleNode, ExprNode);
};

/*!
 * \brief Managed reference to ShuffleNode
 * \sa ShuffleNode
 */
class Shuffle : public PrimExpr {
 public:
  TVM_DLL Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span = Span());
  TVM_DLL static PrimExpr Concat(ffi::Array<PrimExpr> vectors, Span span = Span());
  TVM_DLL static PrimExpr ExtractElement(PrimExpr vector, int index, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Shuffle, PrimExpr, ShuffleNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleNode);
};

}  // namespace prim
}  // namespace tvm
#endif  // TVM_IR_PRIM_VECTOR_EXPR_H_
