/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RELAX_STRUCT_INFO_H_
#define TVM_RELAX_STRUCT_INFO_H_

#include <tvm/ir/env_func.h>
#include <tvm/ir/source_map.h>
#include <tvm/node/node.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

/*!
 * \brief Opaque object.
 */
class ObjectStructInfoNode : public StructInfoNode {
 public:
  void VisitAttrs(AttrVisitor* v) { v->Visit("span", &span); }

  bool SEqualReduce(const ObjectStructInfoNode* other, SEqualReducer equal) const { return true; }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(0); }

  static constexpr const char* _type_key = "relax.ObjectStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(ObjectStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to ObjectStructInfoNode.
 * \sa ObjectStructInfoNode
 */
class ObjectStructInfo : public StructInfo {
 public:
  TVM_DLL ObjectStructInfo(Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ObjectStructInfo, StructInfo, ObjectStructInfoNode);
};

/*!
 * \brief Primitive value.
 */
class PrimStructInfoNode : public StructInfoNode {
 public:
  /*! \brief Underlying primitive value, if known */
  Optional<PrimExpr> value;

  /*! \brief Underlying data type of the primitive value */
  DataType dtype;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const PrimStructInfoNode* other, SEqualReducer equal) const {
    return equal(value, other->value) && equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
    hash_reduce(dtype);
  }

  static constexpr const char* _type_key = "relax.PrimStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to PrimStructInfoNode.
 * \sa PrimStructInfoNode
 */
class PrimStructInfo : public StructInfo {
 public:
  /* Construct a PrimStructInfo with a known dtype, but unknown value */
  TVM_DLL PrimStructInfo(DataType dtype, Span span = Span());

  /* Construct a PrimStructInfo with a known value */
  TVM_DLL PrimStructInfo(PrimExpr value, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PrimStructInfo, StructInfo, PrimStructInfoNode);
};

/*!
 * \brief StructInfo of shape value.
 */
class ShapeStructInfoNode : public StructInfoNode {
 public:
  /*! \brief optionally stores the symbolic value patterns of the shape */
  Optional<Array<PrimExpr>> values;
  /*!
   * \brief The number of dimension of the shape, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim;

  /*! \return Whether the struct info contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("ndim", &ndim);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ShapeStructInfoNode* other, SEqualReducer equal) const {
    return equal(values, other->values) && equal(ndim, other->ndim);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(values);
    hash_reduce(ndim);
  }

  static constexpr const char* _type_key = "relax.ShapeStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to ShapeStructInfoNode.
 * \sa ShapeStructInfoNode
 */
class ShapeStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with known symbolic shape patterns
   * \param values The symbolic shape values
   * \param span The span of the AST.
   */
  TVM_DLL ShapeStructInfo(Array<PrimExpr> values, Span span = Span());
  /*!
   * \brief Construction with known unknown symbolic shape patterns.
   * \param ndim Number of dimensions -- can be kUnknownNDim
   * \param span The span of the AST.
   */
  TVM_DLL ShapeStructInfo(int ndim, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ShapeStructInfo, StructInfo, ShapeStructInfoNode);
};

/*!
 * \brief StructInfo of Tensor.
 */
class TensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief optionally store the shape expression of the tensor.
   * \note shape must be normalized: it can only be NullOpt or ShapeExpr or Var.
   */
  Optional<Expr> shape;
  /*! \brief The virtual device, indicates where the tensor
   *  is expected to be executed.
   */
  Optional<VDevice> vdevice;
  /*! \brief The content data type, use void to denote the dtype is unknown. */
  DataType dtype;
  /*!
   * \brief The number of dimension of the tensor, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim;

  /*! \return Whether the struct info contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  /*! \return Whether the struct info contains unknown dtype. */
  bool IsUnknownDtype() const { return dtype.is_void(); }

  /*! \return Shape if it is known. */
  Optional<Array<PrimExpr>> GetShape() const {
    if (!shape.defined()) return {};
    ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(this->shape.value()->struct_info_);
    return shape_sinfo->values;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("vdevice", &vdevice);
    v->Visit("ndim", &ndim);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TensorStructInfoNode* other, SEqualReducer equal) const {
    return equal(shape, other->shape) && equal(ndim, other->ndim) &&
           equal(vdevice, other->vdevice) && equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(shape);
    hash_reduce(dtype);
    hash_reduce(vdevice);
    hash_reduce(ndim);
  }

  static constexpr const char* _type_key = "relax.TensorStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to TensorStructInfoNode.
 * \sa TensorStructInfoNode
 */
class TensorStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with a known shape expression.
   * \param shape The shape of the tensor.
   * \param dtype The data type of tensor's elements.
   * \param vdevice The virtual device.
   * \param span The span of the AST.
   *
   * \note shape must already be normalized.
   */
  TVM_DLL TensorStructInfo(Expr shape, DataType dtype, Optional<VDevice> vdevice = NullOpt,
                           Span span = Span());

  /*!
   * \brief Construction with an unknown shape expression.
   * \param dtype The data type of tensor's elements.
   * \param ndim The number of dimensions
   * \param vdevice The virtual device.
   * \param span The span of the AST.
   */
  TVM_DLL TensorStructInfo(DataType dtype, int ndim, Optional<VDevice> vdevice = NullOpt,
                           Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(TensorStructInfo, StructInfo, TensorStructInfoNode);
};

/*!
 * \brief StructInfo of Tuple.
 */
class TupleStructInfoNode : public StructInfoNode {
 public:
  /*! \brief The struct info of tuple fields. */
  Array<StructInfo> fields;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TupleStructInfoNode* other, SEqualReducer equal) const {
    return equal(fields, other->fields);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(fields); }

  static constexpr const char* _type_key = "relax.TupleStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to TupleStructInfoNode.
 * \sa TupleStructInfoNode
 */
class TupleStructInfo : public StructInfo {
 public:
  /*!
   * \brief Constructor
   * \param fields Struct info of tuple fields.
   * \param span The span of the AST.
   */
  TVM_DLL TupleStructInfo(Array<StructInfo> fields, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TupleStructInfo, StructInfo, TupleStructInfoNode);
};

/*!
 * \brief custom-defined StructInfo derivation function.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \return The derived struct info of the call.
 */
using StructInfoDeriveFunc = TypedEnvFunc<StructInfo(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Structure information about function.
 *
 * This data structure contains enough information for us to
 * do best-effort structure information deduction.
 */
class FuncStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief The parameter struct info of the function.
   * \note When params is NullOpt means the function can take arbitrary number of arguments.
   *       We define such functions as Opaque function.
   */
  Optional<Array<StructInfo>> params;
  /*!
   * \brief The struct info of the function's return value.
   */
  StructInfo ret;
  /*!
   * \brief Derivation function of opaque functions that may take any number of parameters.
   * \note When derive_func is not empty, then params should be NullOpt,
   *       ret should be ObjectStructInfo()
   */
  Optional<StructInfoDeriveFunc> derive_func;
  /*!
   * \brief Whether the function is pure.
   * \note This parameter should be set to true only if the function is pure on all inputs.
   *   If the function _may_ have visible side effects, set it to false.
   */
  bool purity;

  /*!
   * \return Whether the func struct info is opaque.
   * \note We define a function as opaque we have no constraints on params.
   */
  bool IsOpaque() const { return !params.defined(); }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("ret", &ret);
    v->Visit("derive_func", &derive_func);
    v->Visit("span", &span);
    v->Visit("purity", &purity);
  }

  bool SEqualReduce(const FuncStructInfoNode* other, SEqualReducer equal) const {
    return equal.DefEqual(params, other->params) && equal(ret, other->ret) &&
           equal(purity, other->purity) && equal(derive_func, other->derive_func);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(params);
    hash_reduce(ret);
    hash_reduce(purity);
    hash_reduce(derive_func);
  }

  static constexpr const char* _type_key = "relax.FuncStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuncStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to FuncStructInfoNode.
 * \sa FuncStructInfoNode
 */
class FuncStructInfo : public StructInfo {
 public:
  /*!
   * \brief Constructor from parameter struct info and return value struct info.
   * \param params The struct info of function parameters.
   * \param ret The return value struct info.
   * \param purity The purity of the function (true by default).
   * \param span The span of the AST.
   *
   * \note If the ret contains variables(tir::Var and relax::Var), they must be deducible from
   * params. If you are unsure, you can always erase ret to static.
   */
  TVM_DLL FuncStructInfo(Array<StructInfo> params, StructInfo ret, bool purity = true,
                         Span span = Span());

  /*!
   * \brief Constructing an opaque function struct info using derive_func.
   *
   * \param derive_func Derivation function.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param span The span of the AST.
   *
   * \return The FuncStructInfo for opaque packedfunc.
   * \note Defaults to an derive func that always return ObjectStructInfo if not specified.
   */
  TVM_DLL static FuncStructInfo OpaqueFunc(StructInfoDeriveFunc derive_func, bool purity = false,
                                           Span span = Span());

  /*!
   * \brief Construct an opaque function using from return struct info.
   *
   * \param ret The struct info of the return value.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param span The span of the AST.
   *
   * \return The FuncStructInfo for opaque packedfunc.
   * \note Defaults to an derive func that always return ObjectStructInfo if not specified.
   */
  TVM_DLL static FuncStructInfo OpaqueFunc(StructInfo ret = ObjectStructInfo(), bool purity = false,
                                           Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(FuncStructInfo, StructInfo, FuncStructInfoNode);
};

/*!
 * \brief Match and check if expr have StructInfo T and return it.
 *
 * \param expr The input expression.
 * \return The result of match.
 * \tparam T the underlying structure info type
 */
template <typename T>
inline Optional<T> MatchStructInfo(const Expr& expr) {
  using TNode = typename T::ContainerType;
  if (const TNode* ptr = expr->struct_info_.as<TNode>()) {
    return GetRef<T>(ptr);
  } else {
    return NullOpt;
  }
}

/*!
 * \brief Get the structure info of a given expr and try to cast it as const T*.
 *
 * \param expr The input expression.
 * \return The pointer. Returns nullptr if the type does not match
 * \tparam T the underlying structure info type
 */
template <typename T>
inline const T* GetStructInfoAs(const Expr& expr) {
  ICHECK(expr->struct_info_.defined())
      << "The struct_info is not populated, check if you have normalized the expr";
  return expr->struct_info_.as<T>();
}

/*!
 * \brief Get the underlying structure info of expr.
 *
 * \param expr The input expression.
 * \return underlying struct info.
 */
inline StructInfo GetStructInfo(const Expr& expr) {
  auto* ptr = expr->struct_info_.as<StructInfoNode>();
  ICHECK(ptr) << "The struct_info is not populated, check if you have normalized the expr";
  return GetRef<StructInfo>(ptr);
}

/*!
 * \brief Whether the expr has void struct info.
 *
 * \param expr The input expression.
 * \return Whether the expr has void struct info.
 */
inline bool HasVoidStructInfo(const Expr& expr) {
  auto* ptr = expr->struct_info_.as<TupleStructInfoNode>();
  return ptr != nullptr && ptr->fields.size() == 0;
}

/*!
 * \brief Update the struct info of an Expr.
 * \param expr The Expr whose struct info to be updated.
 * \param struct_info The struct_info assigned.
 * \note We ensure idempotence, that is we can only update the struct_info of an Expr only
 *  if the original one is nullptr.
 */
TVM_DLL void UpdateStructInfo(Expr expr, StructInfo struct_info);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_STRUCT_INFO_H_
