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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/source_map.h>
#include <tvm/node/node.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <utility>

namespace tvm {
namespace relax {

/*!
 * \brief Opaque object.
 */
class ObjectStructInfoNode : public StructInfoNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ObjectStructInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ObjectStructInfo", ObjectStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to ObjectStructInfoNode.
 * \sa ObjectStructInfoNode
 */
class ObjectStructInfo : public StructInfo {
 public:
  TVM_DLL ObjectStructInfo(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ObjectStructInfo, StructInfo, ObjectStructInfoNode);
};

/*!
 * \brief Primitive value.
 */
class PrimStructInfoNode : public StructInfoNode {
 public:
  /*! \brief Underlying primitive value, if known */
  ffi::Optional<PrimExpr> value;

  /*! \brief Underlying data type of the primitive value */
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimStructInfoNode>()
        .def_ro("value", &PrimStructInfoNode::value)
        .def_ro("dtype", &PrimStructInfoNode::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.PrimStructInfo", PrimStructInfoNode, StructInfoNode);
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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimStructInfo, StructInfo, PrimStructInfoNode);
};

/*!
 * \brief StructInfo of shape value.
 */
class ShapeStructInfoNode : public StructInfoNode {
 public:
  /*! \brief optionally stores the symbolic value patterns of the shape */
  ffi::Optional<ffi::Array<PrimExpr>> values;
  /*!
   * \brief The number of dimension of the shape, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim;

  /*! \return Whether the struct info contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeStructInfoNode>()
        .def_ro("values", &ShapeStructInfoNode::values)
        .def_ro("ndim", &ShapeStructInfoNode::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ShapeStructInfo", ShapeStructInfoNode, StructInfoNode);
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
  TVM_DLL ShapeStructInfo(ffi::Array<PrimExpr> values, Span span = Span());
  /*!
   * \brief Construction with known unknown symbolic shape patterns.
   * \param ndim Number of dimensions -- can be kUnknownNDim
   * \param span The span of the AST.
   */
  TVM_DLL ShapeStructInfo(int ndim, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeStructInfo, StructInfo, ShapeStructInfoNode);
};

/*!
 * \brief StructInfo of Tensor.
 */
class TensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief optionally store the shape expression of the tensor.
   * \note shape must be normalized: it can only be std::nullopt or ShapeExpr or Var.
   */
  ffi::Optional<Expr> shape;
  /*! \brief The virtual device, indicates where the tensor
   *  is expected to be executed.
   */
  ffi::Optional<VDevice> vdevice;
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
  ffi::Optional<ffi::Array<PrimExpr>> GetShape() const {
    if (!shape.defined()) return {};
    ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(this->shape.value()->struct_info_);
    return shape_sinfo->values;
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorStructInfoNode>()
        .def_ro("shape", &TensorStructInfoNode::shape)
        .def_ro("dtype", &TensorStructInfoNode::dtype)
        .def_ro("vdevice", &TensorStructInfoNode::vdevice)
        .def_ro("ndim", &TensorStructInfoNode::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.TensorStructInfo", TensorStructInfoNode, StructInfoNode);
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
  TVM_DLL TensorStructInfo(Expr shape, DataType dtype,
                           ffi::Optional<VDevice> vdevice = std::nullopt, Span span = Span());

  /*!
   * \brief Construction with an unknown shape expression.
   * \param dtype The data type of tensor's elements.
   * \param ndim The number of dimensions
   * \param vdevice The virtual device.
   * \param span The span of the AST.
   */
  TVM_DLL TensorStructInfo(DataType dtype, int ndim, ffi::Optional<VDevice> vdevice = std::nullopt,
                           Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorStructInfo, StructInfo, TensorStructInfoNode);
};

/*!
 * \brief StructInfo of Tuple.
 */
class TupleStructInfoNode : public StructInfoNode {
 public:
  /*! \brief The struct info of tuple fields. */
  ffi::Array<StructInfo> fields;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleStructInfoNode>().def_ro("fields", &TupleStructInfoNode::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.TupleStructInfo", TupleStructInfoNode, StructInfoNode);
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
  TVM_DLL TupleStructInfo(ffi::Array<StructInfo> fields, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleStructInfo, StructInfo, TupleStructInfoNode);
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
   * \note When params is std::nullopt means the function can take arbitrary number of arguments.
   *       We define such functions as Opaque function.
   */
  ffi::Optional<ffi::Array<StructInfo>> params;
  /*!
   * \brief The struct info of the function's return value.
   */
  StructInfo ret;
  /*!
   * \brief Derivation function of opaque functions that may take any number of parameters.
   * \note When derive_func is not empty, then params should be std::nullopt,
   *       ret should be ObjectStructInfo()
   */
  ffi::Optional<StructInfoDeriveFunc> derive_func;
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncStructInfoNode>()
        .def_ro("params", &FuncStructInfoNode::params, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("ret", &FuncStructInfoNode::ret)
        .def_ro("derive_func", &FuncStructInfoNode::derive_func)
        .def_ro("purity", &FuncStructInfoNode::purity);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.FuncStructInfo", FuncStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to FuncStructInfoNode.
 * \sa FuncStructInfoNode
 */
class FuncStructInfo : public StructInfo {
 public:
  explicit FuncStructInfo(ObjectPtr<FuncStructInfoNode> data) : StructInfo(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
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
  TVM_DLL FuncStructInfo(ffi::Array<StructInfo> params, StructInfo ret, bool purity = true,
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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncStructInfo, StructInfo, FuncStructInfoNode);
};

/*!
 * \brief Match and check if expr have StructInfo T and return it.
 *
 * \param expr The input expression.
 * \return The result of match.
 * \tparam T the underlying structure info type
 */
template <typename T>
inline ffi::Optional<T> MatchStructInfo(const Expr& expr) {
  using TNode = typename T::ContainerType;
  if (const TNode* ptr = expr->struct_info_.as<TNode>()) {
    return ffi::GetRef<T>(ptr);
  } else {
    return std::nullopt;
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
  return ffi::GetRef<StructInfo>(ptr);
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
