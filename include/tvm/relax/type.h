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
 * \file tvm/relax/type.h
 * \brief Relax types, including the richer dependent Relax type nodes.
 */
#ifndef TVM_RELAX_TYPE_H_
#define TVM_RELAX_TYPE_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/expr.h>

#include <string>
#include <utility>

namespace tvm {
namespace relax {

using Expr = RelaxExpr;
using ExprNode = RelaxExprNode;

class BlockBuilder;
class Call;

/*! \brief Indicates the number of dimensions of a tensor is unknown at compile time. */
static constexpr int kUnknownNDim = -1;

using tvm::TupleType;
using tvm::TupleTypeNode;

class PackedFuncTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PackedFuncTypeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.PackedFuncType", PackedFuncTypeNode, TypeNode);
};

class PackedFuncType : public Type {
 public:
  TVM_DLL PackedFuncType(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PackedFuncType, Type, PackedFuncTypeNode);
};

/*!
 * \brief Base type of all Relax type information.
 *
 * Type stores possible type information deduced during compile-time.
 * It encapsulates both static type and runtime information such as shape.
 *
 * Type of each non-primitive Expr can be deduced during compilation in a
 * "best-effort" manner.
 *
 * When ty appears in function parameter and return signatures, it
 * implies a runtime check that matches the type information with the value.
 *
 * When it appears in Expr, it follows "assume-semantics", which means the
 * compiler will take the deduced information as it is and only do best effort
 * proofs and checks.
 *
 * Each type can be uniquely erased to a static-type.  The compiler will
 * still compile the code, with less information, when we erase to the static
 * type.
 *
 * If a Type contains an Expr field, then that field must already be
 * normalized through NormalizeArg.  This invariant is checked in constructors
 * and simplifies assumptions during type deduction.
 */
/*!
 * \brief Opaque object.
 */
class ObjectTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ObjectTypeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ObjectType", ObjectTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to ObjectTypeNode.
 * \sa ObjectTypeNode
 */
class ObjectType : public Type {
 public:
  TVM_DLL ObjectType(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ObjectType, Type, ObjectTypeNode);
};

/*!
 * \brief Type of shape value.
 */
class ShapeTypeNode : public TypeNode {
 public:
  /*! \brief optionally stores the symbolic value patterns of the shape */
  ffi::Optional<ffi::Array<PrimExpr>> values;
  /*!
   * \brief The number of dimension of the shape, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim{kUnknownNDim};

  /*! \return Whether the type contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeTypeNode>()
        .def_ro("values", &ShapeTypeNode::values)
        .def_ro("ndim", &ShapeTypeNode::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ShapeType", ShapeTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to ShapeTypeNode.
 * \sa ShapeTypeNode
 */
class ShapeType : public Type {
 public:
  /*!
   * \brief Construction with known symbolic shape patterns
   * \param values The symbolic shape values
   * \param span The span of the AST.
   */
  TVM_DLL ShapeType(ffi::Array<PrimExpr> values, Span span = Span());
  /*!
   * \brief Construction with known unknown symbolic shape patterns.
   * \param ndim Number of dimensions -- can be kUnknownNDim
   * \param span The span of the AST.
   */
  TVM_DLL ShapeType(int ndim, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeType, Type, ShapeTypeNode);
};

/*!
 * \brief Type of Tensor.
 */
class TensorTypeNode : public TypeNode {
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
  /*! \brief The content dtype, use void to denote the dtype is unknown. */
  tvm::PrimType dtype{DLDataType{kDLOpaqueHandle, 0, 0}};
  /*!
   * \brief The number of dimension of the tensor, can be unknown.
   * \sa kUnknownNDim
   */
  int ndim{kUnknownNDim};

  /*! \return Whether the type contains unknown ndim. */
  bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  /*! \return Whether the type contains unknown dtype. */
  bool IsUnknownDtype() const { return dtype->dtype == DLDataType{kDLOpaqueHandle, 0, 0}; }

  /*! \return Shape if it is known. */
  ffi::Optional<ffi::Array<PrimExpr>> GetShape() const {
    if (!shape.defined()) return {};
    const Expr& shape_expr = this->shape.value();
    if (!shape_expr->ty.defined()) return {};
    if (const auto* shape_ty = shape_expr->ty.as<ShapeTypeNode>()) {
      return shape_ty->values;
    }
    return {};
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorTypeNode>()
        .def_ro("shape", &TensorTypeNode::shape)
        .def_ro("dtype", &TensorTypeNode::dtype)
        .def_ro("vdevice", &TensorTypeNode::vdevice)
        .def_ro("ndim", &TensorTypeNode::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.TensorType", TensorTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorTypeNode.
 * \sa TensorTypeNode
 */
class TensorType : public Type {
 public:
  explicit TensorType(ffi::ObjectPtr<TensorTypeNode> data) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }

  /*!
   * \brief Construction with a known shape expression.
   * \param shape The shape of the tensor.
   * \param dtype The data type of tensor's elements.
   * \param vdevice The virtual device.
   * \param span The span of the AST.
   *
   * \note shape must already be normalized.
   */
  TVM_DLL TensorType(Expr shape, tvm::PrimType dtype, ffi::Optional<VDevice> vdevice = std::nullopt,
                     Span span = Span());

  /*!
   * \brief Construction with an unknown shape expression.
   * \param dtype The data type of tensor's elements.
   * \param ndim The number of dimensions
   * \param vdevice The virtual device.
   * \param span The span of the AST.
   */
  TVM_DLL TensorType(tvm::PrimType dtype, int ndim, ffi::Optional<VDevice> vdevice = std::nullopt,
                     Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorType, Type, TensorTypeNode);
};

/*!
 * \brief custom-defined Type derivation function.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \return The derived type of the call.
 */
using TypeDeriveFunc = TypedEnvFunc<Type(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Function type information.
 *
 * This data structure contains enough information for us to do best-effort
 * type deduction.
 */
class FuncTypeNode : public TypeNode {
 public:
  /*!
   * \brief The parameter type of the function.
   * \note When params is std::nullopt means the function can take arbitrary number of arguments.
   *       We define such functions as Opaque function.
   */
  ffi::Optional<ffi::Array<Type>> params;
  /*!
   * \brief The type of the function's return value.
   */
  Type ret;
  /*!
   * \brief Derivation function of opaque functions that may take any number of parameters.
   * \note When derive_func is not empty, then params should be std::nullopt,
   *       ret should be ObjectType()
   */
  ffi::Optional<TypeDeriveFunc> derive_func;
  /*!
   * \brief Whether the function is pure.
   * \note This parameter should be set to true only if the function is pure on all inputs.
   *   If the function _may_ have visible side effects, set it to false.
   */
  bool purity;

  /*!
   * \return Whether the func type is opaque.
   * \note We define a function as opaque we have no constraints on params.
   */
  bool IsOpaque() const { return !params.defined(); }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncTypeNode>()
        .def_ro("params", &FuncTypeNode::params, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("ret", &FuncTypeNode::ret)
        .def_ro("derive_func", &FuncTypeNode::derive_func)
        .def_ro("purity", &FuncTypeNode::purity);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.FuncType", FuncTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to FuncTypeNode.
 * \sa FuncTypeNode
 */
class FuncType : public Type {
 public:
  explicit FuncType(ffi::ObjectPtr<FuncTypeNode> data) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  /*!
   * \brief Constructor from parameter type and return value type.
   * \param params The type of function parameters.
   * \param ret The return value type.
   * \param purity The purity of the function (true by default).
   * \param span The span of the AST.
   *
   * \note If the ret contains variables(tirx::Var and relax::Var), they must be deducible from
   * params. If you are unsure, you can always erase ret to static.
   */
  TVM_DLL FuncType(ffi::Array<Type> params, Type ret, bool purity = true, Span span = Span());

  /*!
   * \brief Constructing an opaque function type using derive_func.
   *
   * \param derive_func Derivation function.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param span The span of the AST.
   *
   * \return The FuncType for opaque packedfunc.
   * \note Defaults to an derive func that always return ObjectType if not specified.
   */
  TVM_DLL static FuncType OpaqueFunc(TypeDeriveFunc derive_func, bool purity = false,
                                     Span span = Span());

  /*!
   * \brief Construct an opaque function using from return type.
   *
   * \param ret The type of the return value.
   * \param purity The purity of the function
   *   (false by default: most external functions are not pure).
   * \param span The span of the AST.
   *
   * \return The FuncType for opaque packedfunc.
   * \note Defaults to an derive func that always return ObjectType if not specified.
   */
  TVM_DLL static FuncType OpaqueFunc(Type ret = ObjectType(), bool purity = false,
                                     Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeNode);
};

/*!
 * \brief Match and check if expr has Relax type T and return it.
 *
 * \param expr The input expression.
 * \return The result of match.
 * \tparam T the underlying Relax type
 */
template <typename T>
inline ffi::Optional<T> MatchType(const Expr& expr) {
  if (!expr.defined()) {
    return std::nullopt;
  }
  using TNode = typename T::ContainerType;
  if (const TNode* ptr = expr->ty.as<TNode>()) {
    return ffi::GetRef<T>(ptr);
  } else {
    return std::nullopt;
  }
}

/*!
 * \brief Get the type of a given expr and try to cast it as const T*.
 *
 * \param expr The input expression.
 * \return The pointer. Returns nullptr if the type does not match.
 * \tparam T the underlying Relax type node
 */
template <typename T>
inline const T* GetTypeAs(const Expr& expr) {
  TVM_FFI_ICHECK(expr->ty.defined())
      << "The type is not populated, check if you have normalized the expr";
  return expr->ty.as<T>();
}

/*!
 * \brief Get the underlying Relax type of expr.
 *
 * \param expr The input expression.
 * \return underlying Relax type.
 */
inline Type GetType(const Expr& expr) {
  TVM_FFI_ICHECK(expr->ty.defined())
      << "The type is not populated, check if you have normalized the expr";
  return expr->ty;
}

/*!
 * \brief Whether the expr has void type.
 *
 * \param expr The input expression.
 * \return Whether the expr has void type.
 */
inline bool HasVoidType(const Expr& expr) {
  auto* ptr = expr->ty.as<TupleTypeNode>();
  return ptr != nullptr && ptr->fields.size() == 0;
}

/*!
 * \brief Update the type of an Expr.
 * \param expr The Expr whose type to be updated.
 * \param ty The type assigned.
 * \note We ensure idempotence, that is we can only update the type of an Expr only
 *  if the original one is nullptr.
 */
TVM_DLL void UpdateType(Expr expr, Type ty);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TYPE_H_
