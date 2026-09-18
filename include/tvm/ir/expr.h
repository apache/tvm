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
 * \file tvm/ir/expr.h
 * \brief Base expr nodes in TVM.
 */
#ifndef TVM_IR_EXPR_H_
#define TVM_IR_EXPR_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/base_expr.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/op_attr_types.h>
#include <tvm/ir/source_map.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace tvm {

/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief The fields of the tuple. */
  ffi::Array<Expr> fields;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleNode>().def_ro("fields", &TupleNode::fields);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Tuple", TupleNode, ExprNode);
};

/*! \brief Managed reference to TupleNode. */
class Tuple : public Expr {
 public:
  /*!
   * \brief Construct a tuple from its fields.
   * \param fields The fields of the tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Tuple(ffi::Array<Expr> fields, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tuple, Expr, TupleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
};

/*! \brief Get the index-th field out of a tuple. */
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple expression. */
  Expr tuple;
  /*! \brief The field index. */
  int index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleGetItemNode>()
        .def_ro("tuple_value", &TupleGetItemNode::tuple)
        .def_ro("index", &TupleGetItemNode::index);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleGetItem", TupleGetItemNode, ExprNode);
};

/*! \brief Managed reference to TupleGetItemNode. */
class TupleGetItem : public Expr {
 public:
  /*!
   * \brief Construct a tuple field projection.
   * \param tuple The tuple to get an element from.
   * \param index The field index.
   * \param span The source span of the expression.
   */
  TVM_DLL TupleGetItem(Expr tuple, int index, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TupleGetItem, Expr, TupleGetItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
};

/*! \brief Load a value from an indexed expression source. */
class TensorLoadNode : public ExprNode {
 public:
  /*! \brief The indexed source expression. */
  Expr source;
  /*! \brief The indices at which the source is loaded. */
  ffi::Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorLoadNode>()
        .def_ro("source", &TensorLoadNode::source)
        .def_ro("indices", &TensorLoadNode::indices);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorLoad", TensorLoadNode, ExprNode);
};

/*! \brief Managed reference to TensorLoadNode. */
class TensorLoad : public PrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorLoad, PrimExpr, TensorLoadNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorLoadNode);
};

/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator+(PrimExpr a, PrimExpr b);

/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a, PrimExpr b);

/*!
 * \brief negation.
 *
 * \param a input.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a);

/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator*(PrimExpr a, PrimExpr b);

/*!
 * \brief division operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator/(PrimExpr a, PrimExpr b);

/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<<(PrimExpr a, PrimExpr b);

/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>>(PrimExpr a, PrimExpr b);

/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>(PrimExpr a, PrimExpr b);

/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>=(PrimExpr a, PrimExpr b);

/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<(PrimExpr a, PrimExpr b);

/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<=(PrimExpr a, PrimExpr b);

/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator==(PrimExpr a, PrimExpr b);

/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator!=(PrimExpr a, PrimExpr b);

/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator&&(PrimExpr a, PrimExpr b);

/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator||(PrimExpr a, PrimExpr b);

/*!
 * \brief not
 *
 * \param a left operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator!(PrimExpr a);

/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator&(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator|(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator^(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator~(PrimExpr a);

/*!
 * \brief A local variable in the IR.
 *
 * Variables are uniquely identified by their object identity.  The name is a
 * hint for printing and does not participate in structural equality or
 * hashing.
 */
class VarNode : public ExprNode {
 public:
  /*! \brief The variable name. */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarNode>().def_ro("name", &VarNode::name,
                                      refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr const uint32_t _type_child_slots = 1;
  // VarNode reserves its sole child slot for the final relax::DataflowVarNode subtype, so its
  // descendant type-index range cannot overflow.
  static constexpr bool _type_child_slots_can_overflow = false;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Var", VarNode, ExprNode);
};

/*! \brief Managed reference to VarNode. */
class Var : public Expr {
 public:
  TVM_DLL explicit Var(ffi::String name, ffi::Optional<Type> ty_annotation, Span span = Span());

  /*! \brief Return a fresh ordinary Var with the same type and a new name. */
  TVM_DLL Var CopyWithName(const ffi::String& name) const;

  /*! \brief Return a fresh ordinary Var with a suffix appended to its name. */
  TVM_DLL Var CopyWithSuffix(const ffi::String& suffix) const;

  /*! \brief Return a fresh ordinary Var with a new primitive type. */
  TVM_DLL Var CopyWithDType(PrimType dtype) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Var, Expr, VarNode);
};

/*!
 * \brief Checked scalar view over a VarNode.
 *
 * PrimVar is a zero-state reference view over the same VarNode as Var.  It additionally
 * guarantees that the inherited ExprNode::ty is PrimType.
 */
class PrimVar : public PrimExpr {
 public:
  /*! \brief Construct a scalar variable directly from a primitive type. */
  explicit PrimVar(ffi::String name, PrimType dtype = PrimType::Int(32), Span span = Span())
      : PrimExpr(Var(std::move(name), std::move(dtype), std::move(span)).as_or_throw<PrimExpr>()) {}

  /*! \brief Construct a scalar variable directly from a checked type annotation. */
  explicit PrimVar(ffi::String name, Type type_annotation, Span span = Span())
      : PrimExpr(Var(std::move(name), std::move(type_annotation), std::move(span))
                     .as_or_throw<PrimExpr>()) {}

  /*! \brief Safe widening to a general Var view over the same node. */
  operator Var() const { return this->as_or_throw<Var>(); }

  PrimVar CopyWithSuffix(const ffi::String& suffix) const {
    return this->as_or_throw<Var>().CopyWithSuffix(suffix).as_or_throw<PrimVar>();
  }
  PrimVar CopyWithDType(PrimType dtype) const {
    return this->as_or_throw<Var>().CopyWithDType(dtype).as_or_throw<PrimVar>();
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimVar, PrimExpr, VarNode);
  static constexpr bool _type_container_is_exact = false;
};

class GlobalVar;
/*!
 * \brief Global variable that lives in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarNode
 */
class GlobalVarNode : public ExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  ffi::String name_hint;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GlobalVarNode>().def_ro("name_hint", &GlobalVarNode::name_hint);
    // A GlobalVar identifies a module-level symbol.  Its type is derived from the
    // corresponding function definition and is not part of the symbol identity.
    refl::TypeAttrDef<GlobalVarNode>()
        .def("__s_equal__", &GlobalVarNode::SEqual)
        .def("__s_hash__", &GlobalVarNode::SHash);
  }

  bool SEqual(const GlobalVarNode* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    return equal(name_hint, other->name_hint, false, "name_hint");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    return hash(name_hint, init_hash, false);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarNode, ExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public Expr {
 public:
  TVM_DLL explicit GlobalVar(ffi::String name_hint, Span span = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GlobalVar, Expr, GlobalVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarNode);
};

/*!
 * \brief Call corresponds to callable invocation.
 */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator/function being invoked.
   *
   * It can be an Op, a GlobalVar, a local function value, or another callable
   * expression.
   */
  Expr op;

  /*! \brief The arguments of the call. */
  ffi::Array<Expr> args;

  /*! \brief The additional attributes. */
  Attrs attrs;

  /*! \brief The type information arguments passed to the callee. */
  ffi::Array<Type> ty_args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallNode>()
        .def_ro("op", &CallNode::op)
        .def_ro("args", &CallNode::args)
        .def_ro("attrs", &CallNode::attrs)
        .def_ro("ty_args", &CallNode::ty_args);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Call", CallNode, ExprNode);
};

/*!
 * \brief Managed reference to CallNode.
 */
class Call : public Expr {
 public:
  TVM_DLL Call(Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs = Attrs(),
               ffi::Array<Type> ty_args = ffi::Array<Type>(), Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Call, Expr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*! \brief range over one dimension */
class RangeNode : public ffi::Object {
 public:
  /*! \brief beginning of the node */
  PrimExpr min;
  /*! \brief the extend of range */
  PrimExpr extent;
  /*! \brief the location of this range in the source */
  mutable Span span;
  /*! \brief constructor */
  RangeNode() {}
  RangeNode(PrimExpr min, PrimExpr extent, Span span = Span())
      : min(min), extent(extent), span(span) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RangeNode>()
        .def_ro("min", &RangeNode::min)
        .def_ro("extent", &RangeNode::extent)
        .def_ro("span", &RangeNode::span, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Range", RangeNode, ffi::Object);
};

/*! \brief Range container  */
class Range : public ffi::ObjectRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   * \param span The location of the Range in the source.
   */
  TVM_DLL Range(PrimExpr begin, PrimExpr end, Span span = Span());
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   * \param span The location of the Range in the source.
   */
  TVM_DLL static Range FromMinExtent(PrimExpr min, PrimExpr extent, Span span = Span());
  // declare range.
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Range, ffi::ObjectRef, RangeNode);
};

/*! \brief A region of an indexed expression source. */
class TensorRegionNode : public ExprNode {
 public:
  /*! \brief The indexed source expression. */
  Expr source;
  /*! \brief The ranges selected from the source. */
  ffi::Array<Range> region;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // A region's source and symbolic source type form a definition pattern;
    // ranges then refer to those same identities.
    refl::ObjectDef<TensorRegionNode>()
        .def_ro("source", &TensorRegionNode::source, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("region", &TensorRegionNode::region);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegion", TensorRegionNode, ExprNode);
};

/*! \brief Managed reference to TensorRegionNode. */
class TensorRegion : public Expr {
 public:
  TVM_DLL TensorRegion(Expr source, ffi::Array<Range> region, Type ty, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorRegion, Expr, TensorRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorRegionNode);
};

/*!
 * \brief Analyze the runtime effects of an expression.
 *
 * Accumulates effects of evaluated expression children, excluding type and vector
 * lane-count metadata. Non-operator callees are conservatively opaque. Missing
 * operator effect attributes are errors unless an earlier update effect stops
 * traversal. Returns kPure, kReadState, or kUpdateState.
 *
 * Shared IR contains only minimal analyses of expression properties. Arithmetic
 * reasoning, constraint solving, and target- or dialect-specific analyses belong
 * in their respective modules.
 * \param expr The expression to inspect.
 * \return The strongest runtime effect of the expression.
 */
TVM_DLL CallEffectKind SideEffect(const Expr& expr);

namespace ffi {

template <>
inline constexpr bool use_default_type_traits_v<PrimVar> = false;

template <>
struct TypeTraits<PrimVar> : public ObjectRefTypeTraitsBase<PrimVar> {
  using Base = ObjectRefTypeTraitsBase<PrimVar>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return PrimVar::_type_is_nullable;
    }
    if (src->type_index != VarNode::RuntimeTypeIndex()) {
      return false;
    }
    const auto* var = static_cast<const VarNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<PrimType>(var->ExprNode::ty);
  }

  TVM_FFI_INLINE static std::optional<PrimVar> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<PrimVar>(nullptr);
      }
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<PrimVar>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<VarNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, TensorLoadNode> = true;

}  // namespace ffi

}  // namespace tvm

/* \brief Allow tvm.Var and tvm.GlobalVar as keys in STL tables
 *
 * For most IR expressions, it would be ambiguous whether the
 * expression should follow reference equality or structural equality.
 * This is not the case for variables, which do not contain nested
 * internal structure, and are frequently used as keys in lookup
 * tables.
 *
 * Providing `std::hash` and `std::equal_to` specializations for
 * `tvm::Var` and `tvm::GlobalVar` allows them to be used as keys in STL tables.  For
 * other IR expressions, the user must specify the type of equality
 * used (e.g. `std::unordered_set<T, StructuralHash, StructuralEqual>`
 * or `std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::Var> {
  std::size_t operator()(const tvm::Var& var) const { return tvm::ffi::ObjectPtrHash()(var); }
};

template <>
struct std::equal_to<tvm::Var> {
  bool operator()(const tvm::Var& var_a, const tvm::Var& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};

template <>
struct std::hash<tvm::GlobalVar> {
  std::size_t operator()(const tvm::GlobalVar& var) const { return tvm::ffi::ObjectPtrHash()(var); }
};

template <>
struct std::equal_to<tvm::GlobalVar> {
  bool operator()(const tvm::GlobalVar& var_a, const tvm::GlobalVar& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};
#endif  // TVM_IR_EXPR_H_
