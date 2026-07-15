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
 * \file tvm/tirx/var.h
 * \brief Variables in the TIR.
 */
#ifndef TVM_TIR_VAR_H_
#define TVM_TIR_VAR_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

#include <utility>

namespace tvm {
namespace tirx {

using VarNode = tvm::VarNode;
using Var = tvm::Var;

/*!
 * \brief Checked scalar view over a VarNode.
 *
 * PrimVar is a zero-state reference view over the same VarNode as Var.  It additionally
 * guarantees that the inherited ExprNode::ty is PrimType.
 */
class PrimVar : public PrimExpr {
 public:
  /*! \brief Construct a scalar variable directly from a primitive type. */
  explicit PrimVar(ffi::String name_hint, PrimType dtype = PrimType::Int(32), Span span = Span())
      : PrimExpr(
            Var(std::move(name_hint), std::move(dtype), std::move(span)).as_or_throw<PrimExpr>()) {}

  /*! \brief Construct a scalar variable directly from a checked type annotation. */
  explicit PrimVar(ffi::String name_hint, Type type_annotation, Span span = Span())
      : PrimExpr(Var(std::move(name_hint), std::move(type_annotation), std::move(span))
                     .as_or_throw<PrimExpr>()) {}

  /*! \brief Safe widening to a general Var view over the same node. */
  operator Var() const { return this->as_or_throw<Var>(); }

  PrimVar CopyWithSuffix(const ffi::String& suffix) const {
    return this->as_or_throw<Var>().CopyWithSuffix(suffix).as_or_throw<PrimVar>();
  }
  PrimVar copy_with_dtype(PrimType dtype) const {
    return this->as_or_throw<Var>().copy_with_dtype(dtype).as_or_throw<PrimVar>();
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimVar, PrimExpr, VarNode);
  static constexpr bool _type_container_is_exact = false;
};

using Region = ffi::Array<Range>;

/*!
 * \brief Type of iteration variable.
 *  Each IterVar have a specific type.
 *
 *  The type of iter var can be overriden via
 *  stage.iter_var_attrs given they are compatible.
 */
enum IterVarType : int {
  /*!
   * \brief Data parallel iteration.
   *  This normally corresponds to axis of Tensor.
   *  Allow all IterVar manipulations.
   *
   * \note This does not mean the loop
   *  have to be executed in parallel fashion.
   */
  kDataPar = 0,
  /*!
   * \brief The IterVar itself is a thread-index
   *  of a fixed thread launching group.
   *  Note that this is already assumed to be parallelized.
   *
   *  Disallow: split/fuse/vectorize/parallel
   */
  kThreadIndex = 1,
  /*!
   * \brief Communicative reduction.
   *  Cannot be directly parallelized.
   *
   *  Disallow: parallel/vectorize
   */
  kCommReduce = 2,
  /*!
   * \brief Serial loops with loop carry dependency,
   *  the iteration must execute in order.
   *  Cannot be re-ordered.
   *
   *  Disallow: reorder/parallel/vectorize
   */
  kOrdered = 3,
  /*!
   * \brief IterVar is opaque,
   *
   *  May not corresponds to any generated loop
   *  Disallow all IterVar manipulations and compute_at
   *
   * \note This is usually used to implement composite op
   *  or external op, where the
   */
  kOpaque = 4,
  // The following are possible additional
  // types that are provided during schedule
  /*!
   * \brief The execution is unrolled.
   */
  kUnrolled = 5,
  /*!
   * \brief The loop is vectorized.
   */
  kVectorized = 6,
  /*!
   * \brief The loop is parallelized.
   */
  kParallelized = 7,
  /*!
   * \brief Marks boundary of tensorization intrinsic.
   */
  kTensorized = 8
};

/*!
 * \brief An iteration variable representing an iteration
 *  over a one dimensional interval.
 *
 *  The dtype of the extent of the `dom` of the IterVar must match the dtype of the internal Var.
 */
class IterVarNode : public PrimExprConvertibleNode {
 public:
  /*!
   * \brief the domain of iteration, if known, can be None
   *  For the intermediate schedule node, before schedule.
   */
  Range dom;
  /*! \brief The looping variable */
  PrimVar var;
  /*! \brief The type of the IterVar */
  IterVarType iter_type;
  /*!
   * \brief additional tag on the iteration variable,
   *  set this if this is bound already to a known thread tag.
   */
  ffi::String thread_tag;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  PrimExpr ToPrimExpr() const final { return var; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IterVarNode>()
        .def_ro("dom", &IterVarNode::dom)
        .def_ro("var", &IterVarNode::var, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("iter_type", &IterVarNode::iter_type)
        .def_ro("thread_tag", &IterVarNode::thread_tag);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.IterVar", IterVarNode, PrimExprConvertibleNode);
};

/*!
 * \brief Iteration Variable,
 *  represents an iteration over an integer interval.
 *
 *  The dtype of the extent of the `dom` of the IterVar must match the dtype of the internal Var.
 */
class IterVar : public PrimExprConvertible {
 public:
  TVM_DLL IterVar(Range dom, PrimVar var, IterVarType iter_type, ffi::String thread_tag = "",
                  Span span = Span());
  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator PrimExpr() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IterVar, PrimExprConvertible, IterVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterVarNode);
};

// inline implementations
inline IterVar::operator PrimExpr() const { return (*this)->var; }

inline const char* IterVarType2String(IterVarType t) {
  switch (t) {
    case kDataPar:
      return "DataPar";
    case kThreadIndex:
      return "ThreadIndex";
    case kCommReduce:
      return "CommReduce";
    case kOrdered:
      return "Ordered";
    case kOpaque:
      return "Opaque";
    case kUnrolled:
      return "Unrolled";
    case kVectorized:
      return "Vectorized";
    case kParallelized:
      return "Parallelized";
    case kTensorized:
      return "Tensorized";
  }
  return "Unknown";
}
}  // namespace tirx

}  // namespace tvm

namespace tvm::ffi {

template <>
inline constexpr bool use_default_type_traits_v<tirx::PrimVar> = false;

template <>
struct TypeTraits<tirx::PrimVar> : public ObjectRefTypeTraitsBase<tirx::PrimVar> {
  using Base = ObjectRefTypeTraitsBase<tirx::PrimVar>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return tirx::PrimVar::_type_is_nullable;
    }
    if (src->type_index != tirx::VarNode::RuntimeTypeIndex()) {
      return false;
    }
    const auto* var = static_cast<const tirx::VarNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<PrimType>(var->ExprNode::ty);
  }

  TVM_FFI_INLINE static std::optional<tirx::PrimVar> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<tirx::PrimVar>(nullptr);
      }
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<tirx::PrimVar>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<tirx::VarNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

}  // namespace tvm::ffi

#endif  // TVM_TIR_VAR_H_
