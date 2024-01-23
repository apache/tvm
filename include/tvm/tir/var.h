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
 * \file tvm/tir/var.h
 * \brief Variables in the TIR.
 */
#ifndef TVM_TIR_VAR_H_
#define TVM_TIR_VAR_H_

#include <tvm/ir/expr.h>
#include <tvm/node/node.h>
#include <tvm/runtime/data_type.h>

#include <string>

namespace tvm {
namespace tir {

/*!
 * \brief A variable node in the IR.
 *
 * A variable is uniquely identified by its address.
 *
 * Each variable is only bound once in the following nodes:
 * - Allocate
 * - For
 * - Let
 * - LetStmt
 */
class VarNode : public PrimExprNode {
 public:
  /*!
   * \brief The hint to the variable name.
   * \note Each variable is uniquely identified by its address.
   */
  String name_hint;
  /*!
   * \brief type annotation of the variable.
   *
   * It is an optional field that provides a refined type of the variable than dtype.
   *
   * \sa tvm/ir/type.h for discussion of relations between runtime::DataType and Type.
   */
  Type type_annotation;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("name", &name_hint);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
    if (!equal(dtype, other->dtype)) return false;
    if (!equal(type_annotation, other->type_annotation)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(type_annotation);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tir.Var";
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_DECLARE_BASE_OBJECT_INFO(VarNode, PrimExprNode);
};

/*! \brief a named variable in TIR */
class Var : public PrimExpr {
 public:
  explicit Var(ObjectPtr<Object> n) : PrimExpr(n) {}
  /*!
   * \brief Constructor
   * \param name_hint variable name
   * \param dtype data type
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit Var(String name_hint = "v", DataType dtype = DataType::Int(32),
                       Span span = Span());
  /*!
   * \brief Constructor which provides a more detailed type annotation.
   * \param name_hint variable name.
   * \param type_annotation The type annotation.
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit Var(String name_hint, Type type_annotation, Span span = Span());
  /*!
   * \brief Make a new copy of var with same type, but a different nam
   * \param name The new name to be used.
   * \return the new Var copy
   */
  TVM_DLL Var copy_with_name(const String& name) const;
  /*!
   * \brief Make a new copy of var with same type, append suffix
   * \param suffix The suffix to be appended.
   * \return the new Var copy
   */
  TVM_DLL Var copy_with_suffix(const String& suffix) const;
  /*!
   * \brief Make a new copy of the variable with specified dtype
   * \param dtype The specified dtype
   * \return The new variable
   */
  TVM_DLL Var copy_with_dtype(DataType dtype) const;

  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const VarNode* operator->() const { return get(); }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const VarNode* get() const { return static_cast<const VarNode*>(data_.get()); }
  /*! \brief type indicate the container type */
  using ContainerType = VarNode;
};

/*!
 * \brief A variable node represent a tensor index size,
 * whose value must be non-negative.
 */
class SizeVarNode : public VarNode {
 public:
  static constexpr const char* _type_key = "tir.SizeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(SizeVarNode, VarNode);
};

/*! \brief a named variable represents a tensor index size */
class SizeVar : public Var {
 public:
  explicit SizeVar(ObjectPtr<Object> n) : Var(n) {}
  /*!
   * \brief constructor
   * \param name_hint variable name
   * \param t data type
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit SizeVar(String name_hint = "s", DataType t = DataType::Int(32),
                           Span span = Span());
  /*!
   * \brief Constructor which provides a more detailed type annotation.
   * \param name_hint variable name.
   * \param type_annotation The type annotation.
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit SizeVar(String name_hint, Type type_annotation, Span span = Span());
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const SizeVarNode* operator->() const { return get(); }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const SizeVarNode* get() const { return static_cast<const SizeVarNode*>(data_.get()); }
  /*! \brief type indicate the container type */
  using ContainerType = SizeVarNode;
};

using Region = Array<Range>;

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
class IterVarNode : public Object {
 public:
  /*!
   * \brief the domain of iteration, if known, can be None
   *  For the intermediate schedule node, before schedule.
   */
  Range dom;
  /*! \brief The looping variable */
  Var var;
  /*! \brief The type of the IterVar */
  IterVarType iter_type;
  /*!
   * \brief additional tag on the iteration variable,
   *  set this if this is bound already to a known thread tag.
   */
  String thread_tag;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dom", &dom);
    v->Visit("var", &var);
    v->Visit("iter_type", &iter_type);
    v->Visit("thread_tag", &thread_tag);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const IterVarNode* other, SEqualReducer equal) const {
    return equal(dom, other->dom) && equal.DefEqual(var, other->var) &&
           equal(iter_type, other->iter_type) && equal(thread_tag, other->thread_tag);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dom);
    hash_reduce.DefHash(var);
    hash_reduce(iter_type);
    hash_reduce(thread_tag);
  }

  static constexpr const char* _type_key = "tir.IterVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IterVarNode, Object);
};

/*!
 * \brief Iteration Variable,
 *  represents an iteration over an integer interval.
 *
 *  The dtype of the extent of the `dom` of the IterVar must match the dtype of the internal Var.
 */
class IterVar : public ObjectRef {
 public:
  TVM_DLL IterVar(Range dom, Var var, IterVarType iter_type, String thread_tag = "",
                  Span span = Span());
  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator PrimExpr() const;

  TVM_DEFINE_OBJECT_REF_METHODS(IterVar, ObjectRef, IterVarNode);
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
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_VAR_H_
