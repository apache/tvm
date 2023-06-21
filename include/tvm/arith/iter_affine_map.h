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
 * \file tvm/arith/iter_affine_map.h
 * \brief Iterator quasi-affine mapping patterns.
 *
 *  This file defines a collection of mapping patterns
 *  maps a collection of independent iterators to another
 *  collection of independent iterators.
 *
 *  There are two main kinds of mapping patterns:
 *
 *  - Fuse: fuse a collection of iterators into a single one
 *
 *    domain(x0) = [0, 4), domain(x1) = [0, 3), domain(x2) = [0, 2)
 *    fuse(x0, x1, x2): y = x2 * 12 + x1 * 4 + x0
 *    domain(y) = [0, 24)
 *
 *  - Split: split an iterator into multiple ones
 *
 *    domain(x) = [0, 24)
 *    split(x, 3, 12): [y0, y1, y2] = [x % 3, (x % 12) / 3, x / 12]
 *    domain(y0) = [0, 3), domain(y1) = [0, 4), domain(y2) = [0, 2)
 *
 *  We use the name "(quasi)affine" to be consistent with
 *  the terminology used in the polyhedral compilation.
 *  Notably, fuse is an affine transformation,
 *  while split corresponds to additional floordiv/mod operations
 *  that can appear in quasi-affine transformations.
 */
#ifndef TVM_ARITH_ITER_AFFINE_MAP_H_
#define TVM_ARITH_ITER_AFFINE_MAP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/diagnostic.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace arith {

/*!
 * \brief Base class of all iter map expressions.
 *
 *  An IterMapExpr is a special expression to store
 *  the result of IterMapDetection.
 *  It should not appear in a legal TIR PrimFunc.
 */
class IterMapExprNode : public PrimExprNode {
 public:
  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "arith.IterMapExpr";
  static constexpr const uint32_t _type_child_slots = 3;
  TVM_DECLARE_BASE_OBJECT_INFO(IterMapExprNode, PrimExprNode);
};

/*!
 * \brief Managed reference to IterMapExprNode.
 * \sa IterMapExprNode
 */
class IterMapExpr : public PrimExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(IterMapExpr, PrimExpr, IterMapExprNode);
};

/*!
 * \brief Mark the source as an iterator in [0, extent).
 *
 *  IterMark is used to mark source expression as a valid
 *  iterator to make future analysis easy.
 */
class IterMarkNode : public Object {
 public:
  /*!
   * \brief The source expression, can either be
   *  a IterSumExpr or a Var.
   */
  PrimExpr source;
  /*!
   * \brief The extent of the iteration.
   */
  PrimExpr extent;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("extent", &extent);
  }

  bool SEqualReduce(const IterMarkNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(source, other->source) && equal(extent, other->extent);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(source);
    hash_reduce(extent);
  }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "arith.IterMark";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterMarkNode, Object);
};

/*!
 * \brief Managed reference to IterMarkExprNode.
 * \sa IterMarkExprNode
 */
class IterMark : public ObjectRef {
 public:
  /*!
   * \brief constructor.
   * \param source The source expression.
   * \param extent The extent of the iterator.
   */
  TVM_DLL IterMark(PrimExpr source, PrimExpr extent);

  TVM_DEFINE_OBJECT_REF_METHODS(IterMark, ObjectRef, IterMarkNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterMarkNode);
};

/*!
 * \brief Split of an iterator.
 *
 *  result = floormod(floordiv(source, lower_factor), extent) * scale
 */
class IterSplitExprNode : public IterMapExprNode {
 public:
  /*! \brief The source marked iterator. */
  IterMark source;
  /*! \brief The lower factor to split the source. */
  PrimExpr lower_factor;
  /*! \brief The extent of the split. */
  PrimExpr extent;
  /*! \brief Additional scale. */
  PrimExpr scale;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("lower_factor", &lower_factor);
    v->Visit("extent", &extent);
    v->Visit("scale", &scale);
  }

  bool SEqualReduce(const IterSplitExprNode* other, SEqualReducer equal) const {
    return equal(source, other->source) && equal(lower_factor, other->lower_factor) &&
           equal(extent, other->extent) && equal(scale, other->scale);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(source);
    hash_reduce(lower_factor);
    hash_reduce(extent);
    hash_reduce(scale);
  }

  static constexpr const char* _type_key = "arith.IterSplitExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterSplitExprNode, IterMapExprNode);
};

/*!
 * \brief Managed reference to IterSplitExprNode.
 * \sa IterSplitExprNode
 */
class IterSplitExpr : public IterMapExpr {
 public:
  /*!
   * \brief constructor from just source.
   * \param source The source expression.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source);
  /*!
   * \brief constructor from just source.
   * \param source The source expression.
   * \param scale The additional scaling factor.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source, PrimExpr scale);
  /*!
   * \brief constructor
   * \param source The source expression.
   * \param lower_factor The lower factor to split the source.
   * \param extent The extent of the split.
   * \param scale The additional scaling factor.
   */
  TVM_DLL explicit IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                                 PrimExpr scale);

  TVM_DEFINE_OBJECT_REF_METHODS(IterSplitExpr, IterMapExpr, IterSplitExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSplitExprNode);
};

/*!
 * \brief Fuse multiple iterators by summing them with scaling.
 *
 *  result = sum(args) + base
 */
class IterSumExprNode : public IterMapExprNode {
 public:
  /*! \brief The args to the sum. */
  Array<IterSplitExpr> args;
  /*! \brief The base offset. */
  PrimExpr base;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("args", &args);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const IterSumExprNode* other, SEqualReducer equal) const {
    return equal(args, other->args) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(args);
    hash_reduce(base);
  }

  static constexpr const char* _type_key = "arith.IterSumExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterSumExprNode, IterMapExprNode);
};

/*!
 * \brief Managed reference to IterSumExprNode.
 * \sa IterSumExprNode
 */
class IterSumExpr : public IterMapExpr {
 public:
  /*!
   * \brief constructor.
   * \param args The args to the sum.
   * \param base The base offset.
   */
  TVM_DLL IterSumExpr(Array<IterSplitExpr> args, PrimExpr base);

  TVM_DEFINE_OBJECT_REF_METHODS(IterSumExpr, IterMapExpr, IterSumExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterSumExprNode);
};

/*! \brief Mapping level for iterators. */
enum IterMapLevel {
  // Require the mapping to be bijective.
  Bijective = 0,
  // Require the mapping to be surjective.
  Surjective = 1,
  // No mapping safety check.
  NoCheck = 3
};

/*!
 * \brief Result of DetectIterMap.
 */
class IterMapResultNode : public Object {
 public:
  // The detected pattern if a match exists.
  Array<IterSumExpr> indices;

  // Any errors that occurred while converting the input indices.  If
  // the array is empty, the conversion was successful.
  Array<String> errors;

  /*! \brief Boolean expression indicating if a specific value w
   *
   * `padding_predicate` evaluates to true for a set of indices that
   * are outside the bounds of the provided index iterators, but
   * inside the bounds of the returned index iterators.  This
   * expression is in terms of the variables provided in
   * `input_iters`.
   */
  PrimExpr padding_predicate;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("errors", &errors);
    v->Visit("indices", &indices);
    v->Visit("padding_predicate", &padding_predicate);
  }

  static constexpr const char* _type_key = "arith.IterMapResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterMapResultNode, Object);
};

/*!
 * \brief Managed reference to IterMapResultNode.
 * \sa IterMapResultNode
 */
class IterMapResult : public ObjectRef {
 public:
  // constructor
  IterMapResult() { data_ = make_object<IterMapResultNode>(); }

  /*! \return mutable pointers to the node. */
  IterMapResultNode* operator->() const { return static_cast<IterMapResultNode*>(get_mutable()); }
};

/*!
 * \brief Detect if indices can be written as
 *  [y_0 + c_0, y_1 + c_1, ..., y_n + c_n]
 *
 *  Here y = some-quasi-affine-iter-map(input_iters)
 *  and c are symbolic constants.
 *
 *  We also requires that y_i and y_j to be independent for i != j.
 *
 *  For returned value rv, the following is always true:
 *  - rv[i]->args.size() <=1: only one iterator per element.
 *
 * \param indices The indices to detect pattern for.
 * \param input_iters Map from variable to iterator's range.
 * \param predicate The predicate constraints on the input iterators
 * \param check_level The iter mapping checking level.
 * \param analyzer Analyzer used to get context information.
 * \param simplify_trivial_iterators If true, iterators with extent of
 *           1 will be replaced with a constant value.
 *
 * \return The detected iteration result.
 * The return object's .indices is empty on failure.
 */
IterMapResult DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                            const PrimExpr& predicate, IterMapLevel check_level,
                            arith::Analyzer* analyzer, bool simplify_trivial_iterators = true);

/*!
 * \brief Use IterVarMap detector to rewrite and simplify the indices
 *
 * \param indices The indices to detect pattern for.
 * \param input_iters Map from variable to iterator's range.
 * \param input_pred The predicate constraints on the input iterators
 * \param check_level The iter mapping checking level.
 * \param analyzer Analyzer used to get context information.
 * \param simplify_trivial_iterators If true, iterators with unit extents are simplified
 * \return The indices after rewrite
 */
Array<PrimExpr> IterMapSimplify(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                const PrimExpr& input_pred, IterMapLevel check_level,
                                arith::Analyzer* analyzer, bool simplify_trivial_iterators = true);

/*!
 * \brief Apply the inverse of the affine transformation to the outputs.
 *
 * Similar to the back-propagation, starting from the outputs, it visits the DAG of the expressions
 * in reverse topology order and applies the inverse of the affine transformation until it reaches
 * the input. The affine iter map is required to be bijective.
 *
 * For example, iter_map = [l0 // 16, l0 % 16], outputs = [output_0, output_1],
 * the affine transformation specified by `iter_map` will be applied to `outputs` and the result
 * will be {l0: ((output_0*16) + output_1)}.
 *
 * The range of `outputs` should be the same as the output range of the affine transmation.
 *
 * \sa DetectIterMap
 *
 * \param iter_map The bijective affine iter map.
 * \param outputs The outputs of the affine transformation.
 *
 * \return The map from the input to the transformed result.
 */
Map<Var, PrimExpr> InverseAffineIterMap(const Array<IterSumExpr>& iter_map,
                                        const Array<PrimExpr> outputs);

/*!
 * \brief Detect if bindings can be written as
 * [a_0*e_0 + b_0 + c_0, a_1*e_1 + b_1, ..., a_n*e_n + b_n]
 *
 * where a = some-quasi-affine-iter-map(input_iters set_minus sub_iters)
 *       b = some-quasi-affine-iter-map(sub_iters)
 *       c is constant symbols
 *       e is the extent of b
 *
 * For example, z*12 + y*3 + x + c = (z*4+y)*3 + x, if sub_iters={x}
 *
 * \param bindings The input bindings
 * \param input_iters Map from variable to iterator's range.
 * \param sub_iters Iterators of subspace.
 * \param predicate The predicate constraints on the input iterators
 * \param check_level The iter mapping checking level.
 * \param analyzer Analyzer used to get context information.
 * \param simplify_trivial_iterators If true, iterators with extent of
 *           1 will be replaced with a constant value.
 *
 * \return The result list has length len(bindings) + 1
        [0, len(bindings)): The iter map matching result. The inner list is of length 2.
                            The first expr is the basis of the quotient space.
                            The second expr is the basis of the subspace.
        len(bindings): the predicate of outer space and inner space
        Empty array if no match can be found.
 */
Array<Array<IterMark>> SubspaceDivide(const Array<PrimExpr>& bindings,
                                      const Map<Var, Range>& input_iters,
                                      const Array<Var>& sub_iters, const PrimExpr& predicate,
                                      IterMapLevel check_level, arith::Analyzer* analyzer,
                                      bool simplify_trivial_iterators = true);

/*!
 * \brief Given an expression that may contain IterMapExpr, transform it to normal PrimExpr.
 * \param expr The input expression, which may contain IterMapExpr.
 * \return The corresponding normal PrimExpr.
 */
PrimExpr NormalizeIterMapToExpr(const PrimExpr& expr);

/*!
 * \brief Rewrite index as IterSumExpr
 *
 * ((i0 // b0) % a0) * s0 + ((i0 // b1) % a1) * s1 ... + base
 *
 * The iterators are ordered such that s0 > s1 ...
 * if we can prove the relation.
 *
 * Note that base may contain expressions that cannot be detected
 * as the right pattern.
 *
 * \param index The input index
 * \param input_iters The input iterators.
 * \param analyzer The input analyzer.
 * \note This function is useful to detect iterator stride patterns.
 */
IterSumExpr NormalizeToIterSum(PrimExpr index, const Map<Var, Range>& input_iters,
                               arith::Analyzer* analyzer);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_ITER_AFFINE_MAP_H_
