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
 * \file tvm/tir/index_map.h
 * \brief Defines a remapping of buffer indices
 *
 * For use with tvm::tir::Buffer.
 */
#ifndef TVM_TIR_INDEX_MAP_H_
#define TVM_TIR_INDEX_MAP_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/var.h>

#include <utility>

namespace tvm {
namespace arith {
class Analyzer;
}
}  // namespace tvm

namespace tvm {
namespace tir {

/*!
 * \brief Defines a mapping between two representations of indices
 * into a buffer.
 *
 * This is primarily used for layout transformations of Buffer
 * objects.
 */
class IndexMapNode : public Object {
 public:
  /*! \brief Variables representing the indices prior to remapping.
   *
   * If initial_indices is empty, then final_indices should also be
   * empty, and no mapping is applied.
   */
  Array<Var> initial_indices;

  /*!
   * \brief Expressions defining the indices after remapping.
   *
   * These expressions should only be in terms of the initial_indices,
   * and must be expressible as an IterSumExpr.  The mapping from
   * initial_indices to final_indices must be injective.
   *
   * If final_indices is empty, then initial_indices should also be
   * empty, and the map is an identity function.
   */
  Array<PrimExpr> final_indices;

  /*!
   * \brief The inverse index map.
   *
   * When this is defined, IndexMap::Inverse will return the
   * pre-defined inverse index map.  Otherwise, the inverse index map
   * will be computed on the fly.  It is the user's responsibility to
   * ensure the correctness of the pre-defined inverse index map.
   *
   * \note ObjectRef is used here instead of IndexMap to avoid circular reference.
   */
  Optional<ObjectRef> inverse_index_map;

  /*!
   * \brief Default constructor
   *
   * Defines the mapping as an identity function, with initial_indices
   * equal to the final indices.
   */
  IndexMapNode() {}

  /*!
   * \brief Map indices to the output space
   *
   * \param indices The indices in the input space.  Should contain
   * one value for each variable in `initial_indices`.
   *
   * \param analyzer An optional analyzer to be used to simplify the
   * resulting expressions.  If null, will use a fresh analyzer.
   *
   * \returns The indices in the output space.  Contains one value for
   * each expression in `final_indices`.
   */
  Array<PrimExpr> MapIndices(const Array<PrimExpr>& indices,
                             arith::Analyzer* analyzer = nullptr) const;

  /*! \brief Map a memory range to the output space
   *
   * If contiguous memory locations in the input space are not
   * necessarily contiguous in the output space (e.g. `lambda i:
   * [8*(i%8) + (i//8)]`), then this will return the smallest range
   * such that all valid indices are contained within the given range.
   *
   * \param ranges The ranges in the input space.  Should contain one
   * value for each variable in `initial_indices`.
   *
   * \param analyzer An optional analyzer to be used to simplify the
   * resulting expressions.  If null, will use a fresh analyzer.
   *
   * \returns The ranges in the output space.  Contains one value for
   * each expression in `final_indices`.
   */
  Array<Range> MapRanges(const Array<Range>& ranges, arith::Analyzer* analyzer = nullptr) const;

  /*! \brief Map a buffer shape to the output space
   *
   * \param shape The buffer shape in the input space.  Should contain
   * one value for each variable in `initial_indices`.
   *
   * \param analyzer An optional analyzer to be used to simplify the
   * resulting expressions.  If null, will use a fresh analyzer.
   *
   * \returns The buffer shape in the output space.  Contains one
   * value for each expression in `final_indices`.
   */
  Array<PrimExpr> MapShape(const Array<PrimExpr>& shape, arith::Analyzer* analyzer = nullptr) const;

  /* \brief Map an NDArray according to this index map
   *
   * \param arr_src The NDArray whose layout is transformed by this index map.
   *
   * \returns The transformed NDArray.
   */
  runtime::NDArray MapNDArray(runtime::NDArray arr_src) const;

  /*!
   * \brief Convert to string representation in Python.
   * \return The stringified lambda expression in Python.
   */
  String ToPythonString() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("initial_indices", &initial_indices);
    v->Visit("final_indices", &final_indices);
    v->Visit("inverse_index_map", &inverse_index_map);
  }

  bool SEqualReduce(const IndexMapNode* other, SEqualReducer equal) const {
    return equal.DefEqual(initial_indices, other->initial_indices) &&
           equal(final_indices, other->final_indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(initial_indices);
    hash_reduce(final_indices);
  }

  static constexpr const char* _type_key = "tir.IndexMap";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexMapNode, Object);
};

class IndexMap : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param initial_indices Variables representing the indices prior to remapping
   * \param final_indices Expressions defining the indices after remapping.
   * \param inverse_index_map The optional pre-defined inverse index map
   */
  IndexMap(Array<Var> initial_indices, Array<PrimExpr> final_indices,
           Optional<IndexMap> inverse_index_map = NullOpt);

  /*!
   * \brief Create an index map from a packed function
   * \param ndim The number of dimensions
   * \param func The function to be applied
   * \param inverse_index_map The optional pre-defined inverse index map
   * \return The created index map
   */
  static IndexMap FromFunc(int ndim, runtime::TypedPackedFunc<Array<PrimExpr>(Array<Var>)> func,
                           Optional<IndexMap> inverse_index_map = NullOpt);

  /*! \brief Generate the inverse mapping.
   *
   * The range of the input indices is required in order to ensure
   * that the transformation is bijective over the input domain.
   *
   * If the user has supplied an `inverse_index_map`, that map is
   * assumed to be correct and bijective, and is returned.
   */
  IndexMap Inverse(Array<Range> initial_ranges) const;

  /*! \brief Generate the inverse mapping.
   *
   * Determine the inverse, where the output range may contain
   * addresses that do not correspond to an address in the input
   * range.
   *
   * \return The inverted index map, along with the predicate for
   * which the inverse maps to a valid range.
   */
  std::pair<IndexMap, PrimExpr> NonSurjectiveInverse(Array<Range> initial_ranges) const;

  TVM_DEFINE_OBJECT_REF_METHODS(IndexMap, ObjectRef, IndexMapNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_INDEX_MAP_H_
