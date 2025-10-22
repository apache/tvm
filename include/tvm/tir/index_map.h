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

#include <tvm/ffi/container/array.h>
#include <tvm/ir/expr.h>
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
  ffi::Array<Var> initial_indices;

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
  ffi::Array<PrimExpr> final_indices;

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
  ffi::Optional<ObjectRef> inverse_index_map;

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
  ffi::Array<PrimExpr> MapIndices(const ffi::Array<PrimExpr>& indices,
                                  arith::Analyzer* analyzer) const;

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
  ffi::Array<Range> MapRanges(const ffi::Array<Range>& ranges, arith::Analyzer* analyzer) const;

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
  ffi::Array<PrimExpr> MapShape(const ffi::Array<PrimExpr>& shape, arith::Analyzer* analyzer) const;

  /* \brief Map an Tensor according to this index map
   *
   * \param arr_src The Tensor whose layout is transformed by this index map.
   *
   * \returns The transformed Tensor.
   */
  runtime::Tensor MapTensor(runtime::Tensor arr_src) const;

  /*!
   * \brief Convert to string representation in Python.
   * \param f_name_map Optional function to specify the stringified name of the variables.
   * \return The stringified lambda expression in Python.
   */
  ffi::String ToPythonString(
      const std::function<ffi::Optional<ffi::String>(const Var& var)>& f_name_map = nullptr) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IndexMapNode>()
        .def_ro("initial_indices", &IndexMapNode::initial_indices,
                refl::AttachFieldFlag::SEqHashDef())
        .def_ro("final_indices", &IndexMapNode::final_indices)
        .def_ro("inverse_index_map", &IndexMapNode::inverse_index_map,
                refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.IndexMap", IndexMapNode, Object);
};

class IndexMap : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param initial_indices Variables representing the indices prior to remapping
   * \param final_indices Expressions defining the indices after remapping.
   * \param inverse_index_map The optional pre-defined inverse index map
   */
  IndexMap(ffi::Array<Var> initial_indices, ffi::Array<PrimExpr> final_indices,
           ffi::Optional<IndexMap> inverse_index_map = std::nullopt);

  /*!
   * \brief Create an index map from a packed function
   * \param ndim The number of dimensions
   * \param func The function to be applied
   * \param inverse_index_map The optional pre-defined inverse index map
   * \return The created index map
   */
  static IndexMap FromFunc(int ndim, ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<Var>)> func,
                           ffi::Optional<IndexMap> inverse_index_map = std::nullopt);

  /*! \brief Generate the inverse mapping.
   *
   * The range of the input indices is required in order to ensure
   * that the transformation is bijective over the input domain.
   *
   * If the user has supplied an `inverse_index_map`, that map is
   * assumed to be correct and bijective, and is returned.
   */
  IndexMap Inverse(ffi::Array<Range> initial_ranges, arith::Analyzer* analyzer) const;

  /*! \brief Rename the variables in the index map and ensure the names are unique.
   *
   * Construct a new index map with the same transformation, but with name_hint of variables to be
   * guaranteed unique. The optional f_name_map can be provided to rename the variables.
   *
   * \param f_name_map The optional name map to rename the variables.
   * \return The renamed index map.
   */
  IndexMap RenameVariables(
      const std::function<ffi::Optional<ffi::String>(const Var& var)>& f_name_map = nullptr) const;

  /*! \brief Generate the inverse mapping.
   *
   * Determine the inverse, where the output range may contain
   * addresses that do not correspond to an address in the input
   * range.
   *
   * \return The inverted index map, along with the predicate for
   * which the inverse maps to a valid range.
   */
  std::pair<IndexMap, PrimExpr> NonSurjectiveInverse(ffi::Array<Range> initial_ranges,
                                                     arith::Analyzer* analyzer) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IndexMap, ObjectRef, IndexMapNode);
};

/*! \brief Substitute variables in an index map.
 *
 * \param index_map The index_map
 * \param f_subst The substitution function
 */
IndexMap Substitute(const IndexMap& index_map,
                    std::function<ffi::Optional<PrimExpr>(const Var& var)> f_subst);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_INDEX_MAP_H_
