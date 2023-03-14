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
 * \file tvm/relax/attrs/manipulate.h
 * \brief Attributes for tensor manipulation operators.
 */
#ifndef TVM_RELAX_ATTRS_MANIPULATE_H_
#define TVM_RELAX_ATTRS_MANIPULATE_H_

#include <tvm/relax/expr.h>
#include <tvm/tir/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in concat operators */
struct ConcatAttrs : public tvm::AttrsNode<ConcatAttrs> {
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(ConcatAttrs, "relax.attrs.ConcatAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axis at which the input arrays are concatenated."
        "Should lie in range `[-ndim, ndim)`.");
  }
};  // struct ConcatAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relax.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axes at which the input array are expanded. "
        "All values are required to lie in range `[-data.ndim - 1, data.ndim]`, "
        "with the convention of negative indexing.");
  }
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrs : public tvm::AttrsNode<LayoutTransformAttrs> {
  tir::IndexMap index_map;
  // pad_value is chosen to be of PrimValue type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimValue is evolved to represent symbolic expression in future.
  Optional<PrimValue> pad_value;

  TVM_DECLARE_ATTRS(LayoutTransformAttrs, "relax.attrs.LayoutTransformAttrs") {
    TVM_ATTR_FIELD(index_map).describe("The layout transformation to apply.");
    TVM_ATTR_FIELD(pad_value).describe(
        "The specific value to be used to pad if the layout transform would result in implicit "
        "padding. If not specified, the compiler is free to choose any value.");
  }
};  // struct LayoutTransformAttrs

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrs : public tvm::AttrsNode<PermuteDimsAttrs> {
  Optional<Array<Integer>> axes;

  TVM_DECLARE_ATTRS(PermuteDimsAttrs, "relax.attrs.PermuteDimsAttrs") {
    TVM_ATTR_FIELD(axes).describe("The target axes order, reverse order if not specified.");
  }
};  // struct PermuteDimsAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  ObjectRef indices_or_sections;
  int axis;

  TVM_DECLARE_ATTRS(SplitAttrs, "relax.attrs.SplitAttrs") {
    TVM_ATTR_FIELD(indices_or_sections)
        .describe("The input array of indices or the number of split sections.");
    TVM_ATTR_FIELD(axis).describe("The axis to be splitted");
  }
};  // struct SplitAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public tvm::AttrsNode<SqueezeAttrs> {
  Optional<Array<Integer>> axis;

  TVM_DECLARE_ATTRS(SqueezeAttrs, "relax.attrs.SqueezeAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axis to squeeze in the input tensor."
        "If `axis = None`, all axis of dimension 1 get squeezed;"
        "Else, the dimension in axes get squeezed."
        "It is an error if an axis does not has dimension 1.");
  }
};  // struct SqueezeAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public tvm::AttrsNode<RepeatAttrs> {
  int repeats;
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(RepeatAttrs, "relax.attrs.RepeatAttrs") {
    TVM_ATTR_FIELD(repeats).describe("The number of repetitions.");
    TVM_ATTR_FIELD(axis).describe(
        "The axis along which to repeat values. The negative numbers are interpreted "
        "counting from the backward. By default, use the flattened input array, and "
        "return a flat output array.");
  }
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public tvm::AttrsNode<TileAttrs> {
  Array<Integer> repeats;

  TVM_DECLARE_ATTRS(TileAttrs, "relax.attrs.TileAttrs") {
    TVM_ATTR_FIELD(repeats).describe("The number of repetitions of data along each axis.");
  }
};  // struct TileAttrs

/*! \brief Attributes used in cumsum operators */
struct CumsumAttrs : public tvm::AttrsNode<CumsumAttrs> {
  Optional<Integer> axis;
  DataType dtype;

  TVM_DECLARE_ATTRS(CumsumAttrs, "relax.attrs.CumsumAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "Axis along which the cumulative sum is computed."
        "The default (None) is to compute the cumsum over the flattened array.");
    TVM_ATTR_FIELD(dtype).describe(
        "Type of the returned array and of the accumulator in which the elements are summed."
        "If dtype is not specified, it defaults to the dtype of data.");
  }
};  // struct CumsumAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_MANIPULATE_H_
