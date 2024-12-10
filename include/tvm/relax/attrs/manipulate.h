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
  /*!
   * axis_separators between input axes when generating flattened output axes. For buffers
   * representing flat 1-d memory (e.g. any buffer in RAM), this should be an empty array.
   * For buffers representing non-flat memory, each entry in axis_separators should be the
   * first input axis that is part of a new flattened axis.
   */
  Optional<Array<IntImm>> axis_separators;
  /*!
   * axis_separators for input buffers.
   * Needed to identify if the input buffer to layout_transform
   * contains axis separator.
   */
  Optional<Array<IntImm>> input_axis_separators;

  TVM_DECLARE_ATTRS(LayoutTransformAttrs, "relax.attrs.LayoutTransformAttrs") {
    TVM_ATTR_FIELD(index_map).describe("The layout transformation to apply.");
    TVM_ATTR_FIELD(pad_value).describe(
        "The specific value to be used to pad if the layout transform would result in implicit "
        "padding. If not specified, the compiler is free to choose any value.");
    TVM_ATTR_FIELD(axis_separators)
        .describe("The separators between input axes when generating flat output axes");
    TVM_ATTR_FIELD(input_axis_separators)
        .describe("The separators between axes to regenerate output");
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

/*! \brief Attributes used in flip operators */
struct FlipAttrs : public tvm::AttrsNode<FlipAttrs> {
  Integer axis;
  TVM_DECLARE_ATTRS(FlipAttrs, "relax.attrs.FlipAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe("The axis along which to flip over.");
  }
};  // struct FlipAttrs

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrs : public tvm::AttrsNode<GatherElementsAttrs> {
  Integer axis;

  TVM_DECLARE_ATTRS(GatherElementsAttrs, "relax.attrs.GatherElementsAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0).describe("The axis along which to index.");
  }
};  // struct GatherElementsAttrs

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrs : public tvm::AttrsNode<GatherNDAttrs> {
  Integer batch_dims;
  TVM_DECLARE_ATTRS(GatherNDAttrs, "relax.attrs.GatherNDAttrs") {
    TVM_ATTR_FIELD(batch_dims).set_default(Integer(0)).describe("The number of batch dims.");
  }
};  // struct GatherNDAttrs

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrs : public tvm::AttrsNode<ScatterElementsAttrs> {
  Integer axis;
  String reduction;

  TVM_DECLARE_ATTRS(ScatterElementsAttrs, "relax.attrs.ScatterElementsAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0).describe("The axis over which to select values.");
    TVM_ATTR_FIELD(reduction).set_default("update").describe(
        "Reduction mode of the scatter elements, "
        "either \"update\", \"add\", \"mul\", \"mean\", \"min\" or \"max\".");
  }
};  // struct ScatterElementsAttrs

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrs : public tvm::AttrsNode<ScatterNDAttrs> {
  String reduction;

  TVM_DECLARE_ATTRS(ScatterNDAttrs, "relax.attrs.ScatterNDAttrs") {
    TVM_ATTR_FIELD(reduction).set_default("update").describe(
        "Accumulation mode of the ScatterND, "
        "either \"update\", \"add\", \"mul\", \"min\" or \"max\".");
  }
};  // struct ScatterNDAttrs

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrs : public tvm::AttrsNode<OneHotAttrs> {
  int depth;
  int axis;

  TVM_DECLARE_ATTRS(OneHotAttrs, "relax.attrs.OneHotAttrs") {
    TVM_ATTR_FIELD(depth).describe("Depth of the one hot dimension.");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Axis to fill.");
  }
};  // struct OneHotAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_MANIPULATE_H_
