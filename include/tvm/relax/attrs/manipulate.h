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
struct ConcatAttrs : public AttrsNodeReflAdapter<ConcatAttrs> {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConcatAttrs>().def_ro("axis", &ConcatAttrs::axis,
                                          "The axis at which the input arrays are concatenated."
                                          "Should lie in range `[-ndim, ndim)`.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ConcatAttrs", ConcatAttrs, BaseAttrsNode);
};  // struct ConcatAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public AttrsNodeReflAdapter<ExpandDimsAttrs> {
  ffi::Array<Integer> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExpandDimsAttrs>().def_ro(
        "axis", &ExpandDimsAttrs::axis,
        "The axes at which the input array are expanded. "
        "All values are required to lie in range `[-data.ndim - 1, data.ndim]`, "
        "with the convention of negative indexing.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ExpandDimsAttrs", ExpandDimsAttrs, BaseAttrsNode);
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrs : public AttrsNodeReflAdapter<LayoutTransformAttrs> {
  tir::IndexMap index_map;
  // pad_value is chosen to be of PrimValue type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimValue is evolved to represent symbolic expression in future.
  ffi::Optional<PrimValue> pad_value;
  /*!
   * axis_separators between input axes when generating flattened output axes. For buffers
   * representing flat 1-d memory (e.g. any buffer in RAM), this should be an empty array.
   * For buffers representing non-flat memory, each entry in axis_separators should be the
   * first input axis that is part of a new flattened axis.
   */
  ffi::Optional<ffi::Array<IntImm>> axis_separators;
  /*!
   * axis_separators for input buffers.
   * Needed to identify if the input buffer to layout_transform
   * contains axis separator.
   */
  ffi::Optional<ffi::Array<IntImm>> input_axis_separators;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutTransformAttrs>()
        .def_ro("index_map", &LayoutTransformAttrs::index_map,
                "The layout transformation to apply.")
        .def_ro(
            "pad_value", &LayoutTransformAttrs::pad_value,
            "The specific value to be used to pad if the layout transform would result in implicit "
            "padding. If not specified, the compiler is free to choose any value.")
        .def_ro("axis_separators", &LayoutTransformAttrs::axis_separators,
                "The separators between input axes when generating flat output axes")
        .def_ro("input_axis_separators", &LayoutTransformAttrs::input_axis_separators,
                "The separators between axes to regenerate output");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayoutTransformAttrs", LayoutTransformAttrs,
                                    BaseAttrsNode);
};  // struct LayoutTransformAttrs

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrs : public AttrsNodeReflAdapter<PermuteDimsAttrs> {
  ffi::Optional<ffi::Array<Integer>> axes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PermuteDimsAttrs>().def_ro(
        "axes", &PermuteDimsAttrs::axes, "The target axes order, reverse order if not specified.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PermuteDimsAttrs", PermuteDimsAttrs,
                                    BaseAttrsNode);
};  // struct PermuteDimsAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public AttrsNodeReflAdapter<SplitAttrs> {
  ObjectRef indices_or_sections;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SplitAttrs>()
        .def_ro("indices_or_sections", &SplitAttrs::indices_or_sections,
                "The input array of indices or the number of split sections.")
        .def_ro("axis", &SplitAttrs::axis, "The axis to be splitted");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SplitAttrs", SplitAttrs, BaseAttrsNode);
};  // struct SplitAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public AttrsNodeReflAdapter<SqueezeAttrs> {
  ffi::Optional<ffi::Array<Integer>> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SqueezeAttrs>().def_ro("axis", &SqueezeAttrs::axis,
                                           "The axis to squeeze in the input tensor."
                                           "If `axis = None`, all axis of dimension 1 get squeezed;"
                                           "Else, the dimension in axes get squeezed."
                                           "It is an error if an axis does not has dimension 1.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SqueezeAttrs", SqueezeAttrs, BaseAttrsNode);
};  // struct SqueezeAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public AttrsNodeReflAdapter<StackAttrs> {
  ffi::Optional<Integer> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StackAttrs>().def_ro(
        "axis", &StackAttrs::axis,
        "The axis along which to stack the input tensors. "
        "The axis will be inserted at this position in the output, "
        "so it must be in range [-ndim-1, ndim] where ndim is the "
        "number of dimensions of the input tensors.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StackAttrs", StackAttrs, BaseAttrsNode);
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public AttrsNodeReflAdapter<RepeatAttrs> {
  int repeats;
  ffi::Optional<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RepeatAttrs>()
        .def_ro("repeats", &RepeatAttrs::repeats, "The number of repetitions.")
        .def_ro("axis", &RepeatAttrs::axis,
                "The axis along which to repeat values. The negative numbers are interpreted "
                "counting from the backward. By default, use the flattened input array, and "
                "return a flat output array.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RepeatAttrs", RepeatAttrs, BaseAttrsNode);
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public AttrsNodeReflAdapter<TileAttrs> {
  ffi::Array<Integer> repeats;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TileAttrs>().def_ro("repeats", &TileAttrs::repeats,
                                        "The number of repetitions of data along each axis.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TileAttrs", TileAttrs, BaseAttrsNode);
};  // struct TileAttrs

/*! \brief Attributes used in flip operators */
struct FlipAttrs : public AttrsNodeReflAdapter<FlipAttrs> {
  Integer axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FlipAttrs>().def_ro("axis", &FlipAttrs::axis,
                                        "The axis along which to flip over.",
                                        refl::DefaultValue(NullValue<Integer>()));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.FlipAttrs", FlipAttrs, BaseAttrsNode);
};  // struct FlipAttrs

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrs : public AttrsNodeReflAdapter<GatherElementsAttrs> {
  Integer axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GatherElementsAttrs>().def_ro("axis", &GatherElementsAttrs::axis,
                                                  "The axis along which to index.",
                                                  refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherElementsAttrs", GatherElementsAttrs,
                                    BaseAttrsNode);
};  // struct GatherElementsAttrs

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrs : public AttrsNodeReflAdapter<GatherNDAttrs> {
  Integer batch_dims;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GatherNDAttrs>().def_ro("batch_dims", &GatherNDAttrs::batch_dims,
                                            "The number of batch dims.", refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherNDAttrs", GatherNDAttrs, BaseAttrsNode);
};  // struct GatherNDAttrs

/*! \brief Attributes used in index_put operator */
struct IndexPutAttrs : public AttrsNodeReflAdapter<IndexPutAttrs> {
  bool accumulate;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IndexPutAttrs>().def_ro(
        "accumulate", &IndexPutAttrs::accumulate,
        "Whether to accumulate (add) values rather than replace. "
        "If true, performs tensor[indices] += values, "
        "otherwise performs tensor[indices] = values.",
        refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.IndexPutAttrs", IndexPutAttrs, BaseAttrsNode);
};  // struct IndexPutAttrs

/*! \brief Attribute used in meshgrid operator */
struct MeshgridAttrs : public AttrsNodeReflAdapter<MeshgridAttrs> {
  ffi::Optional<ffi::String> indexing;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MeshgridAttrs>().def_ro("indexing", &MeshgridAttrs::indexing,
                                            "Specifies how the grid dimensions are ordered.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MeshgridAttrs", MeshgridAttrs, BaseAttrsNode);
};

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrs : public AttrsNodeReflAdapter<ScatterElementsAttrs> {
  Integer axis;
  ffi::String reduction;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScatterElementsAttrs>()
        .def_ro("axis", &ScatterElementsAttrs::axis, "The axis over which to select values.",
                refl::DefaultValue(0))
        .def_ro("reduction", &ScatterElementsAttrs::reduction,
                "Reduction mode of the scatter elements, "
                "either \"update\", \"add\", \"mul\", \"mean\", \"min\" or \"max\".",
                refl::DefaultValue("update"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterElementsAttrs", ScatterElementsAttrs,
                                    BaseAttrsNode);
};  // struct ScatterElementsAttrs

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrs : public AttrsNodeReflAdapter<ScatterNDAttrs> {
  ffi::String reduction;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScatterNDAttrs>().def_ro(
        "reduction", &ScatterNDAttrs::reduction,
        "Accumulation mode of the ScatterND, "
        "either \"update\", \"add\", \"mul\", \"min\" or \"max\".",
        refl::DefaultValue("update"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterNDAttrs", ScatterNDAttrs, BaseAttrsNode);
};  // struct ScatterNDAttrs

/*! \brief Attributes used in slice_scatter operator */
struct SliceScatterAttrs : public AttrsNodeReflAdapter<SliceScatterAttrs> {
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SliceScatterAttrs>().def_ro("axis", &SliceScatterAttrs::axis,
                                                "the dimension to insert the slice into ",
                                                refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SliceScatterAttrs", SliceScatterAttrs,
                                    BaseAttrsNode);
};  // struct SliceScatterAttrs

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrs : public AttrsNodeReflAdapter<OneHotAttrs> {
  int depth;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OneHotAttrs>()
        .def_ro("depth", &OneHotAttrs::depth, "Depth of the one hot dimension.")
        .def_ro("axis", &OneHotAttrs::axis, "Axis to fill.", refl::DefaultValue(-1));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.OneHotAttrs", OneHotAttrs, BaseAttrsNode);
};  // struct OneHotAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_MANIPULATE_H_
