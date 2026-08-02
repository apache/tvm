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
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in concat operators */
struct ConcatAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConcatAttrs>().def_ro("axis", &ConcatAttrs::axis,
                                          "The axis at which the input arrays are concatenated."
                                          "Should lie in range `[-ndim, ndim)`.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ConcatAttrs", ConcatAttrs, AttrsNode);
};  // struct ConcatAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public AttrsNode {
  ffi::Array<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExpandDimsAttrs>().def_ro(
        "axis", &ExpandDimsAttrs::axis,
        "The axes at which the input array are expanded. "
        "All values are required to lie in range `[-data.ndim - 1, data.ndim]`, "
        "with the convention of negative indexing.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ExpandDimsAttrs", ExpandDimsAttrs, AttrsNode);
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrs : public AttrsNode {
  tirx::IndexMap index_map;
  // pad_value is chosen to be of PrimExpr type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimExpr is evolved to represent symbolic expression in future.
  ffi::Optional<PrimExpr> pad_value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutTransformAttrs>()
        .def_ro("index_map", &LayoutTransformAttrs::index_map,
                "The layout transformation to apply.")
        .def_ro(
            "pad_value", &LayoutTransformAttrs::pad_value,
            "The specific value to be used to pad if the layout transform would result in implicit "
            "padding. If not specified, the compiler is free to choose any value.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayoutTransformAttrs", LayoutTransformAttrs,
                                    AttrsNode);
};  // struct LayoutTransformAttrs

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PermuteDimsAttrs>().def_ro(
        "axes", &PermuteDimsAttrs::axes, "The target axes order, reverse order if not specified.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PermuteDimsAttrs", PermuteDimsAttrs, AttrsNode);
};  // struct PermuteDimsAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public AttrsNode {
  ffi::ObjectRef indices_or_sections;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SplitAttrs>()
        .def_ro("indices_or_sections", &SplitAttrs::indices_or_sections,
                "The input array of indices or the number of split sections.")
        .def_ro("axis", &SplitAttrs::axis, "The axis to be splitted");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SplitAttrs", SplitAttrs, AttrsNode);
};  // struct SplitAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SqueezeAttrs>().def_ro("axis", &SqueezeAttrs::axis,
                                           "The axis to squeeze in the input tensor."
                                           "If `axis = None`, all axis of dimension 1 get squeezed;"
                                           "Else, the dimension in axes get squeezed."
                                           "It is an error if an axis does not has dimension 1.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SqueezeAttrs", SqueezeAttrs, AttrsNode);
};  // struct SqueezeAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StackAttrs>().def_ro(
        "axis", &StackAttrs::axis,
        "The axis along which to stack the input tensors. "
        "The axis will be inserted at this position in the output, "
        "so it must be in range [-ndim-1, ndim] where ndim is the "
        "number of dimensions of the input tensors.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StackAttrs", StackAttrs, AttrsNode);
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public AttrsNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RepeatAttrs", RepeatAttrs, AttrsNode);
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public AttrsNode {
  ffi::Array<int64_t> repeats;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TileAttrs>().def_ro("repeats", &TileAttrs::repeats,
                                        "The number of repetitions of data along each axis.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TileAttrs", TileAttrs, AttrsNode);
};  // struct TileAttrs

/*! \brief Attributes used in flip operators */
struct FlipAttrs : public AttrsNode {
  int64_t axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FlipAttrs>().def_ro("axis", &FlipAttrs::axis,
                                        "The axis along which to flip over.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.FlipAttrs", FlipAttrs, AttrsNode);
};  // struct FlipAttrs

/*! \brief Attributes used in reverse_sequence operators */
struct ReverseSequenceAttrs : public AttrsNode {
  int64_t seq_axis;
  int64_t batch_axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReverseSequenceAttrs>()
        .def_ro("seq_axis", &ReverseSequenceAttrs::seq_axis,
                "The axis along which to reverse variable length slices.")
        .def_ro("batch_axis", &ReverseSequenceAttrs::batch_axis,
                "The axis that indexes the batch.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ReverseSequenceAttrs", ReverseSequenceAttrs,
                                    AttrsNode);
};  // struct ReverseSequenceAttrs

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrs : public AttrsNode {
  int64_t axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GatherElementsAttrs>().def_ro("axis", &GatherElementsAttrs::axis,
                                                  "The axis along which to index.",
                                                  refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherElementsAttrs", GatherElementsAttrs,
                                    AttrsNode);
};  // struct GatherElementsAttrs

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrs : public AttrsNode {
  int64_t batch_dims;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GatherNDAttrs>().def_ro("batch_dims", &GatherNDAttrs::batch_dims,
                                            "The number of batch dims.", refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherNDAttrs", GatherNDAttrs, AttrsNode);
};  // struct GatherNDAttrs

/*! \brief Attributes used in index_put operator */
struct IndexPutAttrs : public AttrsNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.IndexPutAttrs", IndexPutAttrs, AttrsNode);
};  // struct IndexPutAttrs

/*! \brief Attribute used in meshgrid operator */
struct MeshgridAttrs : public AttrsNode {
  ffi::Optional<ffi::String> indexing;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MeshgridAttrs>().def_ro("indexing", &MeshgridAttrs::indexing,
                                            "Specifies how the grid dimensions are ordered.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MeshgridAttrs", MeshgridAttrs, AttrsNode);
};

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrs : public AttrsNode {
  int64_t axis;
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
                                    AttrsNode);
};  // struct ScatterElementsAttrs

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrs : public AttrsNode {
  ffi::String reduction;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScatterNDAttrs>().def_ro(
        "reduction", &ScatterNDAttrs::reduction,
        "Accumulation mode of the ScatterND, "
        "either \"update\", \"add\", \"mul\", \"min\" or \"max\".",
        refl::DefaultValue("update"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterNDAttrs", ScatterNDAttrs, AttrsNode);
};  // struct ScatterNDAttrs

/*! \brief Attributes used in slice_scatter operator */
struct SliceScatterAttrs : public AttrsNode {
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SliceScatterAttrs>().def_ro("axis", &SliceScatterAttrs::axis,
                                                "the dimension to insert the slice into ",
                                                refl::DefaultValue(0));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SliceScatterAttrs", SliceScatterAttrs, AttrsNode);
};  // struct SliceScatterAttrs

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrs : public AttrsNode {
  int depth;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OneHotAttrs>()
        .def_ro("depth", &OneHotAttrs::depth, "Depth of the one hot dimension.")
        .def_ro("axis", &OneHotAttrs::axis, "Axis to fill.", refl::DefaultValue(-1));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.OneHotAttrs", OneHotAttrs, AttrsNode);
};  // struct OneHotAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_MANIPULATE_H_
