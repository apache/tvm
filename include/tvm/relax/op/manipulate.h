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
 * \file tvm/relax/op/manipulate.h
 * \brief Attributes for tensor manipulation operators.
 */
#ifndef TVM_RELAX_OP_MANIPULATE_H_
#define TVM_RELAX_OP_MANIPULATE_H_

#include <tvm/relax/expr.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in concat operators */
struct ConcatAttrsNode : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ConcatAttrs", ConcatAttrsNode, AttrsNode);
};  // struct ConcatAttrsNode

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrsNode : public AttrsNode {
  ffi::Array<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ExpandDimsAttrs", ExpandDimsAttrsNode, AttrsNode);
};  // struct ExpandDimsAttrsNode

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrsNode : public AttrsNode {
  explicit LayoutTransformAttrsNode(tirx::IndexMap value) : index_map(std::move(value)) {}
  explicit LayoutTransformAttrsNode(ffi::UnsafeInit) : index_map(ffi::UnsafeInit{}) {}
  tirx::IndexMap index_map;
  // pad_value is chosen to be of PrimExpr type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimExpr is evolved to represent symbolic expression in future.
  ffi::Optional<PrimExpr> pad_value;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayoutTransformAttrs", LayoutTransformAttrsNode,
                                    AttrsNode);
};  // struct LayoutTransformAttrsNode

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrsNode : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axes;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PermuteDimsAttrs", PermuteDimsAttrsNode,
                                    AttrsNode);
};  // struct PermuteDimsAttrsNode

/*! \brief Attributes used in split operator */
struct SplitAttrsNode : public AttrsNode {
  ffi::ObjectRef indices_or_sections;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SplitAttrs", SplitAttrsNode, AttrsNode);
};  // struct SplitAttrsNode

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrsNode : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SqueezeAttrs", SqueezeAttrsNode, AttrsNode);
};  // struct SqueezeAttrsNode

/*! \brief Attributes used in stack operators */
struct StackAttrsNode : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StackAttrs", StackAttrsNode, AttrsNode);
};  // struct StackAttrsNode

/*! \brief Attributes used in repeat operators */
struct RepeatAttrsNode : public AttrsNode {
  int repeats;
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RepeatAttrs", RepeatAttrsNode, AttrsNode);
};  // struct RepeatAttrsNode

/*! \brief Attributes used in tile operators */
struct TileAttrsNode : public AttrsNode {
  ffi::Array<int64_t> repeats;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TileAttrs", TileAttrsNode, AttrsNode);
};  // struct TileAttrsNode

/*! \brief Attributes used in flip operators */
struct FlipAttrsNode : public AttrsNode {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.FlipAttrs", FlipAttrsNode, AttrsNode);
};  // struct FlipAttrsNode

/*! \brief Attributes used in reverse_sequence operators */
struct ReverseSequenceAttrsNode : public AttrsNode {
  int64_t seq_axis;
  int64_t batch_axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ReverseSequenceAttrs", ReverseSequenceAttrsNode,
                                    AttrsNode);
};  // struct ReverseSequenceAttrsNode

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrsNode : public AttrsNode {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherElementsAttrs", GatherElementsAttrsNode,
                                    AttrsNode);
};  // struct GatherElementsAttrsNode

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrsNode : public AttrsNode {
  int64_t batch_dims;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherNDAttrs", GatherNDAttrsNode, AttrsNode);
};  // struct GatherNDAttrsNode

/*! \brief Attributes used in index_put operator */
struct IndexPutAttrsNode : public AttrsNode {
  bool accumulate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.IndexPutAttrs", IndexPutAttrsNode, AttrsNode);
};  // struct IndexPutAttrsNode

/*! \brief Attribute used in meshgrid operator */
struct MeshgridAttrsNode : public AttrsNode {
  ffi::Optional<ffi::String> indexing;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MeshgridAttrs", MeshgridAttrsNode, AttrsNode);
};

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrsNode : public AttrsNode {
  int64_t axis;
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterElementsAttrs", ScatterElementsAttrsNode,
                                    AttrsNode);
};  // struct ScatterElementsAttrsNode

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrsNode : public AttrsNode {
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterNDAttrs", ScatterNDAttrsNode, AttrsNode);
};  // struct ScatterNDAttrsNode

/*! \brief Attributes used in slice_scatter operator */
struct SliceScatterAttrsNode : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SliceScatterAttrs", SliceScatterAttrsNode,
                                    AttrsNode);
};  // struct SliceScatterAttrsNode

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrsNode : public AttrsNode {
  int depth;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.OneHotAttrs", OneHotAttrsNode, AttrsNode);
};  // struct OneHotAttrsNode

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_MANIPULATE_H_
