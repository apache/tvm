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
 *//*!
 * \file tvm/tirx/layout.h
 * \brief Definition of layout
 */

#ifndef TVM_TIRX_LAYOUT_H_
#define TVM_TIRX_LAYOUT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/enum.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/var.h>

namespace tvm {

namespace tirx {

class Layout;
class TileLayout;
class Iter;
using ffi::Array;
using ffi::Tuple;

// Base class for layout
class LayoutNode : public ffi::Object {
 public:
  /*! \brief Compatible with shape */
  virtual bool CompatibleWithShape(const ffi::Array<PrimExpr>& shape) const = 0;

  /*! \brief Verify if the layout is well-formed */
  virtual bool VerifyWellFormed() const = 0;

  /*! \brief Get the size of the layout (of some axis) */
  virtual PrimExpr GetSize(ffi::Optional<ffi::String> axis_name = std::nullopt) const = 0;

  /*! \brief Get the span of the layout (of some axis) */
  virtual PrimExpr GetSpan(ffi::Optional<ffi::String> axis_name = std::nullopt) const = 0;

  /*! \brief Apply layout on the input coordinate and get the mapped output */
  virtual ffi::Map<ffi::String, PrimExpr> Apply(ffi::Array<PrimExpr> coord) const = 0;
  virtual ffi::Map<ffi::String, PrimExpr> Apply(PrimExpr coord) const = 0;
  virtual ffi::Map<ffi::String, PrimExpr> Apply(const ffi::Array<PrimExpr>& coord,
                                                const ffi::Array<PrimExpr>& shape) const;

  /*! \brief Turn the layout to canonical form */
  virtual Layout Canonicalize() const = 0;

  /*! \brief Tile the current layout with a given layout */
  virtual Layout Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                      const ffi::Array<PrimExpr>& inner_shape) const = 0;

  /*! \brief Slice the layout with a given shape and region */
  virtual ffi::Optional<Layout> Slice(const ffi::Array<PrimExpr>& shape,
                                      const Region& region) const = 0;

  /*! \brief Direct-sum on the tiling domain (unscaled composition)
   *  Given left layout A (grouped by left_shape) and this layout B (grouped by right_shape),
   *  construct the interleaved-domain direct sum A + B without span scaling.
   */
  virtual Layout DirectSum(const TileLayout& left, const ffi::Array<PrimExpr>& left_shape,
                           const ffi::Array<PrimExpr>& right_shape) const = 0;

  /*! \brief Check if the layout is the inner layout of a tiled layout
   * \param tile_layout The tiled layout to check
   * \param tiled_shape The shape of the tiled layout
   * \param inner_shape The shape of the inner layout
   * \return The outer layout if this layout is the inner layout of tile_layout, std::nullopt
   * otherwise
   */
  virtual ffi::Optional<TileLayout> IsTileInner(const Layout& tile_layout,
                                                const ffi::Array<PrimExpr>& tiled_shape,
                                                const ffi::Array<PrimExpr>& inner_shape) const = 0;

  /*! \brief Check if the layout is the outer layout of a tiled layout
   * \param tile_layout The tiled layout to check
   * \param tiled_shape The shape of the tiled layout
   * \param outer_shape The shape of the outer layout
   * \return The inner layout if this layout is the outer layout of tile_layout, std::nullopt
   * otherwise
   */
  virtual ffi::Optional<Layout> IsTileOuter(const Layout& tile_layout,
                                            const ffi::Array<PrimExpr>& tiled_shape,
                                            const ffi::Array<PrimExpr>& outer_shape) const = 0;

  /*! \brief Check if this layout is the right addend B in a direct-sum A + B over the
   *         interleaved domain S_A \otimes S_B. If so, return the left layout A.
   *  \param sum_layout The resulting direct-sum layout
   *  \param interleaved_shape The interleaved domain S_A \otimes S_B, i.e., [A0, B0, A1, B1, ...]
   *  \param right_shape The shape that groups this (right) layout
   */
  virtual ffi::Optional<TileLayout> IsDirectSumRight(
      const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
      const ffi::Array<PrimExpr>& right_shape) const = 0;

  /*! \brief Check if this layout is the left addend A in a direct-sum A + B over the
   *         interleaved domain S_A \otimes S_B. If so, return the right layout B.
   *  \param sum_layout The resulting direct-sum layout
   *  \param interleaved_shape The interleaved domain S_A \otimes S_B, i.e., [A0, B0, A1, B1, ...]
   *  \param left_shape The shape that groups this (left) layout
   */
  virtual ffi::Optional<Layout> IsDirectSumLeft(const Layout& sum_layout,
                                                const ffi::Array<PrimExpr>& interleaved_shape,
                                                const ffi::Array<PrimExpr>& left_shape) const = 0;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.Layout", LayoutNode, ffi::Object);
};

class Layout : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ffi::ObjectRef, LayoutNode);
};

// target, subscope, scope, iter -> fused_iter
using FAxisFuser = ffi::TypedFunction<ffi::Optional<Iter>(Target, ffi::String, ffi::String, Iter)>;
// target, scope, iter -> (outer_iter, inner_iter)
// Note(@bohao): use ffi::Array<Iter, void> to avoid incomplete type error (SFINAE)
using FAxisSplitter = ffi::TypedFunction<ffi::Array<Iter, void>(Target, ffi::String, Iter)>;

// Axis
class AxisNode : public ffi::EnumObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AxisNode>().def_ro("name", &AxisNode::_str_index);
  }

  /*! \brief Check if the axis is a thread axis. */
  bool IsThreadAxis() const;

  /*! \brief Check if the axis is a memory axis. */
  bool IsMemoryAxis() const;

  /*! \brief Get the scope of the (thread) axis. */
  ffi::Optional<ExecScope> GetScope() const;

  /*! \brief Get the subscope of the (thread) axis. */
  ffi::Optional<ExecScope> GetSubscope() const;

  /*! \brief Get the fuser of the (thread) axis. */
  ffi::Optional<FAxisFuser> GetFuser() const;

  /*! \brief Get the splitter of the (thread) axis. */
  ffi::Optional<FAxisSplitter> GetSplitter() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Axis", AxisNode, ffi::EnumObj);
};

class Axis : public ffi::Enum {
 public:
  Axis() = default;

  /*!
   * \brief Get or create the axis with the given name.
   * \param name Name of the axis to look up or register.
   * \return The registered axis singleton.
   */
  TVM_DLL static Axis Get(const ffi::String& name);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Axis, ffi::Enum, AxisNode);
};

class IterNode : public ffi::Object {
 public:
  PrimExpr extent;
  PrimExpr stride;
  Axis axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IterNode>()
        .def_ro("extent", &IterNode::extent)
        .def_ro("stride", &IterNode::stride)
        .def_ro("axis", &IterNode::axis);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Iter", IterNode, ffi::Object);
};

class Iter : public ffi::ObjectRef {
 public:
  TVM_DLL explicit Iter(PrimExpr extent, PrimExpr stride, Axis axis);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Iter, ffi::ObjectRef, IterNode);
};

class TileLayoutNode : public LayoutNode {
 public:
  ffi::Array<Iter> shard;
  ffi::Array<Iter> replica;
  ffi::Map<Axis, PrimExpr> offset;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TileLayoutNode>()
        .def_ro("shard", &TileLayoutNode::shard)
        .def_ro("replica", &TileLayoutNode::replica)
        .def_ro("offset", &TileLayoutNode::offset);
  }

  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const ffi::Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout (of some axis) */
  PrimExpr GetSize(ffi::Optional<ffi::String> axis_name = std::nullopt) const final;

  /*! \brief Get the span of the layout (of some axis) */
  PrimExpr GetSpan(ffi::Optional<ffi::String> axis_name = std::nullopt) const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  ffi::Map<ffi::String, PrimExpr> Apply(ffi::Array<PrimExpr> coord) const final;
  ffi::Map<ffi::String, PrimExpr> Apply(PrimExpr coord) const final;
  /*! \brief Group-first override: if this layout can be regrouped by ``shape``,
   *  split each ``coord[d]`` against its group's local extents (cleaner
   *  symbolic form than flatten+split-against-shard-shape). Otherwise fall
   *  back to flatten+split. */
  ffi::Map<ffi::String, PrimExpr> Apply(const ffi::Array<PrimExpr>& coord,
                                        const ffi::Array<PrimExpr>& shape) const final;

  /*! \brief Turn the layout to canonical form */
  Layout Canonicalize() const final;

  /*! \brief Tile the layout with an outer layout */
  Layout Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
              const ffi::Array<PrimExpr>& inner_shape) const final;

  Layout DirectSum(const TileLayout& left, const ffi::Array<PrimExpr>& left_shape,
                   const ffi::Array<PrimExpr>& right_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  ffi::Optional<TileLayout> IsTileInner(const Layout& tile_layout,
                                        const ffi::Array<PrimExpr>& tiled_shape,
                                        const ffi::Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  ffi::Optional<Layout> IsTileOuter(const Layout& tile_layout,
                                    const ffi::Array<PrimExpr>& tiled_shape,
                                    const ffi::Array<PrimExpr>& outer_shape) const final;

  ffi::Optional<TileLayout> IsDirectSumRight(const Layout& sum_layout,
                                             const ffi::Array<PrimExpr>& interleaved_shape,
                                             const ffi::Array<PrimExpr>& right_shape) const final;

  ffi::Optional<Layout> IsDirectSumLeft(const Layout& sum_layout,
                                        const ffi::Array<PrimExpr>& interleaved_shape,
                                        const ffi::Array<PrimExpr>& left_shape) const final;

  /*! \brief Get the shape of the shard */
  ffi::Array<PrimExpr> GetShardShape() const;

  /*! \brief Slice the layout with a given shape and region */
  ffi::Optional<Layout> Slice(const ffi::Array<PrimExpr>& shape, const Region& region) const final;

  /*! \brief Is the layout trivial (pure memory, identical mapping) */
  bool IsTrivial() const;

  /*! \brief Check if the layout is trainium layout */
  bool IsTrainium() const;

  /*! \brief Has Memory Axis */
  bool HasMemoryAxis() const;

  /*! \brief Has Thread Axis */
  bool HasThreadAxis() const;

  /*! \brief Get the scope pair of the layout */
  ffi::Optional<Tuple<ExecScope, ExecScope>> GetScope() const;

  /*! \brief Get the default layout for the shape */
  static TileLayout DefaultLayout(ffi::Array<PrimExpr> shape);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TileLayout", TileLayoutNode, LayoutNode);
};

class TileLayout : public Layout {
 public:
  TVM_DLL explicit TileLayout(ffi::Array<Iter> shard, ffi::Array<Iter> replica,
                              ffi::Map<Axis, PrimExpr> offset);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileLayout, Layout, TileLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileLayoutNode);
};

// ComposeLayout
class ComposeLayoutNode : public LayoutNode {
 public:
  int per_element;
  int swizzle_len;
  int atom_len;
  bool swizzle_inner;
  TileLayout tile_layout;
  // Cached swizzle masks: inner_mask = (1 << swizzle_len) - 1;
  // outer_mask = inner_mask << atom_len. Set by the ComposeLayout ctor.
  int inner_mask;
  int outer_mask;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ComposeLayoutNode>()
        .def_ro("per_element", &ComposeLayoutNode::per_element)
        .def_ro("swizzle_len", &ComposeLayoutNode::swizzle_len)
        .def_ro("atom_len", &ComposeLayoutNode::atom_len)
        .def_ro("swizzle_inner", &ComposeLayoutNode::swizzle_inner)
        .def_ro("inner_mask", &ComposeLayoutNode::inner_mask)
        .def_ro("outer_mask", &ComposeLayoutNode::outer_mask)
        .def_ro("tile_layout", &ComposeLayoutNode::tile_layout);
  }

  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const ffi::Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size (of some axis) of the layout */
  PrimExpr GetSize(ffi::Optional<ffi::String> axis_name = std::nullopt) const final;

  /*! \brief Get the span (of some axis) of the layout */
  PrimExpr GetSpan(ffi::Optional<ffi::String> axis_name = std::nullopt) const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  ffi::Map<ffi::String, PrimExpr> Apply(ffi::Array<PrimExpr> coord) const final;
  ffi::Map<ffi::String, PrimExpr> Apply(PrimExpr coord) const final;
  ffi::Map<ffi::String, PrimExpr> Apply(const ffi::Array<PrimExpr>& coord,
                                        const ffi::Array<PrimExpr>& shape) const final;

  /*! \brief Turn the layout to canonical form */
  Layout Canonicalize() const final;

  /*! \brief Tile the layout with an outer layout */
  Layout Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
              const ffi::Array<PrimExpr>& inner_shape) const final;

  Layout DirectSum(const TileLayout& left, const ffi::Array<PrimExpr>& left_shape,
                   const ffi::Array<PrimExpr>& right_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  ffi::Optional<TileLayout> IsTileInner(const Layout& tile_layout,
                                        const ffi::Array<PrimExpr>& tiled_shape,
                                        const ffi::Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  ffi::Optional<Layout> IsTileOuter(const Layout& tile_layout,
                                    const ffi::Array<PrimExpr>& tiled_shape,
                                    const ffi::Array<PrimExpr>& outer_shape) const final;

  ffi::Optional<TileLayout> IsDirectSumRight(const Layout& sum_layout,
                                             const ffi::Array<PrimExpr>& interleaved_shape,
                                             const ffi::Array<PrimExpr>& right_shape) const final;

  ffi::Optional<Layout> IsDirectSumLeft(const Layout& sum_layout,
                                        const ffi::Array<PrimExpr>& interleaved_shape,
                                        const ffi::Array<PrimExpr>& left_shape) const final;

  /*! \brief Slice the layout with a given shape and region */
  ffi::Optional<Layout> Slice(const ffi::Array<PrimExpr>& shape, const Region& region) const final;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ComposeLayout", ComposeLayoutNode, LayoutNode);
};

class ComposeLayout : public Layout {
 public:
  TVM_DLL explicit ComposeLayout(int per_element, int swizzle_len, int atom_len,
                                 TileLayout tile_layout, bool swizzle_inner = true);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComposeLayout, Layout, ComposeLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComposeLayoutNode);
};

constexpr int kPSUMMaxElemPerBank = 512;
constexpr int kPSUMBankNum = 8;

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
