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
#include <tvm/runtime/device_api.h>  // For `kAllocAlignment`
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>

#include <algorithm>
#include <utility>

#include "./utils.h"

namespace tvm {
namespace script {

namespace printer {

ffi::Map<ffi::String, ExprDoc> BufferAttrs(
    tirx::BufferType buffer, const AccessPath& buffer_p, const Frame& frame, const IRDocsifier& d,
    BufferVarDefinition var_definitions, ffi::Optional<Expr> data = std::nullopt,
    ffi::Optional<tirx::BufferVar> buffer_var = std::nullopt) {
  using tvm::tirx::Var;
  using tvm::tirx::VarNode;
  ffi::Map<ffi::String, ExprDoc> kwargs;
  ffi::Array<ExprDoc> var_def_lhs;
  ffi::Array<ExprDoc> var_def_rhs;

  // Step 0. Set up statistics
  std::unordered_map<const ffi::Object*, int> use_count;
  std::unordered_set<const ffi::Object*> def_seen;
  auto count_buffer_var = [&](const Var& var,
                              TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
    if (kind != kTVMFFIDefRegionKindNone) {
      if (!def_seen.insert(var.get()).second) {
        return ffi::WalkResult::Skip();
      }
      return ffi::WalkResult::Advance();
    }
    ++use_count[var.get()];
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(buffer, count_buffer_var);
  if (data.has_value()) {
    auto count_data_var = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      ++use_count[var.get()];
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(data.value(), count_data_var);
  }
  auto is_new_var = [&](const Expr& e) { return e->IsInstance<VarNode>() && !d->IsVarDefined(e); };
  auto add_out_of_line_var_def = [&](const Var& var, const AccessPath& var_p) {
    TVM_FFI_ICHECK(!d->IsVarDefined(var));
    ExprDoc lhs = DefineVar(var, frame, d);
    lhs->source_paths.push_back(var_p);
    var_def_lhs.push_back(lhs);
    var_def_rhs.push_back(PrintVarCreation(var, var_p, d));
  };
  auto try_inline_def = [&](const Expr& e, const AccessPath& e_p,
                            std::function<ExprDoc()> inline_f) {
    TVM_FFI_ICHECK(is_new_var(e));
    Var var = e.as_or_throw<Var>();
    if (use_count[var.get()] == 1) {
      d->Define(e, frame, inline_f);
      return true;
    } else {
      add_out_of_line_var_def(var, e_p);
      return false;
    }
  };
  // Step 1. Handle `buffer.shape`
  {
    const ffi::Array<PrimExpr>& shape = buffer->shape;
    AccessPath shape_p = buffer_p->Attr("shape");
    int n = shape.size();
    ffi::Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = shape[i];
      AccessPath e_p = shape_p->ArrayItem(i);
      bool was_undefined = is_new_var(e);
      if (was_undefined) {
        add_out_of_line_var_def(e.as_or_throw<Var>(), e_p);
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("shape", TupleDoc(results));
  }
  // Step 2. Handle `buffer.dtype`
  {
    DLDataType default_buf_dtype = d->cfg->buffer_dtype;
    if (buffer->dtype->dtype != default_buf_dtype) {
      kwargs.Set("dtype", LiteralDoc::DataType(buffer->dtype->dtype, buffer_p->Attr("dtype")));
    }
  }
  // Step 3. Handle `buffer.data`
  // For tmem scope, DeclBuffer does not accept `data` (it auto-creates the data var).
  bool is_tmem_scope = buffer->storage_scope == "tmem";
  bool is_inline_data = false;
  if (!is_tmem_scope && data.has_value()) {
    Expr source = data.value();
    if (is_new_var(source)) {
      if (buffer_var.has_value() && var_definitions >= BufferVarDefinition::DataPointer) {
        is_inline_data = try_inline_def(source, buffer_p->Attr("data"), [=]() {
          return d->AsDoc<ExprDoc>(buffer_var.value(), buffer_p)->Attr("data");
        });
      } else {
        add_out_of_line_var_def(source.as_or_throw<Var>(), buffer_p->Attr("data"));
      }
    }
    if (!is_inline_data) {
      kwargs.Set("data", d->AsDoc<ExprDoc>(source, buffer_p->Attr("data")));
    }
  }
  // Step 4. Handle `buffer.strides`
  if (!buffer->strides.empty()) {
    const ffi::Array<PrimExpr>& strides = buffer->strides;
    AccessPath strides_p = buffer_p->Attr("strides");
    int n = strides.size();
    ffi::Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = strides[i];
      AccessPath e_p = strides_p->ArrayItem(i);
      if (is_new_var(e)) {
        add_out_of_line_var_def(e.as_or_throw<Var>(), e_p);
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("strides", TupleDoc(results));
  }
  // Step 5. Handle `buffer.elem_offset`
  bool needs_print_factor = false;
  if (const auto* int_imm = buffer->elem_offset.as<IntImmNode>()) {
    if (int_imm->value != 0 ||
        int_imm->ty.as_or_throw<PrimType>()->dtype != buffer->DefaultIndexType()) {
      kwargs.Set("elem_offset",
                 d->AsDoc<ExprDoc>(buffer->elem_offset, buffer_p->Attr("elem_offset")));
    }
  } else if (buffer_var.has_value() && is_new_var(buffer->elem_offset)) {
    try_inline_def(buffer->elem_offset, buffer_p->Attr("elem_offset"), [=]() {
      return d->AsDoc<ExprDoc>(buffer_var.value(), buffer_p)->Attr("elem_offset");
    });
    needs_print_factor = true;
  } else {
    kwargs.Set("elem_offset",
               d->AsDoc<ExprDoc>(buffer->elem_offset, buffer_p->Attr("elem_offset")));
  }
  // Step 6. Handle `buffer.scope`
  {
    ffi::String scope = buffer->storage_scope;
    if (scope != "global") {
      kwargs.Set("scope", LiteralDoc::Str(scope, buffer_p->Attr("storage_scope")));
    }
  }
  // Step 7. Handle `buffer.data_alignment`
  if (buffer->data_alignment != runtime::kAllocAlignment) {
    kwargs.Set("align", LiteralDoc::Int(buffer->data_alignment, buffer_p->Attr("data_alignment")));
  }
  // Step 8. Handle `buffer.offset_factor`
  if (needs_print_factor || buffer->offset_factor != 1) {
    kwargs.Set("offset_factor",
               LiteralDoc::Int(buffer->offset_factor, buffer_p->Attr("offset_factor")));
  }
  // Step 9. Handle `buffer.layout`. Track the enclosing PrimFunc's `s_tir`
  // attr: S-TIR construction uses layout=None by default, while TIRx
  // construction uses DefaultLayout(shape). Mirror
  // that here so the implicit default is omitted and the non-default value
  // is emitted explicitly (round-trips safely under `StructuralEqual`).
  bool enclosing_s_tir = false;
  for (const auto& f : d->frames) {
    if (const auto* tir_f = f.as<TIRFrameNode>()) {
      if (auto func = tir_f->tirx.as<tirx::PrimFuncNode>()) {
        if (func->attrs->dict.count(tvm::attr::kSTir)) {
          enclosing_s_tir = true;
        }
        break;
      }
    }
  }
  if (buffer->layout.has_value()) {
    bool is_default =
        ffi::StructuralEqual()(buffer->layout, tirx::TileLayoutNode::DefaultLayout(buffer->shape));
    if (!is_default) {
      kwargs.Set("layout", d->AsDoc<ExprDoc>(buffer->layout, buffer_p->Attr("layout")));
    }
  } else if (!enclosing_s_tir) {
    kwargs.Set("layout", LiteralDoc::None(buffer_p->Attr("layout")));
  }
  // Step 10. Handle `buffer.allocated_addr`
  if (!buffer->allocated_addr.empty()) {
    if (buffer->allocated_addr.size() == 1) {
      // Unwrap single-element array: DeclBuffer expects Optional<PrimExpr>, not Array.
      // Use the normal expression printer so a bound scalar alias stays a scalar
      // load, while an ordinary buffer load retains its indices.
      kwargs.Set("allocated_addr",
                 d->AsDoc<ExprDoc>(buffer->allocated_addr[0],
                                   buffer_p->Attr("allocated_addr")->ArrayItem(0)));
    } else {
      ffi::Array<ExprDoc> addresses;
      for (size_t i = 0; i < buffer->allocated_addr.size(); ++i) {
        addresses.push_back(d->AsDoc<ExprDoc>(buffer->allocated_addr[i],
                                              buffer_p->Attr("allocated_addr")->ArrayItem(i)));
      }
      kwargs.Set("allocated_addr", TupleDoc(addresses));
    }
  }

  if (var_def_lhs.size() == 1) {
    frame->stmts.push_back(AssignDoc(var_def_lhs[0], var_def_rhs[0], std::nullopt));
  } else if (var_def_lhs.size() > 1) {
    frame->stmts.push_back(AssignDoc(TupleDoc(var_def_lhs), TupleDoc(var_def_rhs), std::nullopt));
  }
  return kwargs;
}

ExprDoc BufferCall(const ExprDoc& prefix, const ffi::Map<ffi::String, ExprDoc>& attrs,
                   ffi::Array<ExprDoc> args) {
  ffi::Array<ffi::String> kwargs_keys;
  ffi::Array<ExprDoc> kwargs_values;
  for (ffi::String s : {"shape", "dtype"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      args.push_back(doc.value());
    }
  }
  for (ffi::String s : {"data", "strides", "elem_offset", "scope", "align", "offset_factor",
                        "layout", "allocated_addr"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tirx::BufferVar& buffer, const ffi::String& method,
                   const ffi::Array<ExprDoc>& args, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d, BufferVarDefinition var_definitions,
                   ffi::Optional<Expr> data) {
  auto prefix = (method == "sblock_alloc_buffer" || method == "match_buffer") ? STIR(d, method)
                                                                              : TIR(d, method);
  auto attrs = BufferAttrs(buffer.var()->ty.as_or_throw<tirx::BufferType>(), p->Attr("ty"), frame,
                           d, var_definitions, data, buffer);
  return BufferCall(prefix, attrs, args);
}

namespace {

/*!
 * \brief Check if a layout is the default layout for a given shape.
 */
bool IsDefaultLayout(const ffi::Optional<tirx::Layout>& layout, const ffi::Array<PrimExpr>& shape) {
  if (!layout.has_value()) return false;
  return StructuralEqual()(layout.value(), tirx::TileLayoutNode::DefaultLayout(shape));
}

/*!
 * \brief Try to produce a DeclBuffer sugar expression for the given child buffer
 *        with respect to a specific parent buffer.
 *
 * Returns std::nullopt if no sugar pattern matches.
 */
ffi::Optional<ExprDoc> TryDeclBufferSugarWithParent(const tirx::BufferType& child,
                                                    const AccessPath& p, const IRDocsifier& d,
                                                    const tirx::BufferVar& parent,
                                                    bool require_same_layout) {
  ffi::Optional<ExprDoc> parent_doc = d->GetVarDoc(parent);
  if (!parent_doc.has_value()) return std::nullopt;
  ExprDoc pdoc = parent_doc.value();

  prim::ExprDeepEqual expr_equal;

  // Check elem_offset equality
  bool same_elem_offset = expr_equal(child->elem_offset, parent->elem_offset);
  // Check dtype equality
  bool same_dtype = (child->dtype == parent->dtype);
  // Check shape equality
  bool same_shape = (child->shape.size() == parent->shape.size());
  if (same_shape) {
    for (size_t i = 0; i < child->shape.size(); ++i) {
      if (!expr_equal(child->shape[i], parent->shape[i])) {
        same_shape = false;
        break;
      }
    }
  }
  bool same_strides = (child->strides.size() == parent->strides.size());
  if (same_strides) {
    for (size_t i = 0; i < child->strides.size(); ++i) {
      if (!expr_equal(child->strides[i], parent->strides[i])) {
        same_strides = false;
        break;
      }
    }
  }

  bool child_is_default = IsDefaultLayout(child->layout, child->shape);
  bool parent_is_default = IsDefaultLayout(parent->layout, parent->shape);

  // NOTE: an earlier sugar printed rank-preserving aliases with a different
  // elem_offset as ``parent[slices]``. That print is not roundtrippable: it
  // reparses as a TensorRegion, not a Buffer, so any later Buffer use of the
  // alias (stores, views) breaks. Such aliases now print as plain
  // T.decl_buffer, which reparses exactly.

  // Differences in these Buffer fields cannot be expressed by the alias sugar
  // below, so conservatively fall back to T.decl_buffer.
  // Shape, strides, elem_offset, dtype, and layout are checked by each helper
  // because those are the fields that individual transformations may change.
  //
  // The explicit data projection was checked by TryDeclBufferSugar.  name/span
  // do not participate in structural equality, and BufferTypeNode has no
  // axis-separators field (unlike tir::Buffer).
  bool same_common_metadata = child->storage_scope == parent.scope() &&
                              child->data_alignment == parent->data_alignment &&
                              child->offset_factor == parent->offset_factor &&
                              StructuralEqual()(child->allocated_addr, parent->allocated_addr);
  if (!same_common_metadata) return std::nullopt;

  // --- (b) Local: parent has thread axes and child spans its physical storage ---
  if (same_elem_offset && same_dtype && same_strides && !parent_is_default &&
      parent->layout.has_value() && child->layout.has_value()) {
    if (auto* parent_tile = parent->layout.value().as<tirx::TileLayoutNode>()) {
      if (parent_tile->HasThreadAxis()) {
        // Compute the raw physical storage span after filtering thread axes.
        std::vector<tirx::Iter> storage_shard;
        std::vector<tirx::Iter> storage_replica;
        ffi::Map<tirx::Axis, PrimExpr> storage_offset;
        for (const auto& iter : parent_tile->shard) {
          if (!iter->axis->IsThreadAxis()) {
            storage_shard.push_back(iter);
          }
        }
        for (const auto& iter : parent_tile->replica) {
          if (!iter->axis->IsThreadAxis()) {
            storage_replica.push_back(iter);
          }
        }
        for (const auto& [axis, off] : parent_tile->offset) {
          if (!axis->IsThreadAxis()) {
            storage_offset.Set(axis, off);
          }
        }
        tirx::TileLayout expected_storage(
            ffi::Array<tirx::Iter>(storage_shard.begin(), storage_shard.end()),
            ffi::Array<tirx::Iter>(storage_replica.begin(), storage_replica.end()), storage_offset);

        PrimExpr storage_span = expected_storage->GetSpan(ffi::Optional<ffi::String>());
        PrimExpr storage_size = expected_storage->GetSize(ffi::Optional<ffi::String>());
        PrimExpr child_total = IntImm::Int32(1);
        for (const PrimExpr& dim : child->shape) {
          child_total = child_total * dim;
        }
        sym::Analyzer analyzer;
        bool default_physical =
            child_is_default && analyzer->CanProveEqual(child_total, storage_span);
        bool child_has_thread_axis = false;
        if (const auto* child_tile = child->layout.value().as<tirx::TileLayoutNode>()) {
          child_has_thread_axis = child_tile->HasThreadAxis();
        }
        bool explicit_override = !default_physical && !child_has_thread_axis;
        if (default_physical || explicit_override) {
          PrimExpr expected_extent = default_physical ? storage_span : storage_size;
          bool auto_shape =
              child->shape.size() == 1 && analyzer->CanProveEqual(child->shape[0], expected_extent);
          ffi::Array<ExprDoc> args;
          if (!auto_shape) {
            for (size_t i = 0; i < child->shape.size(); ++i) {
              args.push_back(d->AsDoc<ExprDoc>(child->shape[i], p->Attr("shape")->ArrayItem(i)));
            }
          }
          ffi::Array<ffi::String> kwargs_keys;
          ffi::Array<ExprDoc> kwargs_values;
          if (explicit_override) {
            kwargs_keys.push_back("layout");
            kwargs_values.push_back(d->AsDoc<ExprDoc>(child->layout.value(), p->Attr("layout")));
          }
          return pdoc->Attr("local")->Call(args, kwargs_keys, kwargs_values);
        }
      }
    }
  }

  // --- (c) View(dtype): different dtype, same elem_offset ---
  if (same_elem_offset && !same_dtype && child->shape.size() == parent->shape.size()) {
    // Verify shape compatibility with dtype reinterpret cast
    int child_bits = child->dtype.bits();
    int parent_bits = parent->dtype.bits();
    bool shapes_compatible = true;
    // All dims except last must match
    for (size_t i = 0; i + 1 < child->shape.size(); ++i) {
      if (!expr_equal(child->shape[i], parent->shape[i])) {
        shapes_compatible = false;
        break;
      }
    }
    if (shapes_compatible && !child->shape.empty()) {
      auto* child_last = child->shape.back().as<IntImmNode>();
      auto* parent_last = parent->shape.back().as<IntImmNode>();
      if (child_last && parent_last) {
        if (child_bits > parent_bits) {
          // Cast up: child_last = parent_last / ratio
          int ratio = child_bits / parent_bits;
          shapes_compatible = (parent_last->value == child_last->value * ratio);
        } else {
          // Cast down: child_last = parent_last * ratio
          int ratio = parent_bits / child_bits;
          shapes_compatible = (child_last->value == parent_last->value * ratio);
        }
      } else {
        shapes_compatible = false;
      }
    }
    // Also verify the parent's layout is compatible with the pack/unpack operation
    if (shapes_compatible && parent->layout.has_value()) {
      if (auto* ptile = parent->layout.value().as<tirx::TileLayoutNode>()) {
        if (!ptile->shard.empty() && child_bits > parent_bits) {
          // Cast up requires pack: last shard iter must have stride=1
          // and extent divisible by ratio
          const auto& last_iter = ptile->shard.back();
          auto* last_stride = last_iter->stride.as<IntImmNode>();
          auto* last_extent = last_iter->extent.as<IntImmNode>();
          int ratio = child_bits / parent_bits;
          if (!last_stride || last_stride->value != 1 || !last_extent ||
              last_extent->value % ratio != 0) {
            shapes_compatible = false;
          }
        }
      }
    }
    if (shapes_compatible) {
      ExprDoc dtype_doc = LiteralDoc::Str(DType2Str(child->dtype->dtype), p->Attr("dtype"));
      return pdoc->Attr("view")->Call({dtype_doc});
    }
  }

  // --- (d) Permute: child shape is a permutation of parent shape, same elem_offset ---
  if (same_elem_offset && same_dtype && !same_shape &&
      child->shape.size() == parent->shape.size()) {
    // Try to find a permutation
    std::vector<int> perm(child->shape.size(), -1);
    std::vector<bool> used(parent->shape.size(), false);
    bool is_permutation = true;
    for (size_t i = 0; i < child->shape.size(); ++i) {
      bool found = false;
      for (size_t j = 0; j < parent->shape.size(); ++j) {
        if (!used[j] && expr_equal(child->shape[i], parent->shape[j])) {
          perm[i] = j;
          used[j] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        is_permutation = false;
        break;
      }
    }
    // Check it's not identity
    bool is_identity = is_permutation;
    if (is_permutation) {
      for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int>(i)) {
          is_identity = false;
          break;
        }
      }
    }
    if (is_permutation && !is_identity) {
      // Verify the layout matches permutation by comparing shard iters directly
      bool layout_matches = false;
      if (parent->layout.has_value() && child->layout.has_value()) {
        auto* parent_tile = parent->layout.value().as<tirx::TileLayoutNode>();
        auto* child_tile = child->layout.value().as<tirx::TileLayoutNode>();
        if (parent_tile && child_tile && parent_tile->shard.size() == child_tile->shard.size()) {
          StructuralEqual seq;
          layout_matches = true;
          for (size_t i = 0; i < perm.size(); ++i) {
            if (!seq(child_tile->shard[i], parent_tile->shard[perm[i]])) {
              layout_matches = false;
              break;
            }
          }
          // Also check replica and offset are unchanged
          if (layout_matches) {
            layout_matches = seq(child_tile->replica, parent_tile->replica) &&
                             seq(child_tile->offset, parent_tile->offset);
          }
        }
      }
      if (layout_matches) {
        ffi::Array<ExprDoc> args;
        for (int idx : perm) {
          args.push_back(LiteralDoc::Int(idx, p->Attr("shape")));
        }
        return pdoc->Attr("permute")->Call(args);
      }
    }
  }

  // --- (e) Partition: child has 2*parent_ndim dims with grid+tile strides ---
  if (same_elem_offset && same_dtype && !parent->shape.empty() &&
      child->shape.size() == 2 * parent->shape.size() && !child->strides.empty() &&
      child->strides.size() == 2 * parent->shape.size()) {
    size_t ndim = parent->shape.size();
    // Compute parent's row-major strides
    std::vector<int64_t> parent_rm_strides(ndim);
    int64_t stride = 1;
    bool all_const = true;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      parent_rm_strides[i] = stride;
      if (auto* s = parent->shape[i].as<IntImmNode>()) {
        auto product = (stride * s->value).as<int64_t>();
        if (!product.has_value()) {
          all_const = false;
          break;
        }
        stride = *product;
      } else {
        all_const = false;
        break;
      }
    }
    if (all_const) {
      bool is_partition = true;
      for (size_t i = 0; i < ndim; ++i) {
        auto* grid_dim = child->shape[i].as<IntImmNode>();
        auto* tile_dim = child->shape[ndim + i].as<IntImmNode>();
        auto* parent_dim = parent->shape[i].as<IntImmNode>();
        auto* grid_stride = child->strides[i].as<IntImmNode>();
        auto* tile_stride = child->strides[ndim + i].as<IntImmNode>();
        if (!grid_dim || !tile_dim || !parent_dim || !grid_stride || !tile_stride) {
          is_partition = false;
          break;
        }
        // grid × tile == parent dim
        if (grid_dim->value * tile_dim->value != parent_dim->value) {
          is_partition = false;
          break;
        }
        // inner strides match parent's row-major strides
        if (tile_stride->value != parent_rm_strides[i]) {
          is_partition = false;
          break;
        }
        // grid stride == tile_dim × inner stride
        if (grid_stride->value != tile_dim->value * tile_stride->value) {
          is_partition = false;
          break;
        }
      }
      if (is_partition) {
        ffi::Array<ExprDoc> tuple_elems;
        for (size_t i = 0; i < ndim; ++i) {
          tuple_elems.push_back(d->AsDoc<ExprDoc>(child->shape[i], p->Attr("shape")->ArrayItem(i)));
        }
        return pdoc->Attr("partition")->Call({}, {"num_tiles"}, {TupleDoc(tuple_elems)});
      }
    }
  }

  // --- (f) View(*shape, layout=L): different shape/layout, same dtype and elem_offset ---
  if (same_elem_offset && same_dtype && !same_shape) {
    // Buffer.view(...) copies the parent's strides onto the child (see
    // python/tvm/tirx/buffer.py:view). If parent has strides but child
    // doesn't (or vice versa), the sugar can't faithfully round-trip
    // through view — fall back to T.decl_buffer where strides is an
    // explicit kwarg.
    if (!same_strides) return std::nullopt;

    ffi::Array<ExprDoc> args;
    ffi::Array<ffi::String> kwargs_keys;
    ffi::Array<ExprDoc> kwargs_values;
    for (size_t i = 0; i < child->shape.size(); ++i) {
      args.push_back(d->AsDoc<ExprDoc>(child->shape[i], p->Attr("shape")->ArrayItem(i)));
    }
    // Check if layout differs
    bool same_layout = false;
    if (child->layout.has_value() && parent->layout.has_value()) {
      same_layout = StructuralEqual()(child->layout.value(), parent->layout.value());
    } else if (!child->layout.has_value() && !parent->layout.has_value()) {
      same_layout = true;
    }
    // First pass prefers a parent whose layout matches structurally, so the
    // sugar prints as a bare reshape instead of restating the layout.
    if (require_same_layout && !same_layout) return std::nullopt;
    // Default layouts are shape-specific objects, but a default-to-default
    // reshape is still represented by view(*shape) without an explicit layout.
    if (!same_layout && !(child_is_default && parent_is_default)) {
      // Buffer.view(..., layout=None) means "inherit the parent layout", so it
      // cannot reconstruct a layout-less child from a laid-out parent.
      if (!child->layout.has_value()) return std::nullopt;
      kwargs_keys.push_back("layout");
      kwargs_values.push_back(d->AsDoc<ExprDoc>(child->layout.value(), p->Attr("layout")));
    }
    return pdoc->Attr("view")->Call(args, kwargs_keys, kwargs_values);
  }

  return std::nullopt;
}

/*!
 * \brief Try to produce a DeclBuffer sugar expression, trying all parent buffer candidates.
 */
ffi::Optional<ExprDoc> TryDeclBufferSugar(const tirx::BufferType& child, const AccessPath& p,
                                          const ffi::Optional<Expr>& data, const IRDocsifier& d) {
  if (!data.has_value()) return std::nullopt;
  const auto* call = data.value().as<CallNode>();
  if (!call || !call->op.same_as(tirx::builtin::buffer_data()) || call->args.size() != 1) {
    return std::nullopt;
  }
  auto parent = call->args[0].as<tirx::BufferVar>();
  if (!parent.has_value() || !d->GetVarDoc(parent.value()).has_value()) return std::nullopt;
  if (auto sugar = TryDeclBufferSugarWithParent(child, p, d, parent.value(),
                                                /*require_same_layout=*/true)) {
    return sugar;
  }
  return TryDeclBufferSugarWithParent(child, p, d, parent.value(),
                                      /*require_same_layout=*/false);
}

}  // namespace

ffi::Optional<ExprDoc> BufferOperationCall(const Call& call, const AccessPath& p,
                                           const IRDocsifier& d) {
  bool is_alloc = call->op.same_as(tirx::builtin::alloc_buffer());
  auto buffer = call->ty.as_or_throw<tirx::BufferType>();
  auto type_p = p->Attr("ty");
  ffi::Optional<Expr> data;
  if (!is_alloc) data = call->args[0];
  // The surface builders emit DictAttrs for allocations and no attrs for declarations.
  if (call->attrs.defined() != is_alloc || !call->ty_args.empty()) return std::nullopt;
  if (call->attrs.defined() && !call->attrs.as<DictAttrsNode>()) return std::nullopt;
  auto annotations = call->attrs.defined() ? call->attrs.as_or_throw<DictAttrs>() : DictAttrs();
  int shape_index = is_alloc ? 0 : 1;
  auto shape = call->args[shape_index].as_or_throw<Tuple>();
  auto dtype = call->args[shape_index + 1].as_or_throw<DataTypeImm>()->value;
  auto scope = call->args[shape_index + 2].as_or_throw<StringImm>()->value;
  // Surface builders infer these fields from operands. Keep a raw Call if that
  // would alter its explicit result type or discard unsupported attributes.
  if (!StructuralEqual()(shape->fields, buffer->shape) || dtype != buffer->dtype->dtype ||
      scope != buffer->storage_scope || (!is_alloc && !annotations->dict.empty())) {
    return std::nullopt;
  }
  if (!is_alloc && scope == "tmem") {
    const auto* pointer = data.value().as<CallNode>();
    if (buffer->allocated_addr.size() != 1 || !pointer ||
        !pointer->op.same_as(tirx::builtin::reinterpret()) || pointer->args.size() != 1 ||
        !StructuralEqual()(pointer->args[0], buffer->allocated_addr[0])) {
      return std::nullopt;
    }
  }
  if (!is_alloc && d->cfg->syntax_sugar && annotations->dict.empty()) {
    if (auto sugar = TryDeclBufferSugar(buffer, type_p, data, d)) {
      sugar.value()->source_paths.push_back(p->Attr("args")->ArrayItem(0));
      return sugar.value();
    }
  }
  auto attrs = BufferAttrs(buffer, type_p, d->frames.back(), d, BufferVarDefinition::None);
  attrs.Set("shape",
            d->AsDoc<ExprDoc>(call->args[shape_index], p->Attr("args")->ArrayItem(shape_index)));
  attrs.Set("dtype", LiteralDoc::DataType(
                         dtype, p->Attr("args")->ArrayItem(shape_index + 1)->Attr("value")));
  attrs.Set("scope", d->AsDoc<ExprDoc>(call->args[shape_index + 2],
                                       p->Attr("args")->ArrayItem(shape_index + 2)));
  if (data.has_value() && scope != "tmem") {
    attrs.Set("data", d->AsDoc<ExprDoc>(data.value(), p->Attr("args")->ArrayItem(0)));
  }
  ExprDoc prefix = TIR(d, is_alloc ? "alloc_buffer" : "decl_buffer");
  if (buffer->IsScalar(is_alloc)) {
    auto dtype = attrs.at("dtype");
    auto scope = attrs.at("scope");
    auto elem_offset = d->AsDoc<ExprDoc>(buffer->elem_offset, type_p->Attr("elem_offset"));
    attrs = {{"dtype", dtype}, {"scope", scope}};
    if (is_alloc) {
      prefix = TIR(d, "alloc_scalar");
      if (buffer->storage_scope == "local" || buffer->storage_scope == "shared") {
        prefix = TIR(d, buffer->storage_scope == "local" ? "local_scalar" : "shared_scalar");
        attrs.erase("scope");
      }
    } else {
      prefix = TIR(d, "decl_scalar");
      attrs.Set("elem_offset", elem_offset);
      attrs.Set("data", d->AsDoc<ExprDoc>(data.value(), p->Attr("args")->ArrayItem(0)));
    }
  } else if (is_alloc && (buffer->storage_scope == "local" || buffer->storage_scope == "shared")) {
    prefix = TIR(d, buffer->storage_scope == "local" ? "alloc_local" : "alloc_shared");
    attrs.erase("scope");
  }
  if (!buffer->IsScalar(is_alloc)) {
    if (dtype == d->cfg->buffer_dtype) attrs.erase("dtype");
    if (scope == "global") attrs.erase("scope");
  }
  ExprDoc result = BufferCall(prefix, attrs, {});
  if (!annotations->dict.empty()) {
    auto call_doc = result.as_or_throw<CallDoc>();
    auto keys = call_doc->kwargs_keys;
    auto values = call_doc->kwargs_values;
    keys.push_back("annotations");
    values.push_back(d->AsDoc<ExprDoc>(annotations->dict, p->Attr("attrs")->Attr("dict")));
    result = CallDoc(call_doc->callee, call_doc->args, keys, values);
  }
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::BufferType>(
      "tirx", [](tirx::BufferType buffer, AccessPath p, IRDocsifier d) -> Doc {
        // Construct the type directly so symbolic fields refer to the current definitions.
        return IdDoc("tvm")
            ->Attr("ir")
            ->Attr("make_node")
            ->Call({LiteralDoc::Str("tirx.BufferType", p)},
                   {"dtype", "storage_scope", "shape", "strides", "elem_offset", "data_alignment",
                    "offset_factor", "layout", "allocated_addr"},
                   {IdDoc("tvm")
                        ->Attr("ir")
                        ->Attr("PrimType")
                        ->Call({LiteralDoc::DataType(buffer->dtype->dtype, p->Attr("dtype"))}),
                    d->AsDoc<ExprDoc>(buffer->storage_scope, p->Attr("storage_scope")),
                    d->AsDoc<ExprDoc>(buffer->shape, p->Attr("shape")),
                    d->AsDoc<ExprDoc>(buffer->strides, p->Attr("strides")),
                    d->AsDoc<ExprDoc>(buffer->elem_offset, p->Attr("elem_offset")),
                    LiteralDoc::Int(buffer->data_alignment, p->Attr("data_alignment")),
                    LiteralDoc::Int(buffer->offset_factor, p->Attr("offset_factor")),
                    d->AsDoc<ExprDoc>(buffer->layout, p->Attr("layout")),
                    d->AsDoc<ExprDoc>(buffer->allocated_addr, p->Attr("allocated_addr"))});
      });
}

ExprDoc BufferAttn(const tirx::BufferVar& buffer, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d) {
  ffi::Map<ffi::String, ExprDoc> attrs =
      BufferAttrs(buffer.var()->ty.as_or_throw<tirx::BufferType>(), p, frame, d,
                  BufferVarDefinition::MatchBuffer, std::nullopt, buffer);
  if (!attrs.count("dtype")) {
    attrs.Set("dtype", LiteralDoc::DataType(buffer->dtype->dtype, p->Attr("dtype")));
  }
  return BufferCall(TIR(d, "Buffer"), attrs, {});
}

ffi::Array<Doc> BufferIndices(const ffi::Array<PrimExpr>& indices, const AccessPath& p,
                              const IRDocsifier& d) {
  int n = indices.size();
  ffi::Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (const auto* ramp = indices[i].as<prim::RampNode>()) {
      if (const auto* stride = ramp->stride.as<IntImmNode>()) {
        AccessPath ramp_p = p->Attr("indices")->ArrayItem(i);
        AccessPath stride_p = ramp_p->Attr("stride");
        ExprDoc start = d->AsDoc<ExprDoc>(ramp->base,  //
                                          ramp_p->Attr("base"));
        ExprDoc stop = d->AsDoc<ExprDoc>(ramp->base + ramp->lanes * ramp->stride,  //
                                         ramp_p->Attr("lanes"));
        ffi::Optional<ExprDoc> step = std::nullopt;
        if (stride->value != 1) {
          step = d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride"));
        }
        indices_doc.push_back(SliceDoc(start, stop, step));
        continue;
      }
    }
    indices_doc.push_back(d->AsDoc<ExprDoc>(indices[i], p->Attr("indices")->ArrayItem(i)));
  }
  return indices_doc;
}

ffi::Array<Doc> BufferLoadIndices(const ffi::Array<PrimExpr>& indices, const AccessPath& p,
                                  const IRDocsifier& d) {
  ffi::Array<Doc> indices_doc;
  indices_doc.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices_doc.push_back(d->AsDoc<ExprDoc>(indices[i], p->Attr("indices")->ArrayItem(i)));
  }
  return indices_doc;
}

ffi::Array<Doc> BufferSlices(const ffi::Array<Range>& region, const AccessPath& p,
                             const IRDocsifier& d) {
  int n = region.size();
  ffi::Array<Doc> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    Range range = region[i];
    AccessPath range_p = p->ArrayItem(i);
    ExprDoc min = d->AsDoc<ExprDoc>(range->min, range_p->Attr("min"));
    if (tvm::prim::is_one(range->extent)) {
      indices.push_back(min);
    } else {
      ExprDoc max = d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent"));
      indices.push_back(SliceDoc(min, max, std::nullopt));
    }
  }
  return indices;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tvm::TensorRegion>(
      "", [](tvm::TensorRegion buffer_region, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->source, p->Attr("source"));
        return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::BufferStore>(  //
      "", [](tirx::BufferStore store, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
        ExprDoc value = d->AsDoc<ExprDoc>(store->value, p->Attr("value"));

        // special case for scalar buffers
        if (store->buffer.IsScalar(true) || store->buffer.IsScalar(false)) {
          // TVM_FFI_ICHECK(store->indices.size() == 1 && tvm::prim::is_zero(store->indices[0]))
          //     << "1-dim buffer with shape (1,) store with indices other than [0] is not "
          //        "supported";
          ffi::Optional<ExprDoc> doc = d->GetVarDoc(store->buffer);
          TVM_FFI_ICHECK(doc.has_value())
              << "buffer is not defined in the environment: " << store->buffer;
          return AssignDoc(doc.value(), value, std::nullopt);
        }

        return AssignDoc(
            /*lhs=*/buffer[BufferIndices(store->indices, p->Attr("indices"), d)],
            /*rhs=*/value, std::nullopt);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<TensorLoad>(  //
      "", [](TensorLoad load, AccessPath p, IRDocsifier d) -> Doc {
        tvm::tirx::BufferVar source = load->source.as_or_throw<tvm::tirx::BufferVar>();
        ExprDoc buffer = d->AsDoc<ExprDoc>(source, p->Attr("source"));

        // special case for scalar
        if (source.IsScalar(true) || source.IsScalar(false)) {
          // TVM_FFI_ICHECK(load->indices.size() == 1 && tvm::prim::is_zero(load->indices[0]))
          //     << "Scalar buffer load with indices other than [0] is not supported";
          ffi::Optional<ExprDoc> doc = d->GetVarDoc(source);
          TVM_FFI_ICHECK(doc.has_value())
              << "Scalar buffer is not defined in the environment: " << source;
          return doc.value();
        }

        return buffer[BufferLoadIndices(load->indices, p->Attr("indices"), d)];
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Axis>(
      "", [](tirx::Axis axis, AccessPath p, IRDocsifier d) -> Doc {
        return LiteralDoc::Str(axis.name(), p->Attr("name"));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::Iter>(
      "", [](tirx::Iter iter, AccessPath p, IRDocsifier d) -> Doc {
        return TIR(d, "Iter")->Call({d->AsDoc<ExprDoc>(iter->extent, p->Attr("extent")),
                                     d->AsDoc<ExprDoc>(iter->stride, p->Attr("stride")),
                                     d->AsDoc<ExprDoc>(iter->axis.name(), p->Attr("axis"))},
                                    {}, {});
      });
}

Doc PrintTileLayout(tirx::TileLayout layout, IRDocsifier d, AccessPath p) {
  using OpKind = OperationDocNode::Kind;

  // `value @ Axis.<name>`, but elide `@m` (the default memory axis).
  auto bind_axis = [&](ExprDoc value, const tirx::Axis& axis) -> ExprDoc {
    if (axis.name() == "m") return value;
    return OperationDoc(OpKind::kMatMul, {value, IdDoc("Axis")->Attr(axis.name())});
  };

  // Build `head[(e0, e1, ...) : (s0@a0, s1@a1, ...)]` (or 1D shorthand
  // `head[e : s@a]`) from a list of Iters.
  auto iters_to_index = [&](ExprDoc head, const ffi::Array<tirx::Iter>& iters) -> ExprDoc {
    ffi::Array<ExprDoc> extents;
    ffi::Array<ExprDoc> strides;
    for (const auto& iter : iters) {
      extents.push_back(d->AsDoc<ExprDoc>(iter->extent, p->Attr("extent")));
      ExprDoc s = d->AsDoc<ExprDoc>(iter->stride, p->Attr("stride"));
      strides.push_back(bind_axis(s, iter->axis));
    }
    ExprDoc start = (extents.size() == 1) ? extents[0] : ExprDoc(TupleDoc(extents));
    ExprDoc stop = (strides.size() == 1) ? strides[0] : ExprDoc(TupleDoc(strides));
    return IndexDoc(head, {SliceDoc(start, stop, std::nullopt)});
  };

  // Degenerate case: no shard / replica iters. Fall back to from_iters so the
  // offset (if any) still round-trips.
  if (layout->shard.size() == 0 && layout->replica.size() == 0) {
    ffi::Array<ffi::String> keys;
    ffi::Array<ExprDoc> values;
    if (layout->offset.size() > 0) {
      ffi::Array<ExprDoc> offset_keys, offset_values;
      for (const auto& [axis, off] : layout->offset) {
        offset_keys.push_back(LiteralDoc::Str(axis.name(), p->Attr("axis")));
        offset_values.push_back(d->AsDoc<ExprDoc>(off, p->Attr("offset")));
      }
      keys.push_back("offset");
      values.push_back(DictDoc(offset_keys, offset_values));
    }
    return TIR(d, "TileLayout")->Attr("from_iters")->Call({}, keys, values);
  }

  // Compose `Tx.S[..] [+ Tx.R[..]] [+ offset_expr]`.
  auto add_term = [&](ffi::Optional<ExprDoc>& acc, ExprDoc term) {
    if (acc) {
      acc = ExprDoc(OperationDoc(OpKind::kAdd, {acc.value(), term}));
    } else {
      acc = term;
    }
  };

  ffi::Optional<ExprDoc> spec;
  if (layout->shard.size() > 0) {
    add_term(spec, iters_to_index(TIR(d, "S"), layout->shard));
  }
  if (layout->replica.size() > 0) {
    add_term(spec, iters_to_index(TIR(d, "R"), layout->replica));
  }
  if (layout->offset.size() > 0) {
    // Sort by axis name so the printed text is deterministic across builds
    // (`ffi::Map` iteration order is implementation-defined).
    std::vector<std::pair<tirx::Axis, PrimExpr>> sorted_offset(layout->offset.begin(),
                                                               layout->offset.end());
    std::sort(sorted_offset.begin(), sorted_offset.end(),
              [](const auto& a, const auto& b) { return a.first.name() < b.first.name(); });

    // Build the offset as a single arithmetic expression first, then add it
    // to the spec in one `+`. Chaining `spec + term1 + term2` would re-enter
    // `_LayoutSpec.__add__` with the second term and overwrite the offset
    // (see `python/tvm/tirx/layout.py::_LayoutSpec.__add__`), silently
    // dropping all but the last axis term. Combining the terms first lets
    // `_OnAxis.__add__` / `_OffsetExpr.__add__` accumulate them correctly.
    ffi::Optional<ExprDoc> off_doc;
    for (const auto& [axis, off] : sorted_offset) {
      ExprDoc term = bind_axis(d->AsDoc<ExprDoc>(off, p->Attr("offset")), axis);
      if (off_doc) {
        off_doc = ExprDoc(OperationDoc(OpKind::kAdd, {off_doc.value(), term}));
      } else {
        off_doc = term;
      }
    }
    add_term(spec, off_doc.value());
  }

  return TIR(d, "TileLayout")->Call({spec.value()}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable()  //
      .set_dispatch<tirx::TileLayout>(
          "", [](tirx::TileLayout layout, AccessPath p, IRDocsifier d) -> Doc {
            return PrintTileLayout(layout, d, p);
          });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable()  //
      .set_dispatch<tirx::ComposeLayout>(
          "", [](tirx::ComposeLayout layout, AccessPath p, IRDocsifier d) -> Doc {
            auto per_element = LiteralDoc::Int(layout->per_element, p->Attr("per_element"));
            auto swizzle_len = LiteralDoc::Int(layout->swizzle_len, p->Attr("swizzle_len"));
            auto atom_len = LiteralDoc::Int(layout->atom_len, p->Attr("atom_len"));
            auto tile_doc = d->AsDoc<ExprDoc>(layout->tile_layout, p->Attr("tile_layout"));
            ffi::Array<ffi::String> kwargs_keys;
            ffi::Array<ExprDoc> kwargs_values;
            if (!layout->swizzle_inner) {
              kwargs_keys.push_back("swizzle_inner");
              kwargs_values.push_back(
                  LiteralDoc::Boolean(layout->swizzle_inner, p->Attr("swizzle_inner")));
            }
            return TIR(d, "ComposeLayout")
                ->Call({per_element, swizzle_len, atom_len, tile_doc}, kwargs_keys, kwargs_values);
          });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<s_tir::MatchBufferRegion>(
      "", [](s_tir::MatchBufferRegion stmt, AccessPath p, IRDocsifier d) -> Doc {
        Frame frame = d->frames.back();
        ExprDoc lhs = DefineVar(stmt->buffer.var(), frame, d);
        ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
        ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                 d->frames.back(), d, BufferVarDefinition::MatchBuffer);
        return AssignDoc(lhs, rhs, std::nullopt);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TVMScriptPrinter::Register<tvm::TensorRegionNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<TensorLoadNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::BufferStoreNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::BufferTypeNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::IterNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::TileLayoutNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<tirx::ComposeLayoutNode>(ReprPrintTIR);
  TVMScriptPrinter::Register<s_tir::MatchBufferRegionNode>(ReprPrintTIR);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
