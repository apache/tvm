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

#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ffi::Map<ffi::String, ExprDoc> BufferAttrs(tirx::Buffer buffer, const AccessPath& buffer_p,
                                           const Frame& frame, const IRDocsifier& d,
                                           BufferVarDefinition var_definitions) {
  using tvm::tirx::Var;
  using tvm::tirx::VarNode;
  ffi::Map<ffi::String, ExprDoc> kwargs;
  ffi::Array<ExprDoc> var_def_lhs;
  ffi::Array<ExprDoc> var_def_rhs;

  // Step 0. Set up statistics
  std::unordered_map<const ffi::Object*, int> use_count;
  auto update_use_count = [&](const Expr& e) {
    tirx::PostOrderVisit(e, [&](const ffi::ObjectRef& n) {
      if (const VarNode* var = n.as<VarNode>()) {
        ++use_count[var];
      }
    });
  };
  update_use_count(buffer->elem_offset);
  update_use_count(buffer->data);
  for (const PrimExpr& e : buffer->strides) {
    update_use_count(e);
  }
  for (const PrimExpr& e : buffer->shape) {
    update_use_count(e);
  }
  auto is_new_var = [&](const Expr& e) {
    return e->IsInstance<VarNode>() && !d->IsVarDefined(e);
  };
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
      if (is_new_var(e)) {
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
  bool is_tmem_scope = false;
  if (auto* ptr_type = buffer->data->ty.as<PointerTypeNode>()) {
    is_tmem_scope = (ptr_type->storage_scope == "tmem");
  }
  bool is_inline_data = false;
  if (!is_tmem_scope) {
    if (is_new_var(buffer->data)) {
      if (var_definitions >= BufferVarDefinition::DataPointer) {
        is_inline_data = try_inline_def(buffer->data, buffer_p->Attr("data"), [=]() {
          return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("data");
        });
      } else {
        add_out_of_line_var_def(buffer->data, buffer_p->Attr("data"));
      }
    }
    if (!is_inline_data) {
      kwargs.Set("data", d->AsDoc<ExprDoc>(buffer->data, buffer_p->Attr("data")));
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
        if (try_inline_def(e, e_p, [=]() {
              return d->AsDoc<ExprDoc>(buffer, buffer_p)
                  ->Attr("strides")[{LiteralDoc::Int(i, std::nullopt)}];
            })) {
          results.push_back(LiteralDoc::Str(e.as_or_throw<Var>()->name_hint, e_p));
          continue;
        }
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
                 d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                   buffer_p->Attr("elem_offset")));
    }
  } else if (is_new_var(buffer->elem_offset)) {
    try_inline_def(buffer->elem_offset, buffer_p->Attr("elem_offset"),
                   [=]() { return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("elem_offset"); });
    needs_print_factor = true;
  } else {
    kwargs.Set("elem_offset",
               d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                 buffer_p->Attr("elem_offset")));
  }
  // Step 6. Handle `buffer.scope`
  {
    ffi::String scope = buffer.scope();
    if (scope != "global") {
      kwargs.Set(
          "scope",
          LiteralDoc::Str(scope,
                          buffer_p->Attr("data")->Attr("ty")->Attr("storage_scope")));
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
  // Step 9. Handle `buffer.buffer_type`
  if (buffer->buffer_type != tirx::BufferType::kDefault) {
    kwargs.Set("buffer_type", LiteralDoc::Str("auto", buffer_p->Attr("buffer_type")));
  }
  // Step 10. Handle `buffer.axis_separator`
  if (!buffer->axis_separators.empty()) {
    kwargs.Set("axis_separators",
               d->AsDoc<ExprDoc>(buffer->axis_separators, buffer_p->Attr("axis_separators")));
  }
  // Step 12. Handle `buffer.layout`. Track the enclosing PrimFunc's `s_tir`
  // attr — in `s_tir=True` mode the parser fills `layout=None` by default,
  // in `s_tir=False` (tirx) mode it fills `DefaultLayout(shape)`. Mirror
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
  if (buffer->layout.defined()) {
    bool is_default =
        ffi::StructuralEqual()(buffer->layout, tirx::TileLayoutNode::DefaultLayout(buffer->shape));
    if (!is_default) {
      kwargs.Set("layout", d->AsDoc<ExprDoc>(buffer->layout, buffer_p->Attr("layout")));
    }
  } else if (!enclosing_s_tir) {
    kwargs.Set("layout", LiteralDoc::None(buffer_p->Attr("layout")));
  }
  // Step 13. Handle `buffer.allocated_addr`
  if (!buffer->allocated_addr.empty()) {
    if (buffer->allocated_addr.size() == 1) {
      // Unwrap single-element array: DeclBuffer expects Optional<PrimExpr>, not Array.
      // For BufferLoad from scalar buffers, we must explicitly print buf[idx] because
      // the scalar shorthand (which drops the index) produces just the variable name,
      // and the parser resolves that to a Buffer object rather than a PrimExpr value.
      PrimExpr addr = buffer->allocated_addr[0];
      AccessPath addr_p = buffer_p->Attr("allocated_addr")->ArrayItem(0);
      if (const auto* bl = addr.as<tirx::BufferLoadNode>()) {
        // Ensure the buffer variable is defined (may emit a T.Buffer(...) statement).
        d->AsDoc<ExprDoc>(bl->buffer, addr_p->Attr("buffer"));
        // Get the variable name bound to this buffer.
        ffi::Optional<ExprDoc> buf_var = d->GetVarDoc(bl->buffer);
        TVM_FFI_ICHECK(buf_var.has_value())
            << "Buffer in allocated_addr is not defined: " << bl->buffer;
        // Build var[indices] explicitly instead of going through the default BufferLoad
        // printer, which would use the scalar shorthand and drop the index.
        int n_idx = bl->indices.size();
        ffi::Array<Doc> idx_docs;
        idx_docs.reserve(n_idx);
        for (int i = 0; i < n_idx; ++i) {
          idx_docs.push_back(
              d->AsDoc<ExprDoc>(bl->indices[i], addr_p->Attr("indices")->ArrayItem(i)));
        }
        kwargs.Set("allocated_addr", buf_var.value()[idx_docs]);
      } else {
        kwargs.Set("allocated_addr", d->AsDoc<ExprDoc>(addr, addr_p));
      }
    } else {
      kwargs.Set("allocated_addr",
                 d->AsDoc<ExprDoc>(buffer->allocated_addr, buffer_p->Attr("allocated_addr")));
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
                        "buffer_type", "axis_separators", "layout", "allocated_addr"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tirx::Buffer& buffer, const ffi::String& method,
                   const ffi::Array<ExprDoc>& args, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d, BufferVarDefinition var_definitions) {
  auto prefix = TIR(d, method);
  auto attrs = BufferAttrs(buffer, p, frame, d, var_definitions);
  if (method == "alloc_buffer") {
    if (buffer.IsScalar()) {
      // The buffer can be allocated by the alloc_scalar function
      auto dtype = d->AsDoc<ExprDoc>(buffer->dtype, p->Attr("dtype"));
      if (buffer.scope() == "shared") {
        // shared_scalar
        prefix = TIR(d, "shared_scalar");
        attrs = ffi::Map<ffi::String, ExprDoc>({{"dtype", dtype}});
      } else if (buffer.scope() == "local") {
        // local_scalar
        prefix = TIR(d, "local_scalar");
        attrs = ffi::Map<ffi::String, ExprDoc>({{"dtype", dtype}});
      } else {
        // alloc_scalar
        prefix = TIR(d, "alloc_scalar");
        auto scope = d->AsDoc<ExprDoc>(buffer.scope(), p->Attr("scope"));
        attrs = ffi::Map<ffi::String, ExprDoc>({{"dtype", dtype}, {"scope", scope}});
      }
    } else {
      if (buffer.scope() == "shared") {
        // alloc_shared
        prefix = TIR(d, "alloc_shared");
        attrs.erase("scope");
      } else if (buffer.scope() == "local") {
        // alloc_local
        prefix = TIR(d, "alloc_local");
        attrs.erase("scope");
      }
    }
  } else if (method == "decl_buffer") {
    if (buffer.IsScalar(false)) {
      // decl_scalar
      prefix = TIR(d, "decl_scalar");
      auto dtype = d->AsDoc<ExprDoc>(buffer->dtype, p->Attr("dtype"));
      auto scope = d->AsDoc<ExprDoc>(buffer.scope(), p->Attr("scope"));
      auto elem_offset = d->AsDoc<ExprDoc>(buffer->elem_offset, p->Attr("elem_offset"));
      auto data = d->AsDoc<ExprDoc>(buffer->data, p->Attr("data"));
      attrs = ffi::Map<ffi::String, ExprDoc>(
          {{"dtype", dtype}, {"scope", scope}, {"elem_offset", elem_offset}, {"data", data}});
    }
  }
  return BufferCall(prefix, attrs, args);
}

ExprDoc BufferAttn(const tirx::Buffer& buffer, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d) {
  ffi::Map<ffi::String, ExprDoc> attrs =
      BufferAttrs(buffer, p, frame, d, BufferVarDefinition::DataPointer);
  ExprDoc shape = attrs.Get("shape").value();
  ExprDoc dtype =
      attrs.Get("dtype").value_or(LiteralDoc::DataType(buffer->dtype->dtype, p->Attr("dtype")));
  return TIR(d, "Buffer")->Call({shape, dtype}, {}, {});
}

ffi::Array<Doc> BufferIndices(const ffi::Array<PrimExpr>& indices, const AccessPath& p,
                              const IRDocsifier& d) {
  int n = indices.size();
  ffi::Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (const auto* ramp = indices[i].as<tirx::RampNode>()) {
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

ffi::Array<Doc> BufferSlices(const ffi::Array<Range>& region, const AccessPath& p,
                             const IRDocsifier& d) {
  int n = region.size();
  ffi::Array<Doc> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    Range range = region[i];
    AccessPath range_p = p->ArrayItem(i);
    ExprDoc min = d->AsDoc<ExprDoc>(range->min, range_p->Attr("min"));
    if (tirx::is_one(range->extent)) {
      indices.push_back(min);
    } else {
      ExprDoc max = d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent"));
      indices.push_back(SliceDoc(min, max, std::nullopt));
    }
  }
  return indices;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferRegion>(
        "", [](tirx::BufferRegion buffer_region, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferStore>(  //
        "", [](tirx::BufferStore store, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
          ExprDoc value = d->AsDoc<ExprDoc>(store->value, p->Attr("value"));

          // special case for scalar buffers
          if ((store->buffer.IsScalar(true) || store->buffer.IsScalar(false)) &&
              !store->predicate.defined()) {
            // TVM_FFI_ICHECK(store->indices.size() == 1 && tirx::is_zero(store->indices[0]))
            //     << "1-dim buffer with shape (1,) store with indices other than [0] is not "
            //        "supported";
            ffi::Optional<ExprDoc> doc = d->GetVarDoc(store->buffer);
            TVM_FFI_ICHECK(doc.has_value())
                << "buffer is not defined in the environment: " << store->buffer;
            return AssignDoc(doc.value(), value, std::nullopt);
          }

          // Use .vstore(...) syntax when there is a predicate
          if (store->predicate.defined()) {
            ExprDoc indices = d->AsDoc<ExprDoc>(store->indices, p->Attr("indices"));
            ExprDoc predicate = d->AsDoc<ExprDoc>(store->predicate, p->Attr("predicate"));
            return ExprStmtDoc(
                buffer->Attr("vstore")->Call({indices, value}, {"predicate"}, {predicate}));
          }

          return AssignDoc(
              /*lhs=*/buffer[BufferIndices(store->indices, p->Attr("indices"), d)],
              /*rhs=*/value, std::nullopt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferLoad>(  //
        "", [](tirx::BufferLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(load->buffer, p->Attr("buffer"));

          // special case for scalar
          if ((load->buffer.IsScalar(true) || load->buffer.IsScalar(false)) &&
              !load->predicate.defined()) {
            // TVM_FFI_ICHECK(load->indices.size() == 1 && tirx::is_zero(load->indices[0]))
            //     << "Scalar buffer load with indices other than [0] is not supported";
            ffi::Optional<ExprDoc> doc = d->GetVarDoc(load->buffer);
            TVM_FFI_ICHECK(doc.has_value())
                << "Scalar buffer is not defined in the environment: " << load->buffer;
            return doc.value();
          }

          // Use .vload(...) syntax when there is a predicate
          if (load->predicate.defined()) {
            ExprDoc indices = d->AsDoc<ExprDoc>(load->indices, p->Attr("indices"));
            ExprDoc predicate = d->AsDoc<ExprDoc>(load->predicate, p->Attr("predicate"));
            return buffer->Attr("vload")->Call({indices}, {"predicate"}, {predicate});
          }

          return buffer[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::Buffer>("", [](tirx::Buffer buffer, AccessPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (ffi::Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                   BufferVarDefinition::DataPointer);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
        }
      }
      if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        // special case for scalar buffer
        if (buffer.IsScalar()) {
          return doc.value()->Attr("buffer");
        }
        return doc.value();
      }
      TVM_FFI_THROW(IndexError) << "Buffer is not defined in the environment: " << buffer;
      TVM_FFI_UNREACHABLE();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Axis>("", [](tirx::Axis axis, AccessPath p, IRDocsifier d) -> Doc {
      return LiteralDoc::Str(axis->name, p->Attr("name"));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Iter>("", [](tirx::Iter iter, AccessPath p, IRDocsifier d) -> Doc {
      return TIR(d, "Iter")->Call({d->AsDoc<ExprDoc>(iter->extent, p->Attr("extent")),
                                   d->AsDoc<ExprDoc>(iter->stride, p->Attr("stride")),
                                   d->AsDoc<ExprDoc>(iter->axis->name, p->Attr("axis"))},
                                  {}, {});
    });

Doc PrintTileLayout(tirx::TileLayout layout, IRDocsifier d, AccessPath p) {
  using OpKind = OperationDocNode::Kind;

  // `value @ Axis.<name>`, but elide `@m` (the default memory axis).
  auto bind_axis = [&](ExprDoc value, const tirx::Axis& axis) -> ExprDoc {
    if (axis->name == "m") return value;
    return OperationDoc(OpKind::kMatMul, {value, IdDoc("Axis")->Attr(axis->name)});
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
        offset_keys.push_back(LiteralDoc::Str(axis->name, p->Attr("axis")));
        offset_values.push_back(d->AsDoc<ExprDoc>(off, p->Attr("offset")));
      }
      keys.push_back("offset");
      values.push_back(DictDoc(offset_keys, offset_values));
    }
    return TIRx(d, "TileLayout")->Attr("from_iters")->Call({}, keys, values);
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
    add_term(spec, iters_to_index(TIRx(d, "S"), layout->shard));
  }
  if (layout->replica.size() > 0) {
    add_term(spec, iters_to_index(TIRx(d, "R"), layout->replica));
  }
  if (layout->offset.size() > 0) {
    // Sort by axis name so the printed text is deterministic across builds
    // (`ffi::Map` iteration order is implementation-defined).
    std::vector<std::pair<tirx::Axis, PrimExpr>> sorted_offset(layout->offset.begin(),
                                                               layout->offset.end());
    std::sort(sorted_offset.begin(), sorted_offset.end(),
              [](const auto& a, const auto& b) { return a.first->name < b.first->name; });

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

  return TIRx(d, "TileLayout")->Call({spec.value()}, {}, {});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::TileLayout>("",
                                    [](tirx::TileLayout layout, AccessPath p, IRDocsifier d)
                                        -> Doc { return PrintTileLayout(layout, d, p); });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::ComposeLayout>(
        "", [](tirx::ComposeLayout layout, AccessPath p, IRDocsifier d) -> Doc {
          auto layoutA = d->AsDoc<ExprDoc>(layout->swizzle, p->Attr("swizzle"));
          auto layoutB = d->AsDoc<ExprDoc>(layout->tile_layout, p->Attr("tile_layout"));
          return TIRx(d, "ComposeLayout")->Call({layoutA, layoutB}, {}, {});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::SwizzleLayout>(
        "", [](tirx::SwizzleLayout layout, AccessPath p, IRDocsifier d) -> Doc {
          return TIRx(d, "SwizzleLayout")
              ->Call(
                  {
                      LiteralDoc::Int(layout->per_element, p->Attr("per_element")),
                      LiteralDoc::Int(layout->swizzle_len, p->Attr("swizzle_len")),
                      LiteralDoc::Int(layout->atom_len, p->Attr("atom_len")),
                  },
                  {"swizzle_inner"},
                  {LiteralDoc::Boolean(layout->swizzle_inner, p->Attr("swizzle_inner"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::MatchBufferRegion>(
        "", [](tirx::MatchBufferRegion stmt, AccessPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                   d->frames.back(), d, BufferVarDefinition::MatchBuffer);
          return AssignDoc(lhs, rhs, std::nullopt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::ProducerLoad>(  //
        "", [](tirx::ProducerLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(load->producer->GetNameHint());
          return prefix[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_SCRIPT_REPR(tirx::BufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferLoadNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferStoreNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::IterNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::TileLayoutNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ComposeLayoutNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SwizzleLayoutNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::MatchBufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ProducerLoadNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
