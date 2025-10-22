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

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ffi::Map<ffi::String, ExprDoc> BufferAttrs(tir::Buffer buffer, const AccessPath& buffer_p,
                                           const Frame& frame, const IRDocsifier& d,
                                           BufferVarDefinition var_definitions) {
  using tvm::tir::Var;
  using tvm::tir::VarNode;
  ffi::Map<ffi::String, ExprDoc> kwargs;
  ffi::Array<ExprDoc> var_def_lhs;
  ffi::Array<ExprDoc> var_def_rhs;

  // Step 0. Set up statistics
  std::unordered_map<const Object*, int> use_count;
  auto update_use_count = [&](const PrimExpr& e) {
    tir::PostOrderVisit(e, [&](const ObjectRef& n) {
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
  auto is_new_var = [&](const PrimExpr& e) {
    return e->IsInstance<VarNode>() && !d->IsVarDefined(e);
  };
  auto add_out_of_line_var_def = [&](const Var& var, const AccessPath& var_p) {
    ICHECK(!d->IsVarDefined(var));
    ExprDoc lhs = DefineVar(var, frame, d);
    lhs->source_paths.push_back(var_p);
    var_def_lhs.push_back(lhs);
    var_def_rhs.push_back(PrintVarCreation(var, var_p, d));
  };
  auto try_inline_def = [&](const PrimExpr& e, const AccessPath& e_p,
                            std::function<ExprDoc()> inline_f) {
    ICHECK(is_new_var(e));
    Var var = Downcast<Var>(e);
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
        add_out_of_line_var_def(Downcast<Var>(e), e_p);
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("shape", TupleDoc(results));
  }
  // Step 2. Handle `buffer.dtype`
  if (buffer->dtype != d->cfg->buffer_dtype) {
    kwargs.Set("dtype", LiteralDoc::DataType(buffer->dtype, buffer_p->Attr("dtype")));
  }
  // Step 3. Handle `buffer.data`
  bool is_inline_data = false;
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
          results.push_back(LiteralDoc::Str(Downcast<Var>(e)->name_hint, e_p));
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
    if (int_imm->value != 0) {
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
                          buffer_p->Attr("data")->Attr("type_annotation")->Attr("storage_scope")));
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
  if (buffer->buffer_type != tir::BufferType::kDefault) {
    kwargs.Set("buffer_type", LiteralDoc::Str("auto", buffer_p->Attr("buffer_type")));
  }
  // Step 10. Handle `buffer.axis_separator`
  if (!buffer->axis_separators.empty()) {
    kwargs.Set("axis_separators",
               d->AsDoc<ExprDoc>(buffer->axis_separators, buffer_p->Attr("axis_separators")));
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
                        "buffer_type", "axis_separators"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tir::Buffer& buffer, const ffi::String& method,
                   const ffi::Array<ExprDoc>& args, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d, BufferVarDefinition var_definitions) {
  return BufferCall(/*prefix=*/TIR(d, method),
                    /*attrs=*/BufferAttrs(buffer, p, frame, d, var_definitions),
                    /*args=*/args);
}

ExprDoc BufferAttn(const tir::Buffer& buffer, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d) {
  ffi::Map<ffi::String, ExprDoc> attrs =
      BufferAttrs(buffer, p, frame, d, BufferVarDefinition::DataPointer);
  ExprDoc shape = attrs.Get("shape").value();
  ExprDoc dtype =
      attrs.Get("dtype").value_or(LiteralDoc::DataType(buffer->dtype, p->Attr("dtype")));
  return TIR(d, "Buffer")->Call({shape, dtype}, {}, {});
}

ffi::Array<Doc> BufferIndices(const ffi::Array<PrimExpr>& indices, const AccessPath& p,
                              const IRDocsifier& d) {
  int n = indices.size();
  ffi::Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (const auto* ramp = indices[i].as<tir::RampNode>()) {
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
    if (tir::is_one(range->extent)) {
      indices.push_back(min);
    } else {
      ExprDoc max = d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent"));
      indices.push_back(SliceDoc(min, max, std::nullopt));
    }
  }
  return indices;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRegion>(
        "", [](tir::BufferRegion buffer_region, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>(  //
        "", [](tir::BufferStore store, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
          ExprDoc value = d->AsDoc<ExprDoc>(store->value, p->Attr("value"));

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
    .set_dispatch<tir::BufferLoad>(  //
        "", [](tir::BufferLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(load->buffer, p->Attr("buffer"));

          // Use .vload(...) syntax when there is a predicate
          if (load->predicate.defined()) {
            ExprDoc indices = d->AsDoc<ExprDoc>(load->indices, p->Attr("indices"));
            ExprDoc predicate = d->AsDoc<ExprDoc>(load->predicate, p->Attr("predicate"));
            return buffer->Attr("vload")->Call({indices}, {"predicate"}, {predicate});
          }

          return buffer[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tir::Buffer>("", [](tir::Buffer buffer, AccessPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (ffi::Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                   BufferVarDefinition::DataPointer);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
        }
      }
      if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      LOG(FATAL) << "IndexError: Buffer is not defined in the environment: " << buffer;
      TVM_FFI_UNREACHABLE();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::MatchBufferRegion>(
        "", [](tir::MatchBufferRegion stmt, AccessPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                   d->frames.back(), d, BufferVarDefinition::MatchBuffer);
          return AssignDoc(lhs, rhs, std::nullopt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerLoad>(  //
        "", [](tir::ProducerLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(load->producer->GetNameHint());
          return prefix[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_SCRIPT_REPR(tir::BufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferLoadNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferStoreNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::MatchBufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ProducerLoadNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
