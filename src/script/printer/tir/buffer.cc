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

Map<String, ExprDoc> BufferAttrs(tir::Buffer buffer, const ObjectPath& buffer_p, const Frame& frame,
                                 const IRDocsifier& d, BufferVarDefinition var_definitions) {
  using tvm::tir::Var;
  using tvm::tir::VarNode;
  Map<String, ExprDoc> kwargs;
  Array<ExprDoc> var_def_lhs;
  Array<ExprDoc> var_def_rhs;

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
  auto add_out_of_line_var_def = [&](const Var& var, const ObjectPath& var_p) {
    ICHECK(!d->IsVarDefined(var));
    ExprDoc lhs = DefineVar(var, frame, d);
    lhs->source_paths.push_back(var_p);
    var_def_lhs.push_back(lhs);
    var_def_rhs.push_back(PrintVarCreation(var, var_p, d));
  };
  auto try_inline_def = [&](const PrimExpr& e, const ObjectPath& e_p,
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
    const Array<PrimExpr>& shape = buffer->shape;
    ObjectPath shape_p = buffer_p->Attr("shape");
    int n = shape.size();
    Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = shape[i];
      ObjectPath e_p = shape_p->ArrayIndex(i);
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
    const Array<PrimExpr>& strides = buffer->strides;
    ObjectPath strides_p = buffer_p->Attr("strides");
    int n = strides.size();
    Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = strides[i];
      ObjectPath e_p = strides_p->ArrayIndex(i);
      if (is_new_var(e)) {
        if (try_inline_def(e, e_p, [=]() {
              return d->AsDoc<ExprDoc>(buffer, buffer_p)
                  ->Attr("strides")[{LiteralDoc::Int(i, NullOpt)}];
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
    String scope = buffer.scope();
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
    frame->stmts.push_back(AssignDoc(var_def_lhs[0], var_def_rhs[0], NullOpt));
  } else if (var_def_lhs.size() > 1) {
    frame->stmts.push_back(AssignDoc(TupleDoc(var_def_lhs), TupleDoc(var_def_rhs), NullOpt));
  }
  return kwargs;
}

ExprDoc BufferCall(const ExprDoc& prefix, const Map<String, ExprDoc>& attrs, Array<ExprDoc> args) {
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  for (String s : {"shape", "dtype"}) {
    if (Optional<ExprDoc> doc = attrs.Get(s)) {
      args.push_back(doc.value());
    }
  }
  for (String s : {"data", "strides", "elem_offset", "scope", "align", "offset_factor",
                   "buffer_type", "axis_separators"}) {
    if (Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tir::Buffer& buffer, const String& method, const Array<ExprDoc>& args,
                   const ObjectPath& p, const Frame& frame, const IRDocsifier& d,
                   BufferVarDefinition var_definitions) {
  return BufferCall(/*prefix=*/TIR(d, method),
                    /*attrs=*/BufferAttrs(buffer, p, frame, d, var_definitions),
                    /*args=*/args);
}

ExprDoc BufferAttn(const tir::Buffer& buffer, const ObjectPath& p, const Frame& frame,
                   const IRDocsifier& d) {
  Map<String, ExprDoc> attrs = BufferAttrs(buffer, p, frame, d, BufferVarDefinition::DataPointer);
  ExprDoc shape = attrs.Get("shape").value();
  ExprDoc dtype =
      attrs.Get("dtype").value_or(LiteralDoc::DataType(buffer->dtype, p->Attr("dtype")));
  return TIR(d, "Buffer")->Call({shape, dtype}, {}, {});
}

Array<Doc> BufferIndices(const Array<PrimExpr>& indices, const ObjectPath& p,
                         const IRDocsifier& d) {
  int n = indices.size();
  Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (const auto* ramp = indices[i].as<tir::RampNode>()) {
      if (const auto* stride = ramp->stride.as<IntImmNode>()) {
        ObjectPath ramp_p = p->Attr("indices")->ArrayIndex(i);
        ObjectPath stride_p = ramp_p->Attr("stride");
        ExprDoc start = d->AsDoc<ExprDoc>(ramp->base,  //
                                          ramp_p->Attr("base"));
        ExprDoc stop = d->AsDoc<ExprDoc>(ramp->base + ramp->lanes * ramp->stride,  //
                                         ramp_p->Attr("lanes"));
        Optional<ExprDoc> step = NullOpt;
        if (stride->value != 1) {
          step = d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride"));
        }
        indices_doc.push_back(SliceDoc(start, stop, step));
        continue;
      }
    }
    indices_doc.push_back(d->AsDoc<ExprDoc>(indices[i], p->Attr("indices")->ArrayIndex(i)));
  }
  return indices_doc;
}

Array<Doc> BufferSlices(const Array<Range>& region, const ObjectPath& p, const IRDocsifier& d) {
  int n = region.size();
  Array<Doc> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    Range range = region[i];
    ObjectPath range_p = p->ArrayIndex(i);
    ExprDoc min = d->AsDoc<ExprDoc>(range->min, range_p->Attr("min"));
    if (tir::is_one(range->extent)) {
      indices.push_back(min);
    } else {
      ExprDoc max = d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent"));
      indices.push_back(SliceDoc(min, max, NullOpt));
    }
  }
  return indices;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRegion>(
        "", [](tir::BufferRegion buffer_region, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>(  //
        "", [](tir::BufferStore store, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
          return AssignDoc(/*lhs=*/buffer[BufferIndices(store->indices, p->Attr("indices"), d)],
                           /*rhs=*/d->AsDoc<ExprDoc>(store->value, p->Attr("value")), NullOpt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferLoad>(  //
        "", [](tir::BufferLoad load, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(load->buffer, p->Attr("buffer"));
          return buffer[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tir::Buffer>("", [](tir::Buffer buffer, ObjectPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                   BufferVarDefinition::DataPointer);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      if (Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      LOG(FATAL) << "IndexError: Buffer is not defined in the environment: " << buffer;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::MatchBufferRegion>(
        "", [](tir::MatchBufferRegion stmt, ObjectPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                   d->frames.back(), d, BufferVarDefinition::MatchBuffer);
          return AssignDoc(lhs, rhs, NullOpt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerLoad>(  //
        "", [](tir::ProducerLoad load, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(load->producer->GetNameHint());
          return prefix[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>(  //
        "", [](tir::ProducerStore store, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(store->producer->GetNameHint());
          prefix = prefix[BufferIndices(store->indices, p->Attr("indices"), d)];
          return AssignDoc(prefix, d->AsDoc<ExprDoc>(store->value, p->Attr("value")), NullOpt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>(  //
        "", [](tir::ProducerRealize stmt, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(stmt->producer->GetNameHint());
          prefix = prefix[BufferSlices(stmt->bounds, p->Attr("bounds"), d)];
          prefix = TIR(d, "ProducerRealize")
                       ->Call({prefix, d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"))});
          With<TIRFrame> f(d, stmt);
          AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
          return ScopeDoc(NullOpt, prefix, (*f)->stmts);
        });

TVM_SCRIPT_REPR(tir::BufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferLoadNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferStoreNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::MatchBufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ProducerLoadNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ProducerStoreNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ProducerRealizeNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
