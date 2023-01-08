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
#include <tvm/runtime/device_api.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Map<String, ExprDoc> BufferAttrs(const tir::Buffer& buffer, const ObjectPath& p, const Frame& frame,
                                 const IRDocsifier& d) {
  Map<String, ExprDoc> kwargs;
  auto implicit_var_def = [&](const PrimExpr& e, const ObjectPath& p, const String& key) {
    if (Optional<ExprDoc> doc = d->GetVarDoc(e)) {
      kwargs.Set(key, doc.value());
      return false;
    }
    if (e->IsInstance<tir::VarNode>()) {
      d->Define(e, frame, [=]() { return d->AsDoc<IdDoc>(buffer, p)->Attr(key); });
      return true;
    }
    kwargs.Set(key, d->AsDoc<ExprDoc>(e, p));
    return false;
  };
  auto array_out_line_var_def = [&](const Array<PrimExpr>& array, const ObjectPath& p,
                                    const String& key) {
    int n = array.size();
    Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr s = array[i];
      ObjectPath s_path = p->ArrayIndex(i);
      // Add out-of-line definition for a new Var in shape
      results.push_back(d->AsDoc<ExprDoc>(s, s_path));
    }
    kwargs.Set(key, TupleDoc(results));
  };
  // Step 1. Handle `buffer.shape`
  array_out_line_var_def(buffer->shape, p->Attr("shape"), "shape");
  // Step 2. Handle `buffer.dtype`
  if (buffer->dtype != Default::BufferDType()) {
    kwargs.Set("dtype", LiteralDoc::DataType(buffer->dtype));
  }
  // Step 3. Handle `buffer.data`
  implicit_var_def(buffer->data, p->Attr("data"), "data");
  // Step 4. Handle `buffer.strides`
  if (!buffer->strides.empty()) {
    array_out_line_var_def(buffer->strides, p->Attr("strides"), "strides");
  }
  // Step 5. Handle `buffer.elem_offset`
  bool needs_print_factor = false;
  if (const auto* int_imm = buffer->elem_offset.as<IntImmNode>()) {
    if (int_imm->value != 0) {
      kwargs.Set("elem_offset", d->AsDoc<ExprDoc>(buffer->elem_offset, p->Attr("elem_offset")));
    }
  } else {
    needs_print_factor =
        implicit_var_def(buffer->elem_offset, p->Attr("elem_offset"), "elem_offset");
  }
  // Step 6. Handle `buffer.scope`
  {
    String scope = buffer.scope();
    if (scope != "global") {
      kwargs.Set("scope", LiteralDoc::Str(scope));
    }
  }
  // Step 7. Handle `buffer.data_alignment`
  if (buffer->data_alignment != runtime::kAllocAlignment) {
    kwargs.Set("align", LiteralDoc::Int(buffer->data_alignment));
  }
  // Step 8. Handle `buffer.offset_factor`
  if (needs_print_factor || buffer->offset_factor != 1) {
    kwargs.Set("offset_factor", LiteralDoc::Int(buffer->offset_factor));
  }
  // Step 9. Handle `buffer.buffer_type`
  if (buffer->buffer_type != tir::BufferType::kDefault) {
    kwargs.Set("type", LiteralDoc::Str("auto"));
  }
  // Step 10. Handle `buffer.axis_separator`
  if (!buffer->axis_separators.empty()) {
    kwargs.Set("axis_separators",
               d->AsDoc<ExprDoc>(buffer->axis_separators, p->Attr("axis_separators")));
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
  for (String s : {"data", "strides", "elem_offset", "scope", "align", "offset_factor", "type",
                   "axis_separators"}) {
    if (Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tir::Buffer& buffer, const String& method, const Array<ExprDoc>& args,
                   const ObjectPath& p, const Frame& frame, const IRDocsifier& d) {
  return BufferCall(/*prefix=*/TIR(d)->Attr(method),
                    /*attrs=*/BufferAttrs(buffer, p, frame, d),
                    /*args=*/args);
}

Doc BufferIndex(const PrimExpr& index, const ObjectPath& p, const IRDocsifier& d) {
  if (const auto* ramp = index.as<tir::RampNode>()) {
    if (const auto* stride = ramp->stride.as<IntImmNode>()) {
      ExprDoc start = d->AsDoc<ExprDoc>(ramp->base, p->Attr("base"));
      ExprDoc stop = d->AsDoc<ExprDoc>(ramp->base + ramp->lanes * ramp->stride, p->Attr("lanes"));
      Optional<ExprDoc> step = NullOpt;
      if (stride->value != 1) {
        step = d->AsDoc<ExprDoc>(ramp->stride, p->Attr("stride"));
      }
      return SliceDoc(start, stop, step);
    }
  }
  return d->AsDoc<ExprDoc>(index, p);
}

ExprDoc BufferIndices(const tir::Buffer& buffer, const Array<PrimExpr>& indices,
                      const ObjectPath& p, const IRDocsifier& d) {
  int n = indices.size();
  Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    indices_doc.push_back(BufferIndex(indices[i], p->Attr("indices")->ArrayIndex(i), d));
  }
  return d->AsDoc<ExprDoc>(buffer, p->Attr("buffer"))[indices_doc];
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRegion>(
        "", [](tir::BufferRegion buffer_region, ObjectPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          p = p->Attr("region");
          Array<Range> region = buffer_region->region;
          int n = region.size();
          Array<Doc> indices;
          indices.reserve(n);
          for (int i = 0; i < n; ++i) {
            Range range = region[i];
            ExprDoc min = d->AsDoc<ExprDoc>(range->min, p->ArrayIndex(i)->Attr("min"));
            if (tir::is_one(range->extent)) {
              indices.push_back(min);
            } else {
              ExprDoc max =
                  d->AsDoc<ExprDoc>(range->min + range->extent, p->ArrayIndex(i)->Attr("extent"));
              indices.push_back(SliceDoc(min, max, NullOpt));
            }
          }
          return prefix[indices];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>(  //
        "", [](tir::BufferStore store, ObjectPath p, IRDocsifier d) -> Doc {
          return AssignDoc(/*lhs=*/BufferIndices(store->buffer, store->indices, p, d),
                           /*rhs=*/d->AsDoc<ExprDoc>(store->value, p->Attr("value")), NullOpt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferLoad>(  //
        "", [](tir::BufferLoad load, ObjectPath p, IRDocsifier d) -> Doc {
          return BufferIndices(load->buffer, load->indices, p, d);
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
