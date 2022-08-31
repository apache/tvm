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
#include "./buffer.h"

#include <tvm/runtime/device_api.h>
#include <tvm/script/printer/visit_traced.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "../utils.h"
#include "./tir.h"
#include "tvm/runtime/data_type.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc BufferPrintInfo::AsCall(
    const ExprDoc& prefix, std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const {
  return AsCall(prefix, {}, converter);
}

ExprDoc BufferPrintInfo::AsCall(
    const ExprDoc& prefix, const Array<ExprDoc>& extra_args,
    std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const {
  Array<ExprDoc> args(extra_args);
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  {
    Array<ExprDoc> results;
    results.reserve(shape.size());
    for (TracedObject<PrimExpr> e : shape) {
      results.push_back(converter(e));
    }
    kwargs_keys.push_back("shape");
    kwargs_values.push_back(TupleDoc(results));
  }
  if (dtype.defined()) {
    args.push_back(dtype.value());
  }
  if (data.defined()) {
    kwargs_keys.push_back("data");
    kwargs_values.push_back(converter(data.value()));
  }
  if (strides.defined()) {
    Array<ExprDoc> results;
    results.reserve(strides.value().size());
    for (TracedObject<PrimExpr> stride : strides.value()) {
      results.push_back(converter(stride));
    }
    kwargs_keys.push_back("strides");
    kwargs_values.push_back(TupleDoc(results));
  }
  if (elem_offset.defined()) {
    kwargs_keys.push_back("elem_offset");
    kwargs_values.push_back(converter(elem_offset.value()));
  }
  if (scope.defined()) {
    kwargs_keys.push_back("scope");
    kwargs_values.push_back(scope.value());
  }
  if (align.defined()) {
    kwargs_keys.push_back("align");
    kwargs_values.push_back(align.value());
  }
  if (offset_factor.defined()) {
    kwargs_keys.push_back("offset_factor");
    kwargs_values.push_back(offset_factor.value());
  }
  if (buffer_type.defined()) {
    kwargs_keys.push_back("buffer_type");
    kwargs_values.push_back(buffer_type.value());
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

static Optional<ExprDoc> GetBufferScope(const TracedObject<tir::Buffer>& buffer) {
  auto data = buffer.GetAttr(&tir::BufferNode::data);
  auto type = data.GetAttr(&tir::VarNode::type_annotation).Downcast<PointerType>();
  auto scope = type.GetAttr(&PointerTypeNode::storage_scope);
  if (scope.Get().empty() || scope.Get() == "global") {
    return NullOpt;
  } else {
    return LiteralDoc::Str(scope);
  }
}

static Optional<ExprDoc> GetBufferDtype(const TracedObject<tir::Buffer>& buffer) {
  auto dtype = buffer.GetAttr(&tir::BufferNode::dtype);
  if (dtype.Get() == DataType::Float(32)) {
    return NullOpt;
  } else {
    return DType2Literal(dtype);
  }
}

static bool HasDefaultDataPtr(const tir::Buffer& buffer) {
  const auto* ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "Buffer variable is not of pointer type";
  if (const auto* element_type = ptr_type->element_type.as<PrimTypeNode>()) {
    DataType default_dtype = buffer->dtype;
    if (buffer->dtype.is_bool()) {
      default_dtype = DataType::Int(8);
    }
    return element_type->dtype == default_dtype;
  } else {
    return false;
  }
}

static TracedOptional<tir::Var> GetBufferData(const TracedObject<tir::Buffer>& buffer,
                                              const BufferAssociatedVariables& associated_vars) {
  auto data = buffer.GetAttr(&tir::BufferNode::data);
  if (associated_vars.IsAssociatedWith(data.Get(), buffer.Get()) &&
      HasDefaultDataPtr(buffer.Get())) {
    return TracedOptional<tir::Var>(NullOpt, data.GetPath());
  } else {
    return data;
  }
}

static TracedOptional<Array<PrimExpr>> GetBufferStrides(const TracedObject<tir::Buffer>& buffer) {
  auto strides = buffer.GetAttr(&tir::BufferNode::strides);
  if (!strides.empty()) {
    return TracedOptional<Array<PrimExpr>>(strides);
  } else {
    return TracedOptional<Array<PrimExpr>>(NullOpt, strides.GetPath());
  }
}

static TracedOptional<PrimExpr> GetBufferElemOffset(
    const TracedObject<tir::Buffer>& buffer, const BufferAssociatedVariables& associated_vars) {
  auto elem_offset = buffer.GetAttr(&tir::BufferNode::elem_offset);
  if (elem_offset.defined()) {
    // Don't print the offset if it is an associated variable
    if (elem_offset.IsInstance<tir::Var>() &&
        associated_vars.IsAssociatedWith(elem_offset.Get(), buffer.Get())) {
      return TracedOptional<PrimExpr>(NullOpt, elem_offset.GetPath());
    }

    // Don't print the offset if it is zero
    if (auto i = elem_offset.TryDowncast<IntImm>()) {
      if (i.value().Get()->value == 0 && i.value().Get()->dtype == DataType::Int(32)) {
        return TracedOptional<PrimExpr>(NullOpt, elem_offset.GetPath());
      }
    }
  }
  return elem_offset;
}

static Optional<ExprDoc> GetBufferAlignment(const TracedObject<tir::Buffer>& buffer) {
  auto data_alignment = buffer.GetAttr(&tir::BufferNode::data_alignment);
  if (data_alignment.Get() != runtime::kAllocAlignment) {
    return LiteralDoc::Int(data_alignment);
  } else {
    return NullOpt;
  }
}

static Optional<ExprDoc> GetBufferOffsetFactor(const TracedObject<tir::Buffer>& buffer) {
  auto offset_factor = buffer.GetAttr(&tir::BufferNode::offset_factor);
  if (offset_factor.Get() != 1) {
    return LiteralDoc::Int(offset_factor);
  } else {
    return NullOpt;
  }
}

static Optional<ExprDoc> GetBufferType(const TracedObject<tir::Buffer>& buffer) {
  auto buffer_type = buffer.GetAttr(&tir::BufferNode::buffer_type);
  if (buffer_type.Get() != tir::BufferType::kDefault) {
    return LiteralDoc::Str(MakeTraced(String("auto"), buffer_type.GetPath()));
  } else {
    return NullOpt;
  }
}

std::vector<BufferPrintInfo> GetBufferPrintInfo(
    const std::vector<TracedObject<tir::Buffer>>& buffers,  //
    std::function<bool(const tir::VarNode*)> f_var_defined,
    std::unordered_map<const tir::VarNode*, ObjectPath>* var_explicit_def,
    BufferAssociatedVariables* associated_vars) {
  using namespace tvm::tir;
  auto check_explicit_def = [&](const TracedObject<PrimExpr>& e) -> void {
    PostOrderVisitExprTraced(e, [&](const TracedObject<PrimExpr>& n) -> void {
      if (const auto* v = n.Get().as<VarNode>()) {
        if (!f_var_defined(v) && !associated_vars->IsAssociated(v)) {
          var_explicit_def->insert({v, n.GetPath()});
        }
      }
    });
  };
  for (const TracedObject<Buffer>& traced_buffer : buffers) {
    const Buffer& buffer = traced_buffer.Get();
    if (!f_var_defined(buffer->data.get()) && HasDefaultDataPtr(buffer)) {
      associated_vars->AssociateIfNotAlready(buffer->data.get(), buffer);
    }
    if (const auto* elem_offset = buffer->elem_offset.as<VarNode>()) {
      if (!f_var_defined(elem_offset)) {
        associated_vars->AssociateIfNotAlready(elem_offset, buffer);
      }
    }
  }
  for (TracedObject<Buffer> buffer : buffers) {
    auto shape = buffer.GetAttr(&tir::BufferNode::shape);
    std::for_each(shape.begin(), shape.end(), check_explicit_def);

    auto strides = buffer.GetAttr(&tir::BufferNode::strides);
    std::for_each(strides.begin(), strides.end(), check_explicit_def);

    check_explicit_def(buffer.GetAttr(&tir::BufferNode::data));
    check_explicit_def(buffer.GetAttr(&tir::BufferNode::elem_offset));
  }
  std::vector<BufferPrintInfo> results;
  for (TracedObject<Buffer> buffer : buffers) {
    results.push_back(
        BufferPrintInfo{/* .buffer = */ buffer,
                        /* .shape = */ buffer.GetAttr(&tir::BufferNode::shape),
                        /* .dtype = */ GetBufferDtype(buffer),
                        /* .data = */ GetBufferData(buffer, *associated_vars),
                        /* .strides = */ GetBufferStrides(buffer),
                        /* .elem_offset = */ GetBufferElemOffset(buffer, *associated_vars),
                        /* .scope = */ GetBufferScope(buffer),
                        /* .align = */ GetBufferAlignment(buffer),
                        /* .offset_factor = */ GetBufferOffsetFactor(buffer),
                        /* .buffer_type = */ GetBufferType(buffer)});
  }
  return results;
}

static TracedOptional<tir::Buffer> GetUsedBuffer(const TracedObject<ObjectRef>& stmt_or_expr) {
  if (auto load = stmt_or_expr.TryDowncast<tir::BufferLoad>()) {
    return load.value().GetAttr(&tir::BufferLoadNode::buffer);
  } else if (auto store = stmt_or_expr.TryDowncast<tir::BufferStore>()) {
    return store.value().GetAttr(&tir::BufferStoreNode::buffer);
  } else {
    return TracedOptional<tir::Buffer>(NullOpt, ObjectPath::Root());
  }
}

std::vector<TracedObject<tir::Buffer>> FindAliasingBuffers(tir::Var ptr_var,
                                                           TracedObject<tir::Stmt> body) {
  std::vector<TracedObject<tir::Buffer>> ret;
  PostOrderVisitStmtExprTraced(body, [&ret, ptr_var](const TracedObject<ObjectRef>& stmt_or_expr) {
    if (auto buffer_opt = GetUsedBuffer(stmt_or_expr)) {
      auto buffer = buffer_opt.value();
      if (buffer.Get()->data.same_as(ptr_var) &&
          std::find_if(ret.begin(), ret.end(),
                       [&](const auto& b) { return b.Get() == buffer.Get(); }) == ret.end()) {
        ret.push_back(buffer);
      }
    }
  });
  return ret;
}

TracedObject<String> GetBufferNameHint(const TracedObject<tir::Buffer>& buf) {
  TracedObject<String> name_hint = buf.GetAttr(&tir::BufferNode::name);
  if (name_hint.Get().empty()) {
    return MakeTraced(String("buf"), buf.GetPath());
  } else {
    return name_hint;
  }
}

std::vector<IdDoc> DefineBuffers(const std::vector<TracedObject<tir::Buffer>>& buffers,
                                 const Frame& frame, const IRDocsifier& p,
                                 const ExprDoc& definition_prefix,
                                 std::function<void(IdDoc, ExprDoc)> add_definiton) {
  std::vector<IdDoc> result;

  auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
    return p->vars->IsVarDefined(GetRef<tir::Var>(var));
  };
  std::unordered_map<const tir::VarNode*, ObjectPath> var_explicit_def;
  BufferAssociatedVariables associated_vars;

  std::vector<BufferPrintInfo> buffers_print_info =
      GetBufferPrintInfo(buffers, f_var_defined, &var_explicit_def, &associated_vars);

  for (const BufferPrintInfo& buffer_print_info : buffers_print_info) {
    TracedObject<tir::Buffer> buffer = buffer_print_info.buffer;
    TracedObject<String> name_hint = GetBufferNameHint(buffer);
    IdDoc buf_doc = p->vars->Define(buffer.Get(), name_hint, frame);
    result.push_back(buf_doc);
    ExprDoc buf_definition = buffer_print_info.AsCall(
        definition_prefix,
        [&p](const TracedObject<PrimExpr>& expr) -> ExprDoc { return p->AsDoc<ExprDoc>(expr); });
    add_definiton(buf_doc, buf_definition);
  }
  associated_vars.Define(p->vars.get(), frame);

  return result;
}

ExprDoc PrintBuffer(TracedObject<tir::Buffer> buf, IRDocsifier p) {
  Optional<ExprDoc> doc = p->vars->GetVarDoc(buf);
  if (doc.defined()) {
    return doc.value();
  } else {
    // TODO(yelite): When implementing the PrimFunc printing, the logic here
    // needs to change, putting variable def into PrimFuncFrame if it exists.
    TIRTopLevelFrame top_level_frame = p->GetFrame<TIRTopLevelFrame>().value();
    auto add_free_buffer_definition = [top_level_frame](IdDoc buf_indentifier,
                                                        ExprDoc buf_definition) {
      top_level_frame->free_var_definitions.push_back(
          AssignDoc(buf_indentifier, NullOpt, buf_definition));
    };
    return DefineBuffers({buf}, top_level_frame, p, TIR(p)->Attr("Buffer"),
                         add_free_buffer_definition)[0];
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Buffer>(PrintBuffer);

}  // namespace printer
}  // namespace script
}  // namespace tvm
