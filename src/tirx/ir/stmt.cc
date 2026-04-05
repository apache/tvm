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
 * \file tvm/tirx/stmt.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/ir/traits.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <unordered_set>
#include <vector>

#include "buffer_common.h"
#include "script_print_utils.h"

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  StmtNode::RegisterReflection();
  BindNode::RegisterReflection();
  AttrStmtNode::RegisterReflection();
  AssertStmtNode::RegisterReflection();
  BufferStoreNode::RegisterReflection();
  DeclBufferNode::RegisterReflection();
  AllocBufferNode::RegisterReflection();
  SeqStmtNode::RegisterReflection();
  EvaluateNode::RegisterReflection();
  IfThenElseNode::RegisterReflection();
  ForNode::RegisterReflection();
  WhileNode::RegisterReflection();
  BufferRegionNode::RegisterReflection();
  MatchBufferRegionNode::RegisterReflection();
  SBlockNode::RegisterReflection();
  SBlockRealizeNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx._structured_msg", [](AssertStmt node) -> ffi::Array<ObjectRef> {
        ffi::Array<ObjectRef> parts_arr;
        for (const StringImm& part : node->message_parts) {
          parts_arr.push_back(part);
        }
        return {node->error_kind, parts_arr};
      })
      .def("tirx._evaluate_is_return", [](Evaluate self) -> bool {
        if (auto* call = self->value.as<tirx::CallNode>()) {
          return call->op.same_as(tirx::builtin::ret()) && call->args.size() == 1;
        }
        return false;
      })
      .def("tirx._evaluate_expr", [](Evaluate self) -> PrimExpr {
        if (auto* call = self->value.as<tirx::CallNode>()) {
          if (call->op.same_as(tirx::builtin::ret()) && call->args.size() == 1) {
            return call->args[0];
          }
        }
        return self->value;
      })
      .def("tirx._evaluate_kind", [](Evaluate self) -> ffi::Optional<ffi::String> {
        // For ret() calls, the return check handles it, so kind doesn't matter
        if (auto* call = self->value.as<tirx::CallNode>()) {
          if (call->op.same_as(tirx::builtin::ret())) return {};
        }
        // For other calls, no wrapper needed
        if (self->value->IsInstance<tirx::CallNode>()) return {};
        // Non-call: wrap with T.evaluate
        return ffi::String("T.evaluate");
      });
}

// Bind
Bind::Bind(Var var, PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  auto vdtype = value.dtype();
  // It is still valid to bind a pointer type var to a value that is of type handle.
  if (var->type_annotation.as<PointerTypeNode>()) {
    TVM_FFI_ICHECK(vdtype.is_handle());
  } else {
    TVM_FFI_ICHECK_EQ(value.dtype(), var.dtype());
  }

  ObjectPtr<BindNode> node = ffi::make_object<BindNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Bind",
                        [](Var var, PrimExpr value, Span span) { return Bind(var, value, span); });
}

// AttrStmt
AttrStmt::AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
  auto n = ffi::make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AttrStmt",
                        [](Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
                          // when node is a POD data type like int or bool, first convert to
                          // primexpr.
                          if (node.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
                            return AttrStmt(node.cast<PrimExpr>(), attr_key, value, body, span);
                          }
                          return AttrStmt(node, attr_key, value, body, span);
                        });
}

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, StringImm error_kind,
                       ffi::Array<StringImm> message_parts, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(condition.dtype().is_predicate_dtype())
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition.dtype();
  TVM_FFI_ICHECK(error_kind.defined());

  ObjectPtr<AssertStmtNode> node = ffi::make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->error_kind = std::move(error_kind);
  node->message_parts = std::move(message_parts);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.AssertStmt", [](PrimExpr condition, StringImm error_kind,
                                              ffi::Array<StringImm> message_parts, Span span) {
    return AssertStmt(condition, error_kind, message_parts, span);
  });
}

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         ffi::Optional<IterVar> thread_binding, ffi::Map<ffi::String, Any> annotations,
         ffi::Optional<PrimExpr> step, Span span) {
  TVM_FFI_ICHECK(loop_var.defined());
  TVM_FFI_ICHECK(min.defined());
  TVM_FFI_ICHECK(extent.defined());
  TVM_FFI_ICHECK(body.defined());

  auto require_scalar_int_dtype = [&](PrimExpr expr, const char* field_name) {
    auto dtype = expr.dtype();
    TVM_FFI_ICHECK(dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()))
        << "TIR For nodes require a scalar integer as the " << field_name << ", but received "
        << expr << " with dtype " << dtype;
  };
  require_scalar_int_dtype(loop_var, "loop_var");
  require_scalar_int_dtype(min, "min");
  require_scalar_int_dtype(extent, "extent");

  // When extent, min or step is an IntImm but has narrower dtype than loop_var
  // we directly promote them without raising errors.
  auto try_promote_imm_dtype = [&](const PrimExpr& e) {
    TVM_FFI_ICHECK(e.dtype().bits() <= loop_var.dtype().bits())
        << " Loop variable's dtype (" << loop_var.dtype()
        << ") is narrower than that of `min` or `extent` (" << e.dtype() << ")";
    const IntImmNode* a = e.as<IntImmNode>();
    if (a && e.dtype().bits() < loop_var.dtype().bits()) {
      return make_const(loop_var.dtype(), a->value);
    } else {
      return e;
    }
  };

  min = try_promote_imm_dtype(min);
  extent = try_promote_imm_dtype(extent);

  TVM_FFI_ICHECK(loop_var.dtype() == min.dtype()) << loop_var.dtype() << " vs " << min.dtype();
  TVM_FFI_ICHECK(loop_var.dtype() == extent.dtype())
      << loop_var.dtype() << " vs " << extent.dtype();

  if (step.has_value()) {
    require_scalar_int_dtype(*step, "step");
    step = try_promote_imm_dtype(*step);
    TVM_FFI_ICHECK(loop_var.dtype() == (*step).dtype())
        << loop_var.dtype() << " vs " << (*step).dtype();
  }

  ObjectPtr<ForNode> node = ffi::make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->step = std::move(step);
  node->span = std::move(span);
  data_ = std::move(node);
}

bool ForNode::HasTrivialStep() const { return !step.has_value() || is_one(*step); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.For", [](Var loop_var, PrimExpr min, PrimExpr extent, int kind,
                                       Stmt body, ffi::Optional<IterVar> thread_binding,
                                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                                       ffi::Optional<PrimExpr> step, Span span) {
    return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
               annotations.value_or(ffi::Map<ffi::String, Any>()), step, span);
  });
}

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kSerial:
      out << "for";
      break;
    case ForKind::kParallel:
      out << "parallel";
      break;
    case ForKind::kUnrolled:
      out << "unrolled";
      break;
    case ForKind::kVectorized:
      out << "vectorized";
      break;
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

// While
While::While(PrimExpr condition, Stmt body, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(condition.dtype().is_scalar());
  TVM_FFI_ICHECK(body.defined());

  ObjectPtr<WhileNode> node = ffi::make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.While", [](PrimExpr condition, Stmt body, Span span) {
    return While(condition, body, span);
  });
}

// DeclBuffer
DeclBuffer::DeclBuffer(Buffer buffer, Span span) {
  ObjectPtr<DeclBufferNode> node = ffi::make_object<DeclBufferNode>();
  node->buffer = std::move(buffer);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.DeclBuffer",
                        [](Buffer buffer, Span span) { return DeclBuffer(buffer, span); });
}

// AllocBuffer
AllocBuffer::AllocBuffer(Buffer buffer, ffi::Map<ffi::String, Any> annotations, Span span) {
  ObjectPtr<AllocBufferNode> node = ffi::make_object<AllocBufferNode>();
  node->buffer = std::move(buffer);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.AllocBuffer",
      [](Buffer buffer, ffi::Optional<ffi::Map<ffi::String, Any>> annotations, Span span) {
        return AllocBuffer(buffer, annotations.value_or(ffi::Map<ffi::String, Any>()), span);
      });
}

// SeqStmt
SeqStmt::SeqStmt(ffi::Array<Stmt> seq, Span span) {
  bool requires_flattening = std::any_of(
      seq.begin(), seq.end(), [](const Stmt& stmt) { return stmt->IsInstance<SeqStmtNode>(); });

  if (requires_flattening) {
    auto flattened = SeqStmt::Flatten(seq);
    if (auto* ptr = flattened.as<SeqStmtNode>()) {
      seq = ptr->seq;
    } else {
      seq = {flattened};
    }
  }

  TVM_FFI_ICHECK_NE(seq.size(), 0) << "An empty SeqStmt is prohibited.  "
                                   << "To write a no-op, use Evaluate(0), "
                                   << "or the result of SeqStmt::Flatten()";
  TVM_FFI_ICHECK_NE(seq.size(), 1) << "A SeqStmt of length 1 is prohibited.  "
                                   << "Use the node " << seq[0] << "directly, "
                                   << "or for dynamic usage, normalize using SeqStmt::Flatten()";

  auto node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.SeqStmt", [](ffi::Array<Stmt> seq, Span span) {
    return SeqStmt(std::move(seq), span);
  });
}

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, ffi::Optional<Stmt> else_case,
                       Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = ffi::make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.IfThenElse",
                        [](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
                          return IfThenElse(condition, then_case, else_case, span);
                        });
}

// Evaluate
Evaluate::Evaluate(PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());

  ObjectPtr<EvaluateNode> node = ffi::make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Evaluate",
                        [](PrimExpr value, Span span) { return Evaluate(value, span); });
}

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                         ffi::Optional<PrimExpr> predicate, Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().dtype().is_scalable_vector();
  bool is_buffer_dtype_scalable = buffer->dtype.is_scalable_vector();
  bool is_value_dtype_scalable = value.dtype().is_scalable_vector();

  TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
      << "Index dtype and buffer dtype can't both be scalable.";

  if (predicate.defined()) {
    bool is_predicate_dtype_scalable = predicate.value().dtype().is_scalable_vector();
    TVM_FFI_ICHECK_EQ(is_value_dtype_scalable, is_predicate_dtype_scalable)
        << "Predicate mask dtype and value dtype must both be scalable.";
  }

  if (is_index_scalable || is_buffer_dtype_scalable) {
    TVM_FFI_ICHECK(is_value_dtype_scalable) << "Can't store non-scalable data into scalable buffer";
  }

  int index_lanes = indices.empty() ? 1 : indices.back().dtype().get_lanes_or_vscale_factor();
  int buffer_lanes = buffer->dtype.get_lanes_or_vscale_factor();
  int value_dtype_lanes = value.dtype().get_lanes_or_vscale_factor();

  TVM_FFI_ICHECK_EQ(index_lanes * buffer_lanes, value_dtype_lanes)
      << "Cannot store value with " << value_dtype_lanes << ", expected value with "
      << index_lanes * buffer_lanes << " (" << index_lanes << " index lanes * " << buffer_lanes
      << " buffer element lanes)";

  if (predicate.defined()) {
    DataType predicate_dtype = predicate.value().dtype();
    int predicate_dtype_lanes = predicate_dtype.get_lanes_or_vscale_factor();
    TVM_FFI_ICHECK_EQ(value_dtype_lanes, predicate_dtype_lanes)
        << "Got a predicate mask with " << predicate_dtype_lanes
        << " lanes, but trying to store a value with " << value_dtype_lanes
        << " lanes. The number of lanes must match.";

    DataType predicate_element_dtype = predicate_dtype.element_of();
    TVM_FFI_ICHECK(predicate_element_dtype.is_predicate_dtype())
        << "Predicate mask elements must be boolean values, but got " << predicate_element_dtype
        << ".";
  }

  runtime::DataType buffer_dtype;
  if (is_index_scalable || is_buffer_dtype_scalable) {
    buffer_dtype = buffer->dtype.with_scalable_vscale_factor(buffer_lanes * index_lanes);
  } else {
    buffer_dtype = buffer->dtype.with_lanes(buffer_lanes * index_lanes);
  }
  if (buffer_dtype != value.dtype()) {
    TVM_FFI_THROW(TypeError) << "dtype mismatch on BufferStore: "                 //
                             << "buffer's dtype is `" << buffer->dtype            //
                             << "`, the lanes of indexing are: `" << index_lanes  //
                             << "`, the scalability is: `" << buffer_dtype.is_scalable_vector()
                             << "`, but RHS's dtype is `" << value.dtype() << "`";
  }

  ObjectPtr<BufferStoreNode> node = ffi::make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.BufferStore",
                        [](Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                           ffi::Optional<PrimExpr> predicate, Span span) {
                          return BufferStore(buffer, value, indices, predicate, span);
                        });
}

// BufferRegion
PrimExpr BufferRegionNode::ToPrimExpr() const {
  // Auto convert to PrimExpr if it is a single point load
  ffi::Array<PrimExpr> indices;
  indices.reserve(this->region.size());
  for (const Range& r : this->region) {
    if (tvm::tirx::is_one(r->extent)) {
      indices.push_back(r->min);
    } else if (r->extent.as<IntImmNode>()) {
      indices.push_back(tirx::Ramp(r->min, tvm::tirx::make_const(r->min->dtype, 1), r->extent));
    } else {
      TVM_FFI_THROW(ValueError) << "Cannot convert to BufferLoad: "
                                << ffi::GetRef<BufferRegion>(this);
    }
  }
  return tirx::BufferLoad(this->buffer, indices);
}

BufferRegion::BufferRegion(Buffer buffer, ffi::Array<Range> region) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), region.size())
      << "The dimension between " << buffer << " and region " << region
      << " mismatched, the buffer is " << buffer;
  ObjectPtr<BufferRegionNode> node = ffi::make_object<BufferRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

BufferRegion BufferRegion::FullRegion(Buffer buffer) {
  ffi::Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

BufferRegion BufferRegion::FromPoint(Buffer buffer, ffi::Array<PrimExpr> indices) {
  ffi::Array<Range> region;
  for (const PrimExpr& index : indices) {
    if (const RampNode* ramp_index = index.as<RampNode>()) {
      region.push_back(
          Range::FromMinExtent(ramp_index->base, ramp_index->stride * ramp_index->lanes));
    } else {
      region.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
    }
  }
  return BufferRegion(buffer, region);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.BufferRegion", [](Buffer buffer, ffi::Array<Range> region) {
    return BufferRegion(buffer, region);
  });
}

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(Buffer buffer, BufferRegion source) {
  const Buffer& source_buffer = source->buffer;
  arith::Analyzer analyzer;
  // Check scope and dtype
  TVM_FFI_ICHECK_EQ(buffer.scope(), source_buffer.scope())
      << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << " vs. "
      << source_buffer.scope();
  TVM_FFI_ICHECK_EQ(buffer->dtype, source_buffer->dtype)
      << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << " vs. "
      << source_buffer->dtype;

  // Check data_alignment
  TVM_FFI_ICHECK(source_buffer->data_alignment % buffer->data_alignment == 0)
      << "Trying to match buffer to another one with lower alignment requirement "
      << " required alignment=" << buffer->data_alignment
      << ", provided alignment=" << source_buffer->data_alignment;

  // Check BufferType. AutoBroadcast is not allowed for now.
  TVM_FFI_ICHECK(buffer->buffer_type == BufferType::kDefault &&
                 source_buffer->buffer_type == BufferType::kDefault)
      << "AutoBroadcast is not allowed in MatchBuffer";

  // Validate shape
  TVM_FFI_ICHECK(source->region.size() >= buffer->shape.size())
      << "Dimension of source Region expected to be larger or equal than target buffer shape, but "
         "got "
      << source->region.size() << " vs. " << buffer->shape.size();
  size_t offset = source->region.size() - buffer->shape.size();
  for (size_t i = 0; i < offset; ++i) {
    TVM_FFI_ICHECK(analyzer.CanProve(source->region[i]->extent == 1))
        << "The higher dimension should be 1, but got " << source->region[i]->extent << ".";
  }
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const PrimExpr& buffer_shape = buffer->shape[i];
    if (!buffer_shape->IsInstance<VarNode>()) {
      TVM_FFI_ICHECK(analyzer.CanProve(source_range->extent == buffer_shape))
          << "The dimension mismatched between source region and target buffer shape, got "
          << source_range->extent << " vs. " << buffer_shape << ".";
    }
  }
  // Note that we do not check elem_offset and strides in this function

  // Construction
  ObjectPtr<MatchBufferRegionNode> node = ffi::make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.MatchBufferRegion", [](Buffer buffer, BufferRegion source) {
    return MatchBufferRegion(buffer, source);
  });
}

// Block
SBlock::SBlock(ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
               ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
               ffi::Optional<Stmt> init, ffi::Array<Buffer> alloc_buffers,
               ffi::Array<MatchBufferRegion> match_buffers, ffi::Map<ffi::String, Any> annotations,
               Span span) {
  ObjectPtr<SBlockNode> node = ffi::make_object<SBlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::move(init);
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = std::move(match_buffers);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.SBlock",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
                           ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
                           ffi::Optional<Stmt> init, ffi::Array<Buffer> alloc_buffers,
                           ffi::Array<MatchBufferRegion> match_buffers,
                           ffi::Map<ffi::String, Any> annotations, Span span) {
                          return SBlock(iter_vars, reads, writes, name_hint, body, init,
                                        alloc_buffers, match_buffers, annotations, span);
                        });
}

// BlockRealize
SBlockRealize::SBlockRealize(ffi::Array<PrimExpr> values, PrimExpr predicate, SBlock block,
                             Span span) {
  TVM_FFI_CHECK_EQ(block->iter_vars.size(), values.size(), ValueError)
      << "BlockRealize needs to have the same number of iter_vars and binding values";
  TVM_FFI_CHECK(predicate.dtype().is_bool() || predicate.dtype() == DataType::UInt(1), TypeError)
      << "Expect Block.predicate to be a bool expression";
  ObjectPtr<SBlockRealizeNode> node = ffi::make_object<SBlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.SBlockRealize", [](ffi::Array<PrimExpr> iter_values,
                                                 PrimExpr predicate, SBlock block, Span span) {
    return SBlockRealize(iter_values, predicate, block, span);
  });
}

PrimExpr TypeAnnotation(DataType dtype, Span span) {
  static auto op = Op::Get("tirx.type_annotation");
  return tirx::Call(dtype, op, {}, span);
}

TVM_TIR_REGISTER_OP("type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));


// ---------------------------------------------------------------------------
// __ffi_text_print__ overrides
// ---------------------------------------------------------------------------

// Static helper for SBlockRealize/SBlock printing
static ::tvm::ffi::ir::text::NodeAST PrintSBlockRealize(
    SBlockRealize realize, ::tvm::ffi::ir::text::IRPrinter printer,
    ::tvm::ffi::ir::text::AccessPath path) {
  using namespace printer;
  SBlock block = realize->block;
  text::AccessPath block_p = path->Attr("block");

  text::DefaultFrame frame;
  printer->FramePush(frame);

  // Build context expr: T.sblock("name")
  text::ExprAST ctx = text::ExprCall(TIR("sblock"), {text::LiteralAST::Str(block->name_hint)});

  // Define iter_vars and build as_var
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    IterVar iv = block->iter_vars[i];
    text::AccessPath iv_p = block_p->Attr("iter_vars")->ArrayItem(i);

    text::ExprAST var_id = DefineVar(iv->var, printer, iv_p->Attr("var"));

    std::string axis_type;
    switch (static_cast<int>(iv->iter_type)) {
      case kDataPar: axis_type = "spatial"; break;
      case kCommReduce: axis_type = "reduce"; break;
      case kOrdered: axis_type = "scan"; break;
      default: axis_type = "opaque"; break;
    }

    text::ExprAST dom(ffi::UnsafeInit{});
    if (iv->dom.defined()) {
      if (is_zero(iv->dom->min)) {
        dom = Print(printer, iv->dom->extent, iv_p->Attr("dom")->Attr("extent"));
      } else {
        dom = text::TupleAST({}, {Print(printer, iv->dom->min, iv_p->Attr("dom")->Attr("min")),
                            Print(printer, iv->dom->min + iv->dom->extent,
                                  iv_p->Attr("dom")->Attr("extent"))});
      }
    } else {
      dom = text::LiteralAST::Null();
    }

    text::ExprAST val(ffi::UnsafeInit{});
    if (i < static_cast<int>(realize->iter_values.size())) {
      val = Print(printer, realize->iter_values[i], path->Attr("iter_values")->ArrayItem(i));
    } else {
      val = text::LiteralAST::Null();
    }

    text::ExprAST rhs = text::ExprCall(text::ExprAttr(text::ExprAttr(text::IdAST("T"), "axis"), axis_type), {dom, val});
    frame->stmts.push_back(
        text::AssignAST(var_id, rhs, ffi::Optional<text::ExprAST>()));
  }

  // Predicate
  if (!is_one(realize->predicate)) {
    text::ExprAST pred = Print(printer, realize->predicate, path->Attr("predicate"));
    frame->stmts.push_back(
        text::ExprStmtAST(text::ExprCall(TIR("where"), {pred})));
  }

  // Reads
  {
    ffi::List<text::ExprAST> reads;
    for (int i = 0; i < static_cast<int>(block->reads.size()); ++i) {
      reads.push_back(
          Print(printer, block->reads[i], block_p->Attr("reads")->ArrayItem(i)));
    }
    frame->stmts.push_back(
        text::ExprStmtAST(text::ExprCall(TIR("reads"), std::move(reads))));
  }

  // Writes
  {
    ffi::List<text::ExprAST> writes;
    for (int i = 0; i < static_cast<int>(block->writes.size()); ++i) {
      writes.push_back(
          Print(printer, block->writes[i], block_p->Attr("writes")->ArrayItem(i)));
    }
    frame->stmts.push_back(
        text::ExprStmtAST(text::ExprCall(TIR("writes"), std::move(writes))));
  }

  // Annotations
  if (!block->annotations.empty()) {
    text::ExprAST annot = Print(printer, block->annotations, block_p->Attr("annotations"));
    frame->stmts.push_back(
        text::ExprStmtAST(text::ExprCall(TIR("sblock_attr"), {annot})));
  }

  // Alloc buffers
  for (int i = 0; i < static_cast<int>(block->alloc_buffers.size()); ++i) {
    Buffer buf = block->alloc_buffers[i];
    text::AccessPath buffer_p = block_p->Attr("alloc_buffers")->ArrayItem(i);
    std::string buf_name = buf->name;
    if (buf_name.empty()) buf_name = "buffer";
    printer->VarDef(buf_name, buf, frame);
    text::ExprAST buf_id = printer->VarGet(buf).value();
    ffi::List<text::ExprAST> no_extra;
    text::ExprAST rhs = PrintBufferDecl(buf, "sblock_alloc_buffer", std::move(no_extra),
                                   printer, buffer_p);
    DefineBufferDataVar(buf, printer);
    frame->stmts.push_back(
        text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>()));
  }

  // Match buffers
  for (int i = 0; i < static_cast<int>(block->match_buffers.size()); ++i) {
    text::NodeAST s = printer->operator()(ffi::Any(block->match_buffers[i]),
                                     block_p->Attr("match_buffers")->ArrayItem(i))
                    .cast<text::NodeAST>();
    if (s->IsInstance<text::StmtASTObj>()) {
      frame->stmts.push_back(Downcast<text::StmtAST>(s));
    }
  }

  // Init
  if (block->init.defined()) {
    ffi::List<text::StmtAST> init_body = PrintBodyStmts(block->init.value(), printer,
                                              block_p->Attr("init"));
    text::ExprAST init_ctx = text::ExprCall(TIR("init"), {});
    frame->stmts.push_back(
        text::WithAST(ffi::Optional<text::ExprAST>(), init_ctx, init_body));
  }

  // Body
  ffi::List<text::StmtAST> body = PrintBodyStmts(block->body, printer, block_p->Attr("body"));

  // Merge
  ffi::List<text::StmtAST> all_body;
  for (const auto& s : frame->stmts) all_body.push_back(s);
  for (const auto& s : body) all_body.push_back(s);

  printer->FramePop();
  return text::WithAST(ffi::Optional<text::ExprAST>(), ctx, all_body);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;

  // AllocBuffer, DeclBuffer, MatchBufferRegion
  refl::TypeAttrDef<AllocBufferNode>().def(
      "__ffi_text_print__",
      [](AllocBuffer node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        Buffer buf = node->buffer;
        text::DefaultFrame frame = printer->frames.back().cast<text::DefaultFrame>();
        printer->VarDef(buf->name, buf, frame);
        text::ExprAST buf_id = printer->VarGet(buf).value();
        text::AccessPath buffer_p = path->Attr("buffer");
        ffi::List<text::ExprAST> no_extra;
        text::ExprAST rhs = PrintBufferDecl(buf, "alloc_buffer", std::move(no_extra),
                                       printer, buffer_p,
                                       node->annotations, path->Attr("annotations"));
        DefineBufferDataVar(buf, printer);
        return text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>());
      });

  refl::TypeAttrDef<DeclBufferNode>().def(
      "__ffi_text_print__",
      [](DeclBuffer node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        Buffer buf = node->buffer;
        text::DefaultFrame frame = printer->frames.back().cast<text::DefaultFrame>();
        DefineBufferVars(buf, printer, frame);
        printer->VarDef(buf->name, buf, frame);
        text::ExprAST buf_id = printer->VarGet(buf).value();
        ffi::List<text::ExprAST> no_extra;
        text::ExprAST rhs = PrintBufferDecl(buf, "decl_buffer", std::move(no_extra),
                                       printer, path->Attr("buffer"));
        DefineBufferDataVar(buf, printer);
        return text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>());
      });

  refl::TypeAttrDef<MatchBufferRegionNode>().def(
      "__ffi_text_print__",
      [](MatchBufferRegion node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        Buffer buf = node->buffer;
        text::DefaultFrame frame = printer->frames.back().cast<text::DefaultFrame>();
        DefineBufferVars(buf, printer, frame);
        printer->VarDef(buf->name, buf, frame);
        text::ExprAST buf_id = printer->VarGet(buf).value();
        text::ExprAST source = Print(printer, node->source, path->Attr("source"));
        ffi::List<text::ExprAST> extra_args;
        extra_args.push_back(source);
        text::ExprAST rhs = PrintBufferDecl(buf, "match_buffer", std::move(extra_args),
                                       printer, path->Attr("buffer"));
        DefineBufferDataVar(buf, printer);
        return text::AssignAST(buf_id, rhs, ffi::Optional<text::ExprAST>());
      });

  // AttrStmt
  refl::TypeAttrDef<AttrStmtNode>().def(
      "__ffi_text_print__",
      [](AttrStmt node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        bool is_iter_var = (node->node.type_index() >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) &&
                           node->node.cast<ffi::ObjectRef>()->IsInstance<IterVarNode>();
        if ((node->attr_key == "thread_extent" || node->attr_key == "virtual_thread") &&
            is_iter_var) {
          IterVar iv = node->node.cast<IterVar>();
          bool was_defined = printer->VarGet(iv->var).has_value();
          if (was_defined) {
            ffi::List<text::StmtAST> body = PrintBodyStmts(node->body, printer, path->Attr("body"));
            text::ExprAST var = printer->VarGet(iv->var).value();
            text::ExprAST ctx = text::ExprCall(TIR("launch_thread"),
                                   {var, Print(printer, node->value, path->Attr("value"))});
            return text::WithAST(ffi::Optional<text::ExprAST>(), ctx, body);
          } else {
            text::DefaultFrame inner_frame;
            printer->FramePush(inner_frame);
            DefineVar(iv->var, printer, path->Attr("node")->Attr("var"));
            ffi::List<text::StmtAST> body = PrintBodyStmts(node->body, printer, path->Attr("body"));
            text::ExprAST var = printer->VarGet(iv->var).value();
            printer->FramePop();
            text::ExprAST ctx = text::ExprCall(TIR("launch_thread"),
                                   {text::LiteralAST::Str(iv->thread_tag),
                                    Print(printer, node->value, path->Attr("value"))});
            return text::WithAST(ffi::Optional<text::ExprAST>(var), ctx, body);
          }
        }
        ffi::List<text::StmtAST> body = PrintBodyStmts(node->body, printer, path->Attr("body"));
        text::ExprAST ctx = text::ExprCall(TIR("attr"),
                               {Print(printer, node->node, path->Attr("node")),
                                text::LiteralAST::Str(node->attr_key, {path->Attr("attr_key")}),
                                Print(printer, node->value, path->Attr("value"))});
        return text::WithAST(ffi::Optional<text::ExprAST>(), ctx, body);
      });

  // ForNode
  refl::TypeAttrDef<ForNode>().def(
      "__ffi_text_print__",
      [](tirx::For loop, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        // Step 1. Check syntactic sugar: T.grid
        std::vector<const ForNode*> grid;
        std::unordered_set<const tirx::VarNode*> grid_loop_vars;
        {
          for (const ForNode* l = loop.get(); l != nullptr;
               l = l->body.as<ForNode>()) {
            if (l->kind != ForKind::kSerial ||
                !is_zero(l->min) ||
                !l->annotations.empty() ||
                !l->HasTrivialStep() ||
                tirx::UsesVar(l->extent, [&grid_loop_vars](const tirx::VarNode* v) {
                  return grid_loop_vars.count(v) > 0;
                })) {
              break;
            }
            grid.push_back(l);
            grid_loop_vars.insert(l->loop_var.get());
          }
        }

        // Step 2. If grid.size() > 1, print as T.grid
        if (grid.size() > 1) {
          text::DefaultFrame frame;
          printer->FramePush(frame);
          int n = grid.size();
          ffi::List<text::ExprAST> lhs_vars;
          ffi::List<text::ExprAST> extents;
          text::AccessPath cur_p = path;
          for (int i = 0; i < n; ++i) {
            const ForNode* g = grid[i];
            lhs_vars.push_back(
                DefineVar(ffi::GetRef<Var>(static_cast<const VarNode*>(g->loop_var.get())),
                          printer, cur_p->Attr("loop_var")));
            extents.push_back(Print(printer, g->extent, cur_p->Attr("extent")));
            cur_p = cur_p->Attr("body");
          }
          text::ExprAST lhs = text::TupleAST({}, std::move(lhs_vars));
          text::ExprAST rhs = text::ExprCall(TIR("grid"), std::move(extents));
          ffi::List<text::StmtAST> body = PrintBodyStmts(
              ffi::GetRef<Stmt>(static_cast<const StmtNode*>(grid.back()->body.get())),
              printer, cur_p);
          ffi::List<text::StmtAST> all_body;
          for (const auto& s : frame->stmts) all_body.push_back(s);
          for (const auto& s : body) all_body.push_back(s);
          printer->FramePop();
          return text::ForAST(lhs, rhs, all_body);
        }

        // Step 3. Single for loop (no grid sugar)
        text::DefaultFrame frame;
        printer->FramePush(frame);
        text::ExprAST lhs = DefineVar(loop->loop_var, printer, path->Attr("loop_var"));

        text::ExprAST rhs(ffi::UnsafeInit{});
        bool is_zero_min = is_zero(loop->min);
        bool has_trivial_step = loop->HasTrivialStep();

        if (loop->kind == ForKind::kSerial && loop->annotations.empty()) {
          ffi::List<text::ExprAST> range_args;
          if (is_zero_min && has_trivial_step) {
            range_args.push_back(Print(printer, loop->extent, path->Attr("extent")));
          } else {
            PrimExpr end = loop->min + loop->extent;
            range_args.push_back(Print(printer, loop->min, path->Attr("min")));
            range_args.push_back(Print(printer, end, path->Attr("extent")));
          }
          if (!has_trivial_step) {
            range_args.push_back(Print(printer, *loop->step, path->Attr("step")));
          }
          rhs = text::ExprCall(text::IdAST("range"), std::move(range_args));
        } else {
          std::string prefix;
          switch (loop->kind) {
            case ForKind::kSerial: prefix = "serial"; break;
            case ForKind::kParallel: prefix = "parallel"; break;
            case ForKind::kUnrolled: prefix = "unroll"; break;
            case ForKind::kVectorized: prefix = "vectorized"; break;
            case ForKind::kThreadBinding: prefix = "thread_binding"; break;
            default: prefix = "serial"; break;
          }
          ffi::List<text::ExprAST> args;
          if (is_zero_min && has_trivial_step) {
            args.push_back(Print(printer, loop->extent, path->Attr("extent")));
          } else {
            PrimExpr end = loop->min + loop->extent;
            args.push_back(Print(printer, loop->min, path->Attr("min")));
            args.push_back(Print(printer, end, path->Attr("extent")));
          }
          ffi::List<ffi::String> kw_keys;
          ffi::List<text::ExprAST> kw_vals;
          if (!loop->annotations.empty()) {
            kw_keys.push_back(ffi::String("annotations"));
            kw_vals.push_back(Print(printer, loop->annotations, path->Attr("annotations")));
          }
          if (loop->kind == ForKind::kThreadBinding && loop->thread_binding.defined()) {
            kw_keys.push_back(ffi::String("thread"));
            kw_vals.push_back(
                text::LiteralAST::Str(loop->thread_binding.value()->thread_tag,
                                {path->Attr("thread_binding")}));
          }
          if (!has_trivial_step) {
            kw_keys.push_back(ffi::String("step"));
            kw_vals.push_back(Print(printer, *loop->step, path->Attr("step")));
          }
          rhs = !kw_keys.empty()
              ? text::ExprCallKw(TIR(prefix), std::move(args), std::move(kw_keys), std::move(kw_vals))
              : text::ExprCall(TIR(prefix), std::move(args));
        }

        ffi::List<text::StmtAST> body = PrintBodyStmts(loop->body, printer, path->Attr("body"));
        ffi::List<text::StmtAST> all_body;
        for (const auto& s : frame->stmts) all_body.push_back(s);
        for (const auto& s : body) all_body.push_back(s);
        printer->FramePop();
        return text::ForAST(lhs, rhs, all_body);
      });

  // SBlockRealize + SBlock
  refl::TypeAttrDef<SBlockRealizeNode>().def(
      "__ffi_text_print__",
      [](SBlockRealize node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        return PrintSBlockRealize(node, printer, path);
      });

  refl::TypeAttrDef<SBlockNode>().def(
      "__ffi_text_print__",
      [](SBlock block, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        ffi::Array<PrimExpr> iter_values;
        for (const auto& iv : block->iter_vars) {
          iter_values.push_back(iv->var);
        }
        SBlockRealize realize(iter_values, Bool(true), block);
        return PrintSBlockRealize(realize, printer, path);
      });
}

}  // namespace tirx
}  // namespace tvm
