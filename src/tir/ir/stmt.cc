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
 * \file tvm/tir/stmt.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "buffer_common.h"

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() {
  StmtNode::RegisterReflection();
  LetStmtNode::RegisterReflection();
  AttrStmtNode::RegisterReflection();
  AssertStmtNode::RegisterReflection();
  BufferStoreNode::RegisterReflection();
  BufferRealizeNode::RegisterReflection();
  AllocateNode::RegisterReflection();
  AllocateConstNode::RegisterReflection();
  DeclBufferNode::RegisterReflection();
  SeqStmtNode::RegisterReflection();
  EvaluateNode::RegisterReflection();
  IfThenElseNode::RegisterReflection();
  ForNode::RegisterReflection();
  WhileNode::RegisterReflection();
  BufferRegionNode::RegisterReflection();
  MatchBufferRegionNode::RegisterReflection();
  BlockNode::RegisterReflection();
  BlockRealizeNode::RegisterReflection();
}

// LetStmt
LetStmt::LetStmt(Var var, PrimExpr value, Stmt body, Span span) {
  ICHECK(value.defined());
  ICHECK(body.defined());
  auto vdtype = value.dtype();
  // It is still valid to bind a pointer type
  // var to a value that is of type handle.
  if (var->type_annotation.as<PointerTypeNode>()) {
    ICHECK(vdtype.is_handle());
  } else {
    ICHECK_EQ(value.dtype(), var.dtype());
  }

  ObjectPtr<LetStmtNode> node = ffi::make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.LetStmt", [](Var var, PrimExpr value, Stmt body, Span span) {
    return LetStmt(var, value, body, span);
  });
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
  refl::GlobalDef().def("tir.AttrStmt",
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
AssertStmt::AssertStmt(PrimExpr condition, PrimExpr message, Stmt body, Span span) {
  ICHECK(condition.defined());
  CHECK(condition.dtype().is_bool())
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition.dtype();
  ICHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = ffi::make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AssertStmt",
                        [](PrimExpr condition, StringImm message, Stmt body, Span span) {
                          return AssertStmt(condition, message, body, span);
                        });
}

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         ffi::Optional<IterVar> thread_binding, ffi::Map<ffi::String, Any> annotations,
         ffi::Optional<PrimExpr> step, Span span) {
  ICHECK(loop_var.defined());
  ICHECK(min.defined());
  ICHECK(extent.defined());
  ICHECK(body.defined());

  auto require_scalar_int_dtype = [&](PrimExpr expr, const char* field_name) {
    auto dtype = expr.dtype();
    CHECK(dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()))
        << "TIR For nodes require a scalar integer as the " << field_name << ", but received "
        << expr << " with dtype " << dtype;
  };
  require_scalar_int_dtype(loop_var, "loop_var");
  require_scalar_int_dtype(min, "min");
  require_scalar_int_dtype(extent, "extent");

  // When extent, min or step is an IntImm but has narrower dtype than loop_var
  // we directly promote them without raising errors.
  auto try_promote_imm_dtype = [&](const PrimExpr& e) {
    ICHECK(e.dtype().bits() <= loop_var.dtype().bits())
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

  ICHECK(loop_var.dtype() == min.dtype()) << loop_var.dtype() << " vs " << min.dtype();
  ICHECK(loop_var.dtype() == extent.dtype()) << loop_var.dtype() << " vs " << extent.dtype();

  if (step.has_value()) {
    require_scalar_int_dtype(*step, "step");
    step = try_promote_imm_dtype(*step);
    ICHECK(loop_var.dtype() == (*step).dtype()) << loop_var.dtype() << " vs " << (*step).dtype();
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
  refl::GlobalDef().def("tir.For", [](Var loop_var, PrimExpr min, PrimExpr extent, int kind,
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
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_scalar());
  ICHECK(body.defined());

  ObjectPtr<WhileNode> node = ffi::make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.While", [](PrimExpr condition, Stmt body, Span span) {
    return While(condition, body, span);
  });
}

// Allocate
Allocate::Allocate(Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body, ffi::Map<ffi::String, Any> annotations, Span span) {

  for (size_t i = 0; i < extents.size(); ++i) {
    ICHECK(extents[i].defined());
    ICHECK(extents[i].dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<AllocateNode> node = ffi::make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

int64_t AllocateNode::ConstantAllocationSize(const ffi::Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int64_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int64_t>(result);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.Allocate",
      [](Var buffer_var, DataType type, ffi::Array<PrimExpr> extents, PrimExpr condition, Stmt body,
         ffi::Map<ffi::String, Any> annotations, Span span) {
        return Allocate(buffer_var, type, extents, condition, body, annotations, span);
      });
}

// Const
// The constructor to create a IRNode with constant data
// depending on the type of ObjectRef, it will either
// create AllocateConstNode with irmod_storage_idx or data
AllocateConst::AllocateConst(Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents,
                             ObjectRef data_or_idx, Stmt body,
                             ffi::Map<ffi::String, Any> annotations, Span span) {
  ICHECK(IsPointerType(buffer_var->type_annotation, dtype))
      << "The allocated data type (" << dtype
      << ") does not match the type annotation of the buffer " << buffer_var << " ("
      << buffer_var->type_annotation
      << "). The data type should be an element of the pointer type.";

  for (size_t i = 0; i < extents.size(); ++i) {
    ICHECK(extents[i].defined());
    ICHECK(extents[i].dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(data_or_idx.defined());

  ObjectPtr<AllocateConstNode> node = ffi::make_object<AllocateConstNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->body = std::move(body);
  node->annotations = annotations;
  node->span = std::move(span);
  if (data_or_idx->IsInstance<runtime::Tensor::ContainerType>()) {
    node->data = ffi::Optional<tvm::runtime::Tensor>(Downcast<runtime::Tensor>(data_or_idx));
    node->irmod_storage_idx = ffi::Optional<Integer>();
  } else if (data_or_idx->IsInstance<IntImmNode>()) {
    node->data = ffi::Optional<tvm::runtime::Tensor>();
    node->irmod_storage_idx = ffi::Optional<Integer>(Downcast<Integer>(data_or_idx));
  } else {
    LOG(FATAL) << "Data type not supported: " << data_or_idx->GetTypeKey();
  }
  data_ = std::move(node);
}

int64_t AllocateConstNode::ConstantAllocationSize(const ffi::Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int64_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int64_t>(result);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.AllocateConst",
      [](Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, ObjectRef data_or_idx,
         Stmt body, ffi::Optional<ffi::Map<ffi::String, Any>> annotations, Span span) {
        return AllocateConst(buffer_var, dtype, extents, data_or_idx, body,
                             annotations.value_or({}), span);
      });
}

// DeclBuffer
DeclBuffer::DeclBuffer(Buffer buffer, Stmt body, Span span) {
  ObjectPtr<DeclBufferNode> node = ffi::make_object<DeclBufferNode>();
  node->buffer = std::move(buffer);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.DeclBuffer", [](Buffer buffer, Stmt body, Span span) {
    return DeclBuffer(buffer, body, span);
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

  ICHECK_NE(seq.size(), 0) << "An empty SeqStmt is prohibited.  "
                           << "To write a no-op, use Evaluate(0), "
                           << "or the result of SeqStmt::Flatten()";
  ICHECK_NE(seq.size(), 1) << "A SeqStmt of length 1 is prohibited.  "
                           << "Use the node " << seq[0] << "directly, "
                           << "or for dynamic usage, normalize using SeqStmt::Flatten()";

  auto node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.SeqStmt", [](ffi::Array<Stmt> seq, Span span) { return SeqStmt(std::move(seq), span); });
}

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, ffi::Optional<Stmt> else_case,
                       Span span) {
  ICHECK(condition.defined());
  ICHECK(then_case.defined());
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
  refl::GlobalDef().def("tir.IfThenElse",
                        [](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
                          return IfThenElse(condition, then_case, else_case, span);
                        });
}

// Evaluate
Evaluate::Evaluate(PrimExpr value, Span span) {
  ICHECK(value.defined());

  ObjectPtr<EvaluateNode> node = ffi::make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Evaluate",
                        [](PrimExpr value, Span span) { return Evaluate(value, span); });
}

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                         ffi::Optional<PrimExpr> predicate, Span span) {
  ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    ICHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().dtype().is_scalable_vector();
  bool is_buffer_dtype_scalable = buffer->dtype.is_scalable_vector();
  bool is_value_dtype_scalable = value.dtype().is_scalable_vector();

  ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
      << "Index dtype and buffer dtype can't both be scalable.";

  if (predicate.defined()) {
    bool is_predicate_dtype_scalable = predicate.value().dtype().is_scalable_vector();
    ICHECK_EQ(is_value_dtype_scalable, is_predicate_dtype_scalable)
        << "Predicate mask dtype and value dtype must both be scalable.";
  }

  if (is_index_scalable || is_buffer_dtype_scalable) {
    ICHECK(is_value_dtype_scalable) << "Can't store non-scalable data into scalable buffer";
  }

  int index_lanes = indices.empty() ? 1 : indices.back().dtype().get_lanes_or_vscale_factor();
  int buffer_lanes = buffer->dtype.get_lanes_or_vscale_factor();
  int value_dtype_lanes = value.dtype().get_lanes_or_vscale_factor();

  ICHECK_EQ(index_lanes * buffer_lanes, value_dtype_lanes)
      << "Cannot store value with " << value_dtype_lanes << ", expected value with "
      << index_lanes * buffer_lanes << " (" << index_lanes << " index lanes * " << buffer_lanes
      << " buffer element lanes)";

  if (predicate.defined()) {
    DataType predicate_dtype = predicate.value().dtype();
    int predicate_dtype_lanes = predicate_dtype.get_lanes_or_vscale_factor();
    ICHECK_EQ(value_dtype_lanes, predicate_dtype_lanes)
        << "Got a predicate mask with " << predicate_dtype_lanes
        << " lanes, but trying to store a value with " << value_dtype_lanes
        << " lanes. The number of lanes must match.";

    DataType predicate_element_dtype = predicate_dtype.element_of();
    ICHECK(predicate_element_dtype.is_predicate_dtype())
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
    LOG(FATAL) << "TypeError: dtype mismatch on BufferStore: "      //
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
  refl::GlobalDef().def("tir.BufferStore",
                        [](Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                           ffi::Optional<PrimExpr> predicate, Span span) {
                          return BufferStore(buffer, value, indices, predicate, span);
                        });
}

// BufferRealize
BufferRealize::BufferRealize(Buffer buffer, ffi::Array<Range> bounds, PrimExpr condition, Stmt body,
                             Span span) {
  data_ = ffi::make_object<BufferRealizeNode>(buffer, bounds, condition, body, span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.BufferRealize", [](Buffer buffer, ffi::Array<Range> bounds,
                                                PrimExpr condition, Stmt body, Span span) {
    return BufferRealize(buffer, bounds, condition, body, span);
  });
}

// BufferRegion
PrimExpr BufferRegionNode::ToPrimExpr() const {
  // Auto convert to PrimExpr if it is a single point load
  ffi::Array<PrimExpr> indices;
  indices.reserve(this->region.size());
  for (const Range& r : this->region) {
    if (tvm::tir::is_one(r->extent)) {
      indices.push_back(r->min);
    } else if (r->extent.as<IntImmNode>()) {
      indices.push_back(tir::Ramp(r->min, tvm::tir::make_const(r->min->dtype, 1), r->extent));
    } else {
      LOG(FATAL) << "ValueError: Cannot convert to BufferLoad: " << ffi::GetRef<BufferRegion>(this);
    }
  }
  return tir::BufferLoad(this->buffer, indices);
}

BufferRegion::BufferRegion(Buffer buffer, ffi::Array<Range> region) {
  CHECK_EQ(buffer->shape.size(), region.size())
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
  refl::GlobalDef().def("tir.BufferRegion", [](Buffer buffer, ffi::Array<Range> region) {
    return BufferRegion(buffer, region);
  });
}

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(Buffer buffer, BufferRegion source) {
  const Buffer& source_buffer = source->buffer;
  arith::Analyzer analyzer;
  // Check scope and dtype
  CHECK_EQ(buffer.scope(), source_buffer.scope())
      << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << " vs. "
      << source_buffer.scope();
  CHECK_EQ(buffer->dtype, source_buffer->dtype)
      << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << " vs. "
      << source_buffer->dtype;

  // Check data_alignment
  CHECK(source_buffer->data_alignment % buffer->data_alignment == 0)
      << "Trying to match buffer to another one with lower alignment requirement "
      << " required alignment=" << buffer->data_alignment
      << ", provided alignment=" << source_buffer->data_alignment;

  // Check BufferType. AutoBroadcast is not allowed for now.
  CHECK(buffer->buffer_type == BufferType::kDefault &&
        source_buffer->buffer_type == BufferType::kDefault)
      << "AutoBroadcast is not allowed in MatchBuffer";

  // Validate shape
  CHECK(source->region.size() >= buffer->shape.size())
      << "Dimension of source Region expected to be larger or equal than target buffer shape, but "
         "got "
      << source->region.size() << " vs. " << buffer->shape.size();
  size_t offset = source->region.size() - buffer->shape.size();
  for (size_t i = 0; i < offset; ++i) {
    CHECK(analyzer.CanProve(source->region[i]->extent == 1))
        << "The higher dimension should be 1, but got " << source->region[i]->extent << ".";
  }
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const PrimExpr& buffer_shape = buffer->shape[i];
    if (!buffer_shape->IsInstance<VarNode>()) {
      CHECK(analyzer.CanProve(source_range->extent == buffer_shape))
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
  refl::GlobalDef().def("tir.MatchBufferRegion", [](Buffer buffer, BufferRegion source) {
    return MatchBufferRegion(buffer, source);
  });
}

// Block
Block::Block(ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
             ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
             ffi::Optional<Stmt> init, ffi::Array<Buffer> alloc_buffers,
             ffi::Array<MatchBufferRegion> match_buffers, ffi::Map<ffi::String, Any> annotations,
             Span span) {
  ObjectPtr<BlockNode> node = ffi::make_object<BlockNode>();
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
  refl::GlobalDef().def("tir.Block",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
                           ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
                           ffi::Optional<Stmt> init, ffi::Array<Buffer> alloc_buffers,
                           ffi::Array<MatchBufferRegion> match_buffers,
                           ffi::Map<ffi::String, Any> annotations, Span span) {
                          return Block(iter_vars, reads, writes, name_hint, body, init,
                                       alloc_buffers, match_buffers, annotations, span);
                        });
}

// BlockRealize
BlockRealize::BlockRealize(ffi::Array<PrimExpr> values, PrimExpr predicate, Block block,
                           Span span) {
  CHECK_EQ(block->iter_vars.size(), values.size())
      << "ValueError: BlockRealize needs to have the same number of iter_vars and binding values";
  CHECK(predicate.dtype().is_bool() || predicate.dtype() == DataType::UInt(1))
      << "TypeError: Expect Block.predicate to be a bool expression";
  ObjectPtr<BlockRealizeNode> node = ffi::make_object<BlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.BlockRealize", [](ffi::Array<PrimExpr> iter_values, PrimExpr predicate,
                                               Block block, Span span) {
    return BlockRealize(iter_values, predicate, block, span);
  });
}

PrimExpr TypeAnnotation(DataType dtype, Span span) {
  static auto op = Op::Get("tir.type_annotation");
  return tir::Call(dtype, op, {}, span);
}

TVM_TIR_REGISTER_OP("type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

}  // namespace tir
}  // namespace tvm
