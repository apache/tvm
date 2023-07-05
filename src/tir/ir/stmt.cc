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
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include "buffer_common.h"

namespace tvm {
namespace tir {

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

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt")
    .set_body_typed([](Var var, PrimExpr value, Stmt body, Span span) {
      return LetStmt(var, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(LetStmtNode);

// AttrStmt
AttrStmt::AttrStmt(ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt")
    .set_body_typed([](ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
      return AttrStmt(node, attr_key, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, PrimExpr message, Stmt body, Span span) {
  ICHECK(condition.defined());
  CHECK(condition.dtype().is_bool())
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition.dtype();
  ICHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AssertStmtNode);

TVM_REGISTER_GLOBAL("tir.AssertStmt")
    .set_body_typed([](PrimExpr condition, ObjectRef message, Stmt body, Span span) {
      if (const auto* str = message.as<StringObj>()) {
        auto msg = StringImm(str->data);
        return AssertStmt(condition, msg, body, span);
      } else {
        return AssertStmt(condition, Downcast<PrimExpr>(message), body, span);
      }
    });

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         Optional<IterVar> thread_binding, Map<String, ObjectRef> annotations, Span span) {
  ICHECK(min.defined());
  ICHECK(extent.defined());
  ICHECK(min.dtype().is_scalar());
  ICHECK(extent.dtype().is_scalar());
  ICHECK(loop_var.dtype().is_scalar());
  ICHECK(body.defined());

  // When extent or min is an IntImm but has narrower dtype than loop_var, we directly promote them
  // without raising errors.
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

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.For").set_body_typed(
    [](Var loop_var, PrimExpr min, PrimExpr extent, int kind, Stmt body,
       Optional<IterVar> thread_binding, Optional<Map<String, ObjectRef>> annotations, Span span) {
      return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
                 annotations.value_or(Map<String, ObjectRef>()), span);
    });

TVM_REGISTER_NODE_TYPE(ForNode);

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
  ICHECK(condition.as<tir::IntImmNode>() == nullptr) << "The condition should not be trivial.";
  ICHECK(body.defined());

  ObjectPtr<WhileNode> node = make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.While").set_body_typed([](PrimExpr condition, Stmt body, Span span) {
  return While(condition, body, span);
});

TVM_REGISTER_NODE_TYPE(WhileNode);

// ProducerStore
ProducerStore::ProducerStore(DataProducer producer, PrimExpr value, Array<PrimExpr> indices,
                             Span span) {
  ObjectPtr<ProducerStoreNode> node = make_object<ProducerStoreNode>();
  node->producer = std::move(producer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerStore")
    .set_body_typed([](DataProducer producer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return ProducerStore(producer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerStoreNode);

// Allocate
Allocate::Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body, Map<String, ObjectRef> annotations, Span span) {
  CHECK(IsPointerType(buffer_var->type_annotation, dtype) ||
        (dtype.is_bool() && IsPointerType(buffer_var->type_annotation, DataType::Int(8))))
      << "The allocated data type (" << dtype
      << ") does not match the type annotation of the buffer " << buffer_var << " ("
      << buffer_var->type_annotation
      << "). The data type should be an element of the pointer type.";

  for (size_t i = 0; i < extents.size(); ++i) {
    ICHECK(extents[i].defined());
    ICHECK(extents[i].dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

int64_t AllocateNode::ConstantAllocationSize(const Array<PrimExpr>& extents) {
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

TVM_REGISTER_GLOBAL("tir.Allocate")
    .set_body_typed([](Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition,
                       Stmt body, Map<String, ObjectRef> annotations, Span span) {
      return Allocate(buffer_var, type, extents, condition, body, annotations, span);
    });

TVM_REGISTER_NODE_TYPE(AllocateNode);

// Const
// The constructor to create a IRNode with constant data
// depending on the type of ObjectRef, it will either
// create AllocateConstNode with irmod_storage_idx or data
AllocateConst::AllocateConst(Var buffer_var, DataType dtype, Array<PrimExpr> extents,
                             ObjectRef data_or_idx, Stmt body, Map<String, ObjectRef> annotations,
                             Span span) {
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

  ObjectPtr<AllocateConstNode> node = make_object<AllocateConstNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->body = std::move(body);
  node->annotations = annotations;
  node->span = std::move(span);
  if (data_or_idx->IsInstance<runtime::NDArray::ContainerType>()) {
    node->data = Optional<tvm::runtime::NDArray>(Downcast<runtime::NDArray>(data_or_idx));
    node->irmod_storage_idx = Optional<Integer>();
  } else if (data_or_idx->IsInstance<IntImmNode>()) {
    node->data = Optional<tvm::runtime::NDArray>();
    node->irmod_storage_idx = Optional<Integer>(Downcast<Integer>(data_or_idx));
  } else {
    LOG(FATAL) << "Data type not supported: " << data_or_idx->GetTypeKey();
  }
  data_ = std::move(node);
}

int64_t AllocateConstNode::ConstantAllocationSize(const Array<PrimExpr>& extents) {
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
TVM_REGISTER_GLOBAL("tir.AllocateConst")
    .set_body_typed([](Var buffer_var, DataType dtype, Array<PrimExpr> extents,
                       ObjectRef data_or_idx, Stmt body, Map<String, ObjectRef> annotations,
                       Span span) {
      return AllocateConst(buffer_var, dtype, extents, data_or_idx, body, annotations, span);
    });

TVM_REGISTER_NODE_TYPE(AllocateConstNode);

// DeclBuffer
DeclBuffer::DeclBuffer(Buffer buffer, Stmt body, Span span) {
  ObjectPtr<DeclBufferNode> node = make_object<DeclBufferNode>();
  node->buffer = std::move(buffer);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.DeclBuffer").set_body_typed([](Buffer buffer, Stmt body, Span span) {
  return DeclBuffer(buffer, body, span);
});

TVM_REGISTER_NODE_TYPE(DeclBufferNode);

// ProducerRealize
ProducerRealize::ProducerRealize(DataProducer producer, Region bounds, PrimExpr condition,
                                 Stmt body, String storage_scope, Span span) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    ICHECK(bounds[i]->min.defined());
    ICHECK(bounds[i]->extent.defined());
    ICHECK(bounds[i]->min.dtype().is_scalar());
    ICHECK(bounds[i]->extent.dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<ProducerRealizeNode> node = make_object<ProducerRealizeNode>();
  node->producer = std::move(producer);
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  node->storage_scope = std::move(storage_scope);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerRealize")
    .set_body_typed([](DataProducer producer, Region bounds, PrimExpr condition, Stmt body,
                       String storage_scope, Span span) {
      return ProducerRealize(producer, bounds, condition, body, storage_scope, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerRealizeNode);

// Prefetch
Prefetch::Prefetch(Buffer buffer, Array<Range> bounds, Span span) {
  data_ = make_object<PrefetchNode>(buffer, bounds, span);
}

TVM_REGISTER_GLOBAL("tir.Prefetch")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, Span span) {
      return Prefetch(buffer, bounds, span);
    });

TVM_REGISTER_NODE_TYPE(PrefetchNode);

// SeqStmt
SeqStmt::SeqStmt(Array<Stmt> seq, Span span) {
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

  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq, Span span) {
  return SeqStmt(std::move(seq), span);
});

TVM_REGISTER_NODE_TYPE(SeqStmtNode);

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, Optional<Stmt> else_case, Span span) {
  ICHECK(condition.defined());
  ICHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IfThenElseNode);

TVM_REGISTER_GLOBAL("tir.IfThenElse")
    .set_body_typed([](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
      return IfThenElse(condition, then_case, else_case, span);
    });

// Evaluate
Evaluate::Evaluate(PrimExpr value, Span span) {
  ICHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed([](PrimExpr value, Span span) {
  return Evaluate(value, span);
});

TVM_REGISTER_NODE_TYPE(EvaluateNode);

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
  ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    ICHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  int index_lanes = indices.size() ? indices.back().dtype().lanes() : 1;
  int buffer_lanes = buffer->dtype.lanes();

  ICHECK_EQ(index_lanes * buffer_lanes, value.dtype().lanes())
      << "Cannot store value with " << value.dtype().lanes() << ", expected value with "
      << index_lanes * buffer_lanes << " (" << index_lanes << " index lanes * " << buffer_lanes
      << " buffer element lanes)";
  if (buffer->dtype.with_lanes(buffer_lanes * index_lanes) != value.dtype()) {
    LOG(FATAL) << "TypeError: dtype mismatch on BufferStore: "      //
               << "buffer's dtype is `" << buffer->dtype            //
               << "`, the lanes of indexing are: `" << index_lanes  //
               << "`, but RHS's dtype is `" << value.dtype() << "`";
  }

  ObjectPtr<BufferStoreNode> node = make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferStore")
    .set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return BufferStore(buffer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(BufferStoreNode);

// BufferRealize
BufferRealize::BufferRealize(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                             Span span) {
  data_ = make_object<BufferRealizeNode>(buffer, bounds, condition, body, span);
}

TVM_REGISTER_GLOBAL("tir.BufferRealize")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                       Span span) { return BufferRealize(buffer, bounds, condition, body, span); });

TVM_REGISTER_NODE_TYPE(BufferRealizeNode);

// BufferRegion
BufferRegion::BufferRegion(Buffer buffer, Array<Range> region) {
  CHECK_EQ(buffer->shape.size(), region.size())
      << "The dimension between " << buffer << " and region " << region
      << " mismatched, the buffer is " << buffer;
  ObjectPtr<BufferRegionNode> node = make_object<BufferRegionNode>();
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

BufferRegion BufferRegion::FullRegion(Buffer buffer) {
  Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

BufferRegion BufferRegion::FromPoint(Buffer buffer, Array<PrimExpr> indices) {
  Array<Range> region;
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

TVM_REGISTER_GLOBAL("tir.BufferRegion").set_body_typed([](Buffer buffer, Array<Range> region) {
  return BufferRegion(buffer, region);
});

TVM_REGISTER_NODE_TYPE(BufferRegionNode);

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
      << " required_alignment=" << buffer->data_alignment
      << ", provided_alignment=" << source_buffer->data_alignment;

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
  ObjectPtr<MatchBufferRegionNode> node = make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.MatchBufferRegion").set_body_typed([](Buffer buffer, BufferRegion source) {
  return MatchBufferRegion(buffer, source);
});

TVM_REGISTER_NODE_TYPE(MatchBufferRegionNode);

// Block
Block::Block(Array<IterVar> iter_vars, Array<BufferRegion> reads, Array<BufferRegion> writes,
             String name_hint, Stmt body, Optional<Stmt> init, Array<Buffer> alloc_buffers,
             Array<MatchBufferRegion> match_buffers, Map<String, ObjectRef> annotations,
             Span span) {
  ObjectPtr<BlockNode> node = make_object<BlockNode>();
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

TVM_REGISTER_GLOBAL("tir.Block")
    .set_body_typed([](Array<IterVar> iter_vars, Array<BufferRegion> reads,
                       Array<BufferRegion> writes, String name_hint, Stmt body, Optional<Stmt> init,
                       Array<Buffer> alloc_buffers, Array<MatchBufferRegion> match_buffers,
                       Map<String, ObjectRef> annotations, Span span) {
      return Block(iter_vars, reads, writes, name_hint, body, init, alloc_buffers, match_buffers,
                   annotations, span);
    });

TVM_REGISTER_NODE_TYPE(BlockNode);

// BlockRealize
BlockRealize::BlockRealize(Array<PrimExpr> values, PrimExpr predicate, Block block, Span span) {
  CHECK_EQ(block->iter_vars.size(), values.size())
      << "ValueError: BlockRealize needs to have the same number of iter_vars and binding values";
  CHECK(predicate.dtype().is_bool()) << "TypeError: Expect Block.predicate to be a bool expression";
  ObjectPtr<BlockRealizeNode> node = make_object<BlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BlockRealize")
    .set_body_typed([](Array<PrimExpr> iter_values, PrimExpr predicate, Block block, Span span) {
      return BlockRealize(iter_values, predicate, block, span);
    });

TVM_REGISTER_NODE_TYPE(BlockRealizeNode);

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
