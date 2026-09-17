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

#include "data_type_rewriter.h"

#include <tvm/tirx/op.h>

#include <functional>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
using namespace tvm::prim;

PrimFunc IndexDataTypeNormalizer::Rewrite(PrimFunc func) {
  // Collect scalar dtype requirements without changing types.  Buffer definitions
  // are rewritten only after every scalar replacement has been seeded.
  class IndexVarCollector : public IndexDataTypeNormalizer {
   public:
    explicit IndexVarCollector(std::function<void(const VarNode*)> collect)
        : IndexDataTypeNormalizer(PrimType::Int(64)), collect_(std::move(collect)) {}
    using IndexDataTypeNormalizer::Mutate;
    using IndexDataTypeNormalizer::Mutate_;
    UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode mode) final {
      if (def_region_kind() == kTVMFFIDefRegionKindNone && is_enabled_) collect_(op);
      return IndexDataTypeNormalizer::Mutate_(op, mode);
    }

   protected:
    bool CanRewriteDType(PrimType dtype) const final { return false; }

   private:
    std::function<void(const VarNode*)> collect_;
  };
  auto seed = [this](const VarNode* var) {
    auto dtype = var->ty.as<PrimType>();
    if (dtype && CanRewriteDType(dtype.value()) && dtype.value() != target_data_type_ &&
        VarRemapGet(ffi::AnyView(var)) == nullptr) {
      VarRemapSet(ffi::AnyView(var), ffi::GetRef<Var>(var).CopyWithDType(target_data_type_));
    }
  };
  auto collector = ffi::make_object<IndexVarCollector>(seed);
  collector->Mutate(func->body);
  for (const Var& param : func->params) {
    if (param.as<BufferVar>()) {
      collector->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                   [&] { return collector->Mutate(param); });
    } else {
      seed(param.get());
    }
  }
  ffi::Array<Var> params = func->params.Map([this](const Var& param) {
    return WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return Mutate(param).ValueOrUnchanged(param).as_or_throw<Var>();
    });
  });
  PrimFuncNode* new_func = func.CopyOnWrite();
  new_func->params = std::move(params);
  new_func->body = Mutate(new_func->body).ValueOrUnchanged(new_func->body);
  return func;
}

UnchangedOr<Stmt> IndexDataTypeNormalizer::Mutate_(const SBlockRealizeNode* op,
                                                   InplaceMode inplace_mode) {
  bool is_condition = this->is_condition_;
  this->is_condition_ = true;
  auto new_predicate_result = this->Mutate(op->predicate, inplace_mode);
  bool new_predicate_unchanged = new_predicate_result.UnchangedOrSameAs(op->predicate);
  auto new_predicate = std::move(new_predicate_result).ValueOrUnchanged(op->predicate);
  this->is_condition_ = is_condition;

  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  auto new_iter_values = this->Mutate(op->iter_values, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                             .ValueOrUnchanged(op->iter_values);
  this->is_enabled_ = is_enabled;
  SBlock new_body =
      this->Mutate(op->block, inplace_mode).ValueOrUnchanged(op->block).as_or_throw<SBlock>();
  if (!new_predicate_unchanged || !new_iter_values.same_as(op->iter_values) ||
      !new_body.same_as(op->block)) {
    SBlockRealize new_block_realize = ffi::GetRef<SBlockRealize>(op);
    auto* n = new_block_realize.CopyOnWrite();
    n->predicate = std::move(new_predicate);
    n->iter_values = std::move(new_iter_values);
    n->block = std::move(new_body);
    return new_block_realize;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeNormalizer::Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
  auto new_alloc_buffers = this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
    return this->Mutate(op->alloc_buffers, inplace_mode)
        .as_or_throw<UnchangedOr<ffi::Array<BufferVar>>>()
        .ValueOrUnchanged(op->alloc_buffers);
  });
  auto new_match_buffers = op->match_buffers.Map([this](const MatchBufferRegion& match) {
    BufferVar buffer = this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return this->Mutate(match->buffer, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<BufferVar>>()
          .ValueOrUnchanged(match->buffer);
    });
    TensorRegion source = VisitBufferRegion(match->source);
    if (buffer.same_as(match->buffer) && source.same_as(match->source)) return match;
    return MatchBufferRegion(buffer, source);
  });
  ffi::Array<TensorRegion> new_reads = op->reads.Map(
      [this](const TensorRegion& buffer_region) { return VisitBufferRegion(buffer_region); });
  ffi::Array<TensorRegion> new_writes = op->writes.Map(
      [this](const TensorRegion& buffer_region) { return VisitBufferRegion(buffer_region); });
  ffi::Array<IterVar> new_iter_vars =
      op->iter_vars.Map([this](const IterVar& iter_var) { return VisitIterVar(iter_var); });
  ffi::Optional<Stmt> new_init = std::nullopt;
  if (op->init.has_value()) {
    new_init = this->Mutate(op->init.value(), inplace_mode).ValueOrUnchanged(op->init.value());
  }
  ffi::Map<ffi::String, ffi::Any> new_annotations = VisitBlockAnnotations(op->annotations);
  auto new_body_result = this->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_init.same_as(op->init) || !new_body_unchanged ||
      !new_alloc_buffers.same_as(op->alloc_buffers) ||
      !new_match_buffers.same_as(op->match_buffers) || !new_reads.same_as(op->reads) ||
      !new_writes.same_as(op->writes) || !new_iter_vars.same_as(op->iter_vars) ||
      !new_annotations.same_as(op->annotations)) {
    SBlock new_block = ffi::GetRef<SBlock>(op);
    SBlockNode* n = new_block.CopyOnWrite();
    n->alloc_buffers = std::move(new_alloc_buffers);
    n->match_buffers = std::move(new_match_buffers);
    n->reads = std::move(new_reads);
    n->writes = std::move(new_writes);
    n->iter_vars = std::move(new_iter_vars);
    n->init = std::move(new_init);
    n->annotations = std::move(new_annotations);
    n->body = std::move(new_body);
    return new_block;
  }
  return ffi::Unchanged();
}

ffi::Map<ffi::String, ffi::Any> IndexDataTypeNormalizer::VisitBlockAnnotations(
    const ffi::Map<ffi::String, ffi::Any>& annotations) {
  auto new_annotations = annotations;

  std::function<Any(const Any&)> f_mutate_obj = [this, &f_mutate_obj](const Any& obj) -> Any {
    if (obj == nullptr) {
      return obj;
    }
    if (auto var = obj.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      BufferVar buffer(var.value());
      if (BufferVar new_buffer = this->Mutate(buffer, InplaceMode::kDisallow)
                                     .as_or_throw<UnchangedOr<BufferVar>>()
                                     .ValueOrUnchanged(buffer);
          !new_buffer.same_as(buffer)) {
        return new_buffer;
      }
    } else if (obj.as<ffi::ArrayObj>()) {
      return obj.as_or_throw<ffi::Array<Any>>().Map(f_mutate_obj);
    }
    return obj;
  };
  for (const auto& [key, value] : annotations) {
    if (auto opt_object_ref = value.as<ffi::ObjectRef>()) {
      auto new_value = f_mutate_obj(*opt_object_ref);
      if (!new_value.same_as(*opt_object_ref)) {
        new_annotations.Set(key, new_value);
      }
    }
  }
  return new_annotations;
}

IterVar IndexDataTypeNormalizer::VisitIterVar(const IterVar& iter_var) {
  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  PrimVar new_var = this->Mutate(iter_var->var, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->var)
                        .as_or_throw<PrimVar>();
  PrimExpr min =
      this->Mutate(iter_var->dom->min, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->min);
  PrimExpr extent = this->Mutate(iter_var->dom->extent, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->dom->extent);
  this->is_enabled_ = is_enabled;
  if (!new_var.same_as(iter_var->var) || !min.same_as(iter_var->dom->min) ||
      !extent.same_as(iter_var->dom->extent)) {
    IterVar new_iter_var = iter_var;
    IterVarNode* n = new_iter_var.CopyOnWrite();
    n->var = std::move(new_var);
    n->dom = Range(min, extent);
    return new_iter_var;
  }
  return iter_var;
}

TensorRegion IndexDataTypeNormalizer::VisitBufferRegion(const TensorRegion& buffer_region) {
  BufferVar remapped_buffer =
      this->Mutate(buffer_region->source.as_or_throw<BufferVar>(), InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<BufferVar>>()
          .ValueOrUnchanged(buffer_region->source.as_or_throw<BufferVar>());

  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  auto new_region = buffer_region->region.Map([&](const Range& range) {
    return Range::FromMinExtent(
        this->Mutate(range->min, InplaceMode::kDisallow).ValueOrUnchanged(range->min),
        this->Mutate(range->extent, InplaceMode::kDisallow).ValueOrUnchanged(range->extent));
  });
  this->is_enabled_ = is_enabled;

  if (!remapped_buffer.same_as(buffer_region->source.as_or_throw<BufferVar>()) ||
      !new_region.same_as(buffer_region->region)) {
    return BufferRegion(remapped_buffer, new_region);
  } else {
    return buffer_region;
  }
}

}  // namespace s_tir
}  // namespace tvm
