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
 * \file src/tirx/ir/specialize.cc
 * \brief Specialize parameters of PrimFunc.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_set>

#include "../transform/ir_utils.h"
#include "functor_common.h"

namespace tvm {
namespace tirx {

using VarMap = std::unordered_map<Var, Expr>;

/**************** Helper functions ****************/

/*! \brief Helper function to check whether the given var is in function parameter list. */
inline bool IsParam(const PrimFunc& func, const Var& param) {
  return std::any_of(func->params.begin(), func->params.end(),
                     [&](const Var& var) { return var.same_as(param); });
}

/**************** Specializer ****************/

// Try fold constants if op's child get specialized to constant.
#define DEFINE_SPECIALIZER_BINARY_OP_MUTATE(BinaryNode, BinaryFunc) \
  Expr VisitExpr_(const BinaryNode* op) final {                     \
    PrimExpr a = VisitPrimExpr(op->a);                              \
    PrimExpr b = VisitPrimExpr(op->b);                              \
    if (a.same_as(op->a) && b.same_as(op->b)) {                     \
      return ffi::GetRef<PrimExpr>(op);                             \
    } else {                                                        \
      return BinaryFunc(a, b);                                      \
    }                                                               \
  }
#define DEFINE_SPECIALIZER_UNARY_OP_MUTATE(UnaryNode, UnaryFunc) \
  Expr VisitExpr_(const UnaryNode* op) final {                   \
    PrimExpr a = VisitPrimExpr(op->a);                           \
    if (a.same_as(op->a)) {                                      \
      return ffi::GetRef<PrimExpr>(op);                          \
    } else {                                                     \
      return UnaryFunc(a);                                       \
    }                                                            \
  }

/*! \brief Mutator to specialize function and remove const parameters */
class PrimFuncSpecializer : public StmtExprMutator {
 public:
  explicit PrimFuncSpecializer(const VarMap& var_map) : var_map_(var_map) {}

  static PrimFunc Specialize(PrimFunc f, const VarMap& var_map) {
    PrimFuncSpecializer specializer(var_map);
    for (const Var& param : f->params) {
      auto buffer = param.as<BufferVar>();
      auto replacement = var_map.find(param);
      if (!buffer || replacement == var_map.end()) {
        continue;
      }
      if (auto replacement_var = replacement->second.as<Var>()) {
        if (auto replacement_buffer = replacement_var.value().as<BufferVar>()) {
          if (IsParam(f, replacement_var.value())) {
            specializer.buffer_aliases_[buffer.value()] = replacement_buffer.value();
          } else {
            specializer.constrained_buffer_params_.insert(param.get());
          }
        }
      }
    }

    // Updating parameters
    ffi::Array<Var> params;
    bool param_updated = false;
    for (const auto& var : f->params) {
      Var new_var = var;
      if (auto buffer = var.as<BufferVar>()) {
        BufferVar new_buffer = specializer.MutateBuffer(buffer.value());
        new_var = new_buffer.var();
        if (!new_buffer.same_as(buffer.value())) {
          param_updated = true;
          specializer.buffer_map_[buffer.value()] = new_buffer;
        }
      }
      // Remove parmeters which has been specialized.
      if (var_map.find(var) == var_map.end() ||
          specializer.constrained_buffer_params_.count(var.get())) {
        params.push_back(new_var);
      } else {
        param_updated = true;
      }
    }

    // Updating function body
    Stmt body = specializer(f->body);

    if (param_updated || !f->body.same_as(body)) {
      return PrimFunc(params, body, f->ret_type, f->attrs, f->span);
    } else {
      return f;
    }
  }

 private:
  Stmt VisitStmt_(const SBlockNode* op) final {
    // Step.0. Define buffer mappings which is allocated inside the block
    ffi::Array<BufferVar> alloc_buffers =
        op->alloc_buffers.Map([this](const auto& buf) { return MutateAllocBuffer(buf); });

    // Step.1. Recursively visit block body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<SBlockNode>();
    TVM_FFI_ICHECK(op != nullptr);

    ffi::Array<BufferRegion> reads =
        op->reads.Map([this](const auto& region) { return MutateBufferRegion(region); });
    ffi::Array<BufferRegion> writes =
        op->writes.Map([this](const auto& region) { return MutateBufferRegion(region); });

    if (alloc_buffers.same_as(op->alloc_buffers) && reads.same_as(op->reads) &&
        writes.same_as(op->writes)) {
      return ffi::GetRef<SBlock>(op);
    } else {
      ffi::ObjectPtr<SBlockNode> n = CopyOnWrite(op);
      n->alloc_buffers = std::move(alloc_buffers);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    // Visit the buffer before delegating to StmtExprMutator, so the
    // buffer's replacement will be defined before the point of use.
    BufferVar new_buf = MutateAllocBuffer(op->buffer);

    auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<DeclBuffer>();

    if (!new_buf.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buf;
    }

    return node;
  }

  // Override VisitBufferUse to use our own buffer_map_ instead of base class field visiting.
  BufferVar VisitBufferUse(const BufferVar& buffer) final { return GetNewBuffer(buffer); }

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (constrained_buffer_params_.count(op)) {
      return var;
    }
    auto it = var_map_.find(var);
    if (it == var_map_.end()) {
      return StmtExprMutator::VisitExpr_(op);
    } else {
      return it->second;
    }
  }

  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::AddNode, add);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::SubNode, sub);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MulNode, mul);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::DivNode, div);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::ModNode, truncmod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::FloorDivNode, floordiv);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::FloorModNode, floormod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MaxNode, max);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MinNode, min);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::EQNode, equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::NENode, not_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::LTNode, less);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::LENode, less_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::GTNode, greater);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::GENode, greater_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::AndNode, logical_and);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::OrNode, logical_or);
  DEFINE_SPECIALIZER_UNARY_OP_MUTATE(prim::NotNode, logical_not);

 private:
  BufferVar MutateBuffer(const BufferVar& buffer) {
    if (auto it = buffer_aliases_.find(buffer); it != buffer_aliases_.end()) {
      return it->second;
    }

    ffi::Optional<ffi::String> specialized_storage_scope;
    if (auto it = var_map_.find(buffer.var()); it != var_map_.end()) {
      if (const auto* new_var = it->second.as<VarNode>()) {
        if (new_var->ty.as<BufferTypeNode>()) {
          BufferVar replacement(ffi::GetRef<Var>(new_var));
          specialized_storage_scope = replacement->storage_scope;
        }
      }
    }

    ffi::Array<PrimExpr> shape =
        buffer->shape.Map([this](const PrimExpr& e) { return VisitPrimExpr(e); });
    ffi::Array<PrimExpr> strides =
        buffer->strides.Map([this](const PrimExpr& e) { return VisitPrimExpr(e); });

    PrimExpr elem_offset = VisitPrimExpr(buffer->elem_offset);

    // Layout iter extents/strides may reference the same shape vars; remap
    // them in lock-step with shape (otherwise the specialized buffer keeps
    // stale layout extents from before specialization).
    ffi::Optional<Layout> layout = buffer->layout;
    bool layout_changed = false;
    if (buffer->layout.has_value()) {
      if (auto opt_tile = buffer->layout.value().as<TileLayoutNode>()) {
        auto remap_iter = [this](const Iter& it) -> Iter {
          PrimExpr new_extent = VisitPrimExpr(it->extent);
          PrimExpr new_stride = VisitPrimExpr(it->stride);
          if (new_extent.same_as(it->extent) && new_stride.same_as(it->stride)) {
            return it;
          }
          return Iter(new_extent, new_stride, it->axis);
        };
        auto new_shard = opt_tile->shard.Map(remap_iter);
        auto new_replica = opt_tile->replica.Map(remap_iter);
        if (!new_shard.same_as(opt_tile->shard) || !new_replica.same_as(opt_tile->replica)) {
          layout = TileLayout(new_shard, new_replica, opt_tile->offset);
          layout_changed = true;
        }
      }
    }

    bool storage_scope_changed = specialized_storage_scope.has_value() &&
                                 specialized_storage_scope.value() != buffer->storage_scope;
    if (buffer->elem_offset.same_as(elem_offset) && buffer->shape.same_as(shape) &&
        buffer->strides.same_as(strides) && !layout_changed && !storage_scope_changed) {
      return buffer;
    } else {
      auto n = CopyBufferType(buffer);
      n->elem_offset = std::move(elem_offset);
      n->shape = std::move(shape);
      n->strides = std::move(strides);
      if (layout_changed) {
        n->layout = std::move(layout);
      }
      if (storage_scope_changed) {
        n->storage_scope = specialized_storage_scope.value();
      }
      return RebuildBufferVar(buffer, std::move(n));
    }
  }

  Range MutateRange(const Range& range) {
    PrimExpr min = this->VisitPrimExpr(range->min);
    PrimExpr extent = this->VisitPrimExpr(range->extent);
    if (min.same_as(range->min) && extent.same_as(range->extent)) {
      return range;
    } else {
      return Range::FromMinExtent(std::move(min), std::move(extent));
    }
  }

  BufferVar MutateAllocBuffer(const BufferVar& alloc_buf) {
    TVM_FFI_ICHECK(!buffer_map_.count(alloc_buf))
        << "Multiple points of definition found for buffer " << alloc_buf;

    BufferVar buf = MutateBuffer(alloc_buf);
    buffer_map_[alloc_buf] = buf;
    return buf;
  }

  BufferVar GetNewBuffer(const BufferVar& old_buffer) {
    if (auto it = buffer_map_.find(old_buffer); it != buffer_map_.end()) {
      return it->second;
    }

    auto mutated = MutateBuffer(old_buffer);
    TVM_FFI_ICHECK(mutated.same_as(old_buffer))
        << "BufferVar " << old_buffer << " (shape = " << old_buffer->shape << ")"
        << " was used without a declaration, "
        << "and would be specialized into " << mutated << " (shape = " << mutated->shape << ").  "
        << "While usage of an undeclared buffer is currently allowed in TIR, "
        << "mutation must occur at the buffer's point of definition "
        << "(see discussion on https://github.com/apache/tvm/pull/14565 for more details).  "
        << "Please add a definition for this buffer, "
        << "either as a BufferType-annotated PrimFunc parameter, "
        << "in a tirx::SBlock's alloc_buffer, "
        << "or in a DeclBuffer statement.";

    return old_buffer;
  }

  BufferRegion MutateBufferRegion(const BufferRegion& buffer_region) {
    auto it = buffer_map_.find(buffer_region->buffer);
    const BufferVar& buffer = it != buffer_map_.end() ? it->second : buffer_region->buffer;
    ffi::Array<Range> region = buffer_region->region.Map(
        std::bind(&PrimFuncSpecializer::MutateRange, this, std::placeholders::_1));
    if (it == buffer_map_.end() && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(buffer, std::move(region));
    }
  }

 private:
  /*! \brief The vars to be substitute and their values */
  const VarMap& var_map_;
  /*! \brief map from old buffer to mutated buffer */
  std::unordered_map<BufferVar, BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_map_;
  /*! \brief Direct aliases between buffer parameters. */
  std::unordered_map<BufferVar, BufferVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_aliases_;
  /*! \brief Buffer parameters constrained by a concrete, non-parameter buffer. */
  std::unordered_set<const VarNode*> constrained_buffer_params_;
};

/*!
 * \brief Update Specialize var map with buffer matching.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_buf The matching buffer.
 * \param var_map The var mapping to be updated.
 * \note This function will match target buffer's shape, strides and element_offset
 *   For example, we define a buffer in PrimFunc:
 *   A: T.Buffer([m, n])
 *
 *   Then we match it with a buffer B =  tirx.decl_buffer((8, 16))
 *
 *   It means we have two var mappings here: m = 8 and n = 16
 *
 *   If the buffer signature is not a Var, the mapping will fail.
 *   e.g. A = T.match_buffer(a, [m * 2, n + 1])
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const BufferVar& specific_buf,
                            VarMap* var_map) {
  // preliminaries
  tirx::ExprDeepEqual equal;

  auto opt_buffer = param.as<BufferVar>();
  TVM_FFI_CHECK(opt_buffer, ValueError)
      << "specialize expects param to have a BufferType annotation";
  const BufferVar& buf_to_specialize = opt_buffer.value();

  // build var mapping using specific_buf's parameters
  auto build_var_mapping = [&](const Expr& new_expr, const Expr& old_expr) {
    auto expr_equal = [&](const Expr& lhs, const Expr& rhs) {
      if (auto lhs_prim = lhs.as<PrimExpr>()) {
        auto rhs_prim = rhs.as<PrimExpr>();
        return rhs_prim && equal(lhs_prim.value(), rhs_prim.value());
      }
      return lhs.same_as(rhs);
    };
    if (!expr_equal(new_expr, old_expr)) {
      auto maybe_var = old_expr.as<Var>();
      TVM_FFI_CHECK(maybe_var, TypeError)
          << "The signature of target buffer exprected an independent Var, but got " << old_expr
          << ".";
      const Var& var = maybe_var.value();
      auto it = var_map->find(var);
      if (it != var_map->end()) {
        TVM_FFI_CHECK(expr_equal(it->second, new_expr), ValueError)
            << "The assigned value of var " << var << " mismatched. " << it->second << " vs. "
            << new_expr << ".";
      } else {
        (*var_map)[var] = new_expr;
      }
    }
  };

  // Check buffer dimensions
  TVM_FFI_CHECK(specific_buf->shape.size() == buf_to_specialize->shape.size(), ValueError)
      << "The buffer dimensions mismatched" << buf_to_specialize->shape.size() << " vs. "
      << specific_buf->shape.size() << ".";

  TVM_FFI_CHECK(specific_buf->strides.size() == buf_to_specialize->strides.size(), ValueError)
      << "The buffer strides dimensions mismatched" << buf_to_specialize->strides.size() << " vs. "
      << specific_buf->strides.size() << ".";

  // Updating var mapping using specific_expr
  for (size_t i = 0; i < specific_buf->shape.size(); ++i) {
    build_var_mapping(specific_buf->shape[i], buf_to_specialize->shape[i]);
  }
  for (size_t i = 0; i < specific_buf->strides.size(); ++i) {
    build_var_mapping(specific_buf->strides[i], buf_to_specialize->strides[i]);
  }
  build_var_mapping(specific_buf->elem_offset, buf_to_specialize->elem_offset);
  // The specializer distinguishes a concrete buffer constraint (which keeps
  // the parameter) from another function parameter (which aliases it).
  build_var_mapping(specific_buf.var(), buf_to_specialize.var());

  // Check data_alignment and offset_factor.
  // These two signatures are int, so we do not need map them.
  TVM_FFI_CHECK_EQ(specific_buf->data_alignment, buf_to_specialize->data_alignment, ValueError)
      << "The buffer data_alignment mismatched" << buf_to_specialize->data_alignment << " vs. "
      << specific_buf->data_alignment << ".";

  TVM_FFI_CHECK_EQ(specific_buf->offset_factor, buf_to_specialize->offset_factor, ValueError)
      << "The buffer offset_factor mismatched" << buf_to_specialize->offset_factor << " vs. "
      << specific_buf->offset_factor << ".";
}

/*!
 * \brief Update Specialize var map with parameter value.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_expr The parameter value.
 * \param var_map The var mapping to be updated.
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const Expr& specific_expr,
                            VarMap* var_map) {
  // check param is in PrimFunc's parameters
  TVM_FFI_CHECK(IsParam(func, param), ValueError)
      << "Specialize expects param to be in PrimFunc's params";
  // Specialize a scalar parameter rather than a buffer parameter.
  TVM_FFI_CHECK(!param.as<BufferVar>(), ValueError)
      << "Specialize expects param to not have a BufferType annotation";
  // build var mapping using specific_expr
  (*var_map)[param] = specific_expr;
}

/**************** Implementation ****************/

PrimFunc Specialize(PrimFunc func, const ffi::Map<Var, ffi::Variant<BufferVar, Expr>>& param_map) {
  VarMap var_map;
  for (const auto& kv : param_map) {
    const Var& param = kv.first;
    const ffi::Variant<BufferVar, Expr>& instance = kv.second;
    if (auto opt_buffer = instance.as<BufferVar>()) {
      UpdateSpecializeVarMap(func, param, opt_buffer.value(), &var_map);
    } else if (auto opt_expr = instance.as<Expr>()) {
      UpdateSpecializeVarMap(func, param, opt_expr.value(), &var_map);
    } else {
      TVM_FFI_THROW(TypeError) << "specialize expected instance to be BufferVar or Expr";
    }
  }
  return PrimFuncSpecializer::Specialize(func, std::move(var_map));
}

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Specialize", [](PrimFunc func,
                                              const ffi::Map<Var, ffi::Any>& param_map) {
    VarMap var_map;
    for (const auto& [param, instance] : param_map) {
      if (auto buffer = instance.as<BufferVar>()) {
        UpdateSpecializeVarMap(func, param, buffer.value(), &var_map);
      } else if (const ExprNode* expr = instance.as<ExprNode>()) {
        UpdateSpecializeVarMap(func, param, ffi::GetRef<Expr>(expr), &var_map);
      } else if (instance.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
        UpdateSpecializeVarMap(func, param, instance.cast<PrimExpr>(), &var_map);
      } else {
        TVM_FFI_THROW(TypeError) << "specialize expected instance to be BufferVar or Expr, but got "
                                 << instance.GetTypeKey();
      }
    }
    return PrimFuncSpecializer::Specialize(func, std::move(var_map));
  });
}

}  // namespace tirx
}  // namespace tvm
