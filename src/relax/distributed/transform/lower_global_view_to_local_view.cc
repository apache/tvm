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
 * \file tvm/relax/distributed/transform/lower_global_view_to_local_view.cc
 * \brief Pass for lowering global view TensorIR into local view
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/ccl.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../../s_tir/schedule/transform.h"
#include "utils.h"
namespace tvm {
namespace tirx {
using namespace tvm::relax::distributed;
using s_tir::ReplaceBuffer;

class DistBufferReplacer : public StmtExprMutator {
 public:
  static Stmt BufferReplace(Stmt stmt, ffi::Map<BufferVar, BufferVar> buffer_map) {
    DistBufferReplacer replacer(buffer_map);
    return replacer(stmt);
  }

 private:
  explicit DistBufferReplacer(ffi::Map<BufferVar, BufferVar> buffer_map)
      : buffer_map_(buffer_map) {}

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = StmtExprMutator::VisitStmt_(_store).as_or_throw<BufferStore>();
    if (buffer_map_.count(store->buffer)) {
      ffi::ObjectPtr<BufferStoreNode> new_store = ffi::make_object<BufferStoreNode>(*store.get());
      new_store->buffer = buffer_map_[store->buffer];
      return BufferStore(new_store);
    }
    return store;
  }

  Expr VisitExpr_(const TensorLoadNode* _load) final {
    TensorLoad load = StmtExprMutator::VisitExpr_(_load).as_or_throw<TensorLoad>();
    if (buffer_map_.count(load->source.as_or_throw<tvm::tirx::BufferVar>())) {
      return BufferLoad(buffer_map_[load->source.as_or_throw<tvm::tirx::BufferVar>()],
                        load->indices, load->span);
    }
    return load;
  }

  Stmt VisitStmt_(const SBlockNode* _block) final {
    SBlock old_block = ffi::GetRef<SBlock>(_block);
    SBlock block = StmtExprMutator::VisitStmt_(_block).as_or_throw<SBlock>();
    ffi::ObjectPtr<SBlockNode> new_block = ffi::make_object<SBlockNode>(*block.get());
    new_block->reads = ReplaceBuffer(new_block->reads, buffer_map_);
    new_block->writes = ReplaceBuffer(new_block->writes, buffer_map_);
    return SBlock(new_block);
  }

  ffi::Map<BufferVar, BufferVar> buffer_map_;
};

class DistSBlockInfoCollector : public StmtExprVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    buffer_access_indices[op->buffer].push_back(op->indices);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const TensorLoadNode* op) final {
    buffer_access_indices[op->source.as_or_throw<tvm::tirx::BufferVar>()].push_back(op->indices);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const SBlockNode* op) final {
    for (const auto& iter_var : op->iter_vars) {
      if (iter_var->iter_type == kCommReduce) {
        TVM_FFI_ICHECK(op->writes.size() == 1);
        reduce_buffer_ = op->writes[0]->buffer;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool IsReduceBufferAccess(const PrimExpr& expr) {
    if (const auto* buffer_load = expr.as<TensorLoadNode>()) {
      return buffer_load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(reduce_buffer_);
    }
    return false;
  }

  void VisitExpr_(const prim::AddNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "sum";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const prim::MulNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "prod";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const prim::MinNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "min";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const prim::MaxNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "max";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferVar reduce_buffer_;

 public:
  std::unordered_map<BufferVar, ffi::Array<ffi::Array<PrimExpr>>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      buffer_access_indices;
  std::string reduce_kind;
};

class DistributedBufferCompactor : StmtExprMutator {
  // FIXME: change to use unordered_map<int, AxisShardingSpec> (represent dim and sharding spec)
  // Currently we assume device mesh is only 1d, but when we support 2d, we need to change this
  using DimShard = std::unordered_map<int, int>;

 public:
  static std::tuple<PrimFunc, std::string> DistBufferCompact(
      const std::vector<ShardingSpec>& sharding_specs, PrimFunc prim_func) {
    prim_func = s_tir::RenewDefs(prim_func);
    DistributedBufferCompactor compactor(sharding_specs, prim_func);
    ffi::Array<Var> new_params;
    ffi::Map<BufferVar, BufferVar> replace_buffer_map;
    for (const Var& param : prim_func->params) {
      if (!param->ty.as<BufferTypeNode>()) {
        new_params.push_back(param);
        continue;
      }
      BufferVar buffer(param);
      BufferVar shard_buffer = compactor.ShardBuffer(buffer);
      new_params.push_back(shard_buffer.var());
      if (!shard_buffer.same_as(buffer)) {
        replace_buffer_map.Set(buffer, shard_buffer);
      }
    }
    Stmt new_body = compactor(prim_func->body);
    new_body = DistBufferReplacer::BufferReplace(new_body, replace_buffer_map);
    PrimFunc new_func(new_params, new_body, prim_func->ret_type, prim_func->attrs, prim_func->span);
    return std::make_tuple(new_func, compactor.add_allreduce_kind_);
  }

 private:
  DistributedBufferCompactor(const std::vector<ShardingSpec>& sharding_specs, PrimFunc prim_func)
      : sharding_specs_(sharding_specs) {
    PropagateShardingSpecOnBlock(prim_func);
  }
  // todo: if cannot propagate, insert allgather
  // todo: if reduce, insert allreduce
  void PropagateShardingSpecOnBlock(PrimFunc prim_func) {
    extractor_(prim_func->body);
    std::unordered_set<BufferAxis, BufferAxisHash> visited;
    for (int i = 0, j = 0; i < static_cast<int>(prim_func->params.size()); i++) {
      Var param_var = prim_func->params[i];
      if (!param_var->ty.as<BufferTypeNode>()) {
        continue;
      }
      BufferVar param_buffer(param_var);
      ShardingSpec spec = sharding_specs_[j++];

      for (int mesh_dim = 0; mesh_dim < static_cast<int>(spec.first->shape.size()); mesh_dim++) {
        PlacementSpec dim_placement = spec.second->dim_specs[mesh_dim];
        if (dim_placement->kind == PlacementSpecKind::kReplica) {
          continue;
        }
        std::vector<BufferAxis> buffer_axis_group;
        extractor_.DFSGraph({param_buffer, dim_placement->axis}, &visited, &buffer_axis_group);
        for (const auto& buffer_axis : buffer_axis_group) {
          buffer_shards_[buffer_axis.first][buffer_axis.second] = spec.first->shape[mesh_dim];
        }
      }
    }
  }

  ffi::Array<IterVar> ShardIterVar(
      SBlock block,
      const std::unordered_map<BufferVar, ffi::Array<ffi::Array<PrimExpr>>, ffi::ObjectPtrHash,
                               ffi::ObjectPtrEqual>& buffer_access_indices) {
    std::vector<BufferVar> buffers;
    for (const auto& read : block->reads) {
      buffers.push_back(read->buffer);
    }
    for (const auto& write : block->writes) {
      buffers.push_back(write->buffer);
    }
    ffi::Map<Var, Range> iter_var_range;
    for (const auto& iter_var : block->iter_vars) {
      iter_var_range.Set(iter_var->var, iter_var->dom);
    }
    arith::Analyzer analyzer;
    for (const auto& buffer : buffers) {
      if (buffer_access_indices.count(buffer) == 0 || buffer_shards_.count(buffer) == 0) {
        continue;
      }
      ffi::Array<ffi::Array<PrimExpr>> access_indices = buffer_access_indices.at(buffer);
      DimShard dim_shards = buffer_shards_[buffer];
      for (const auto& access_index : access_indices) {
        for (const auto& pr : dim_shards) {
          int dim = pr.first;
          int shard = pr.second;
          Var var = GetShardingVarFromIndex(access_index[dim], iter_var_range, analyzer);
          TVM_FFI_ICHECK(!iter_var_shards_.count(var) || iter_var_shards_[var] == shard)
              << "A loop cannot have different sharding";
          iter_var_shards_[var] = shard;
        }
      }
    }

    ffi::Array<IterVar> new_iter_vars;
    for (const auto& iter_var : block->iter_vars) {
      if (iter_var_shards_.count(iter_var->var)) {
        int shard = iter_var_shards_[iter_var->var];
        if (shard > 1) {
          Range dom = iter_var->dom;
          TVM_FFI_ICHECK(is_zero(dom->min));
          arith::Analyzer analyzer;
          TVM_FFI_ICHECK(analyzer->CanProve(floormod(dom->extent, shard) == 0));
          new_iter_vars.push_back(
              IterVar(Range::FromMinExtent(dom->min, floordiv(dom->extent, shard)), iter_var->var,
                      iter_var->iter_type, iter_var->thread_tag));
          continue;
        }
      }
      new_iter_vars.push_back(iter_var);
    }
    return new_iter_vars;
  }

  BufferVar ShardBuffer(BufferVar buffer) {
    if (buffer_shards_.count(buffer) == 0) {
      return buffer;
    }
    DimShard dim_shards = buffer_shards_[buffer];
    ffi::Array<PrimExpr> shape;
    for (int i = 0; i < static_cast<int>(buffer->shape.size()); i++) {
      if (dim_shards.count(i)) {
        shape.push_back(floordiv(buffer->shape[i], dim_shards[i]));
      } else {
        shape.push_back(buffer->shape[i]);
      }
    }
    BufferType new_type(buffer->storage_scope, buffer->dtype, std::move(shape), buffer->strides,
                        buffer->elem_offset, buffer->data_alignment, buffer->offset_factor,
                        buffer->layout, buffer->allocated_addr);
    return BufferVar(buffer.name(), std::move(new_type), buffer.span());
  }

  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = StmtExprMutator::VisitStmt_(op).as_or_throw<SBlock>();
    DistSBlockInfoCollector collector;
    collector(block);
    ffi::Array<IterVar> new_iter_vars = ShardIterVar(block, collector.buffer_access_indices);
    ffi::Array<BufferVar> new_alloc_buffers;
    ffi::Map<BufferVar, BufferVar> buffer_map;
    for (const BufferVar& buffer : block->alloc_buffers) {
      BufferVar sharded_buffer = ShardBuffer(buffer);
      if (!sharded_buffer.same_as(buffer)) {
        buffer_map.Set(buffer, sharded_buffer);
      }
      new_alloc_buffers.push_back(sharded_buffer);
    }
    // condition for adding allreduce:
    // sharding on reduction axis
    for (const IterVar& iter_var : new_iter_vars) {
      if (iter_var->iter_type == kCommReduce && iter_var_shards_.count(iter_var->var)) {
        TVM_FFI_ICHECK(add_allreduce_kind_ == "");
        AddAllReduceBlock(collector.reduce_kind);
        break;
      }
    }
    ffi::ObjectPtr<SBlockNode> new_block = ffi::make_object<SBlockNode>(*block.operator->());
    new_block->iter_vars = new_iter_vars;
    new_block->alloc_buffers = new_alloc_buffers;
    if (new_block->name_hint == "root") {
      new_block->alloc_buffers.insert(new_block->alloc_buffers.end(),
                                      allocated_buffer_under_root.begin(),
                                      allocated_buffer_under_root.end());
    }
    new_block->body = DistBufferReplacer::BufferReplace(block->body, buffer_map);
    return SBlock(new_block);
  }

  void AddAllReduceBlock(std::string reduce_kind) { add_allreduce_kind_ = reduce_kind; }

  Stmt VisitStmt_(const SBlockRealizeNode* op) final {
    SBlockRealize realize = StmtExprMutator::VisitStmt_(op).as_or_throw<SBlockRealize>();

    for (int i = 0; i < static_cast<int>(realize->iter_values.size()); i++) {
      PrimExpr iter_value = realize->iter_values[i];
      IterVar iter_var = realize->block->iter_vars[i];
      if (!iter_var_shards_.count(iter_var->var)) {
        continue;
      }
      auto loop_var = iter_value.as<PrimVar>();
      TVM_FFI_ICHECK(loop_var);
      loop_var_shards_[loop_var.value()] = iter_var_shards_[iter_var->var];
    }
    return realize;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For new_loop = StmtExprMutator::VisitStmt_(op).as_or_throw<For>();
    if (loop_var_shards_.count(op->loop_var)) {
      int shard = loop_var_shards_[op->loop_var];
      if (shard > 1) {
        arith::Analyzer analyzer;
        TVM_FFI_ICHECK(analyzer->CanProve(floormod(new_loop->extent, shard) == 0));
        new_loop.CopyOnWrite()->extent = floordiv(new_loop->extent, shard);
        return new_loop;
      }
    }
    return new_loop;
  }

  std::unordered_map<Var, int> iter_var_shards_;
  std::unordered_map<Var, int> loop_var_shards_;
  ffi::Array<BufferVar> allocated_buffer_under_root;
  BufferAxisGraphExtractor extractor_;
  std::vector<ShardingSpec> sharding_specs_;
  std::unordered_map<BufferVar, DimShard, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_shards_;
  std::string add_allreduce_kind_;
};

}  // namespace tirx
}  // namespace tvm

namespace tvm {
namespace relax {
namespace distributed {

class LowerTIRToLocalView : public ExprMutator {
 public:
  explicit LowerTIRToLocalView(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr || !IsDistIRFunc(ffi::GetRef<Function>(func_))) {
        continue;
      }
      Expr new_func_body = this->VisitExpr(func_->body);
      ffi::ObjectPtr<FunctionNode> new_func = ffi::make_object<FunctionNode>(*func_);
      new_func->body = new_func_body;
      builder_->UpdateFunction(gv, Function(new_func));
    }
    return builder_->GetContextIRModule();
  }

 private:
  inline ffi::Array<DTensorType> ExtractDTensorType(Var var) {
    if (const auto* dtensor_ty = GetTypeAs<DTensorTypeNode>(var)) {
      return {ffi::GetRef<DTensorType>(dtensor_ty)};
    } else if (const auto* tuple_ty = GetTypeAs<TupleTypeNode>(var)) {
      ffi::Array<DTensorType> ret;
      for (const auto& field : tuple_ty->fields) {
        ret.push_back(field.as_or_throw<DTensorType>());
      }
      return ret;
    } else {
      TVM_FFI_THROW(InternalError)
          << "The output of a call_tir should be a DTensorType or TupleType";
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (!val->op.same_as(call_tir_op)) {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    std::vector<ShardingSpec> sharding_specs;
    ffi::Array<Expr> args = val->args[1].as_or_throw<Tuple>()->fields;
    GlobalVar gvar = val->args[0].as_or_throw<GlobalVar>();
    tirx::PrimFunc prim_func = MatchPrimFunc(builder_->GetContextIRModule(), gvar).value();
    TVM_FFI_ICHECK_LE(args.size(), prim_func->params.size());
    for (size_t i = 0; i < args.size(); ++i) {
      const Expr& arg = args[i];
      const tirx::Var& param = prim_func->params[i];
      if (param->ty.as<tirx::BufferTypeNode>()) {
        const auto* ty = GetTypeAs<DTensorTypeNode>(arg);
        TVM_FFI_CHECK(ty, TypeError)
            << "Expected buffer parameter " << param << " to receive a distributed tensor, but "
            << arg << " has type " << GetType(arg);
        sharding_specs.push_back(ShardingSpec(ty->device_mesh, ty->placement));
      } else {
        TVM_FFI_CHECK(arg.as<PrimExpr>(), TypeError)
            << "Expected scalar parameter " << param
            << " to receive an individual primitive expression, but " << arg << " has type "
            << GetType(arg);
      }
    }
    Var output_var = binding->var;
    ffi::Array<DTensorType> output_tys = ExtractDTensorType(output_var);
    for (const auto& ty : output_tys) {
      sharding_specs.push_back(ShardingSpec(ty->device_mesh, ty->placement));
    }
    tirx::PrimFunc new_prim_func;
    std::string allreduce_kind;
    std::tie(new_prim_func, allreduce_kind) =
        tirx::DistributedBufferCompactor::DistBufferCompact(sharding_specs, prim_func);
    auto new_gvar = builder_->AddFunction(new_prim_func, gvar->name_hint);
    Call call = this->VisitExpr(binding->value).as_or_throw<Call>();
    ffi::ObjectPtr<CallNode> new_call_node = ffi::make_object<CallNode>(*call.get());
    new_call_node->op = Op::Get("relax.dist.call_tir_local_view");
    new_call_node->args.Set(0, new_gvar);
    Call new_call(new_call_node);
    if (allreduce_kind != "") {
      ffi::ObjectPtr<AllReduceAttrs> attrs = ffi::make_object<AllReduceAttrs>();
      attrs->op_type = allreduce_kind;
      new_call =
          Call(Type::Missing(), Op::Get("relax.ccl.allreduce"), {new_call}, Attrs(attrs), {});
    }
    ReEmitBinding(binding, this->builder_->Normalize(new_call));
  }
};

namespace transform {

Pass LowerGlobalViewToLocalView() {
  auto pass_func = [=](IRModule m, PassContext pc) { return LowerTIRToLocalView(m).Lower(); };
  return CreateModulePass(pass_func, 1, "LowerGlobalViewToLocalView", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.distributed.transform.LowerGlobalViewToLocalView",
                        LowerGlobalViewToLocalView);
}
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
