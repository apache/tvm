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
#include <tvm/relax/attrs/ccl.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../tir/schedule/transform.h"
#include "utils.h"
namespace tvm {
namespace tir {
using namespace tvm::relax::distributed;

class DistBufferReplacer : public StmtExprMutator {
 public:
  static Stmt BufferReplace(Stmt stmt, Map<Buffer, Buffer> buffer_map) {
    DistBufferReplacer replacer(buffer_map);
    return replacer(stmt);
  }

 private:
  explicit DistBufferReplacer(Map<Buffer, Buffer> buffer_map) : buffer_map_(buffer_map) {}

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (buffer_map_.count(store->buffer)) {
      ObjectPtr<BufferStoreNode> new_store = make_object<BufferStoreNode>(*store.get());
      new_store->buffer = buffer_map_[store->buffer];
      return BufferStore(new_store);
    }
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (buffer_map_.count(load->buffer)) {
      ObjectPtr<BufferLoadNode> new_load = make_object<BufferLoadNode>(*load.get());
      new_load->buffer = buffer_map_[load->buffer];
      return BufferLoad(new_load);
    }
    return load;
  }

  Stmt VisitStmt_(const BlockNode* _block) final {
    Block old_block = GetRef<Block>(_block);
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_block));
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block.get());
    new_block->reads = ReplaceBuffer(new_block->reads, buffer_map_);
    new_block->writes = ReplaceBuffer(new_block->writes, buffer_map_);
    return Block(new_block);
  }

  Map<Buffer, Buffer> buffer_map_;
};

class DistBlockInfoCollector : public StmtExprVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    buffer_access_indices[op->buffer].push_back(op->indices);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    buffer_access_indices[op->buffer].push_back(op->indices);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const auto& iter_var : op->iter_vars) {
      if (iter_var->iter_type == kCommReduce) {
        ICHECK(op->writes.size() == 1);
        reduce_buffer_ = op->writes[0]->buffer;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool IsReduceBufferAccess(const PrimExpr& expr) {
    if (const auto* buffer_load = expr.as<BufferLoadNode>()) {
      return buffer_load->buffer.same_as(reduce_buffer_);
    }
    return false;
  }

  void VisitExpr_(const AddNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "sum";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "prod";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MinNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "min";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MaxNode* op) final {
    if (IsReduceBufferAccess(op->a) || IsReduceBufferAccess(op->b)) {
      reduce_kind = "max";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Buffer reduce_buffer_;

 public:
  std::unordered_map<Buffer, Array<Array<PrimExpr>>, ObjectPtrHash, ObjectPtrEqual>
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
    prim_func = RenewDefs(prim_func);
    DistributedBufferCompactor compactor(sharding_specs, prim_func);
    Map<Var, Buffer> new_func_buffer_map;
    Map<Buffer, Buffer> replace_buffer_map;
    for (const auto& pr : prim_func->buffer_map) {
      Buffer shard_buffer = compactor.ShardBuffer(pr.second);
      new_func_buffer_map.Set(pr.first, shard_buffer);
      if (!shard_buffer.same_as(pr.second)) {
        replace_buffer_map.Set(pr.second, shard_buffer);
      }
    }
    Stmt new_body = compactor(prim_func->body);
    new_body = DistBufferReplacer::BufferReplace(new_body, replace_buffer_map);
    ObjectPtr<PrimFuncNode> new_func = make_object<PrimFuncNode>(*prim_func.get());
    new_func->buffer_map = new_func_buffer_map;
    new_func->body = new_body;
    return std::make_tuple(PrimFunc(new_func), compactor.add_allreduce_kind_);
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
      if (!prim_func->buffer_map.count(param_var)) {
        continue;
      }
      Buffer param_buffer = prim_func->buffer_map[param_var];
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

  Array<IterVar> ShardIterVar(
      Block block,
      const std::unordered_map<Buffer, Array<Array<PrimExpr>>, ObjectPtrHash, ObjectPtrEqual>&
          buffer_access_indices) {
    std::vector<Buffer> buffers;
    for (const auto& read : block->reads) {
      buffers.push_back(read->buffer);
    }
    for (const auto& write : block->writes) {
      buffers.push_back(write->buffer);
    }
    Map<Var, Range> iter_var_range;
    for (const auto& iter_var : block->iter_vars) {
      iter_var_range.Set(iter_var->var, iter_var->dom);
    }
    arith::Analyzer analyzer;
    for (const auto& buffer : buffers) {
      if (buffer_access_indices.count(buffer) == 0 || buffer_shards_.count(buffer) == 0) {
        continue;
      }
      Array<Array<PrimExpr>> access_indices = buffer_access_indices.at(buffer);
      DimShard dim_shards = buffer_shards_[buffer];
      for (const auto& access_index : access_indices) {
        for (const auto& pr : dim_shards) {
          int dim = pr.first;
          int shard = pr.second;
          Var var = GetShardingVarFromIndex(access_index[dim], iter_var_range, &analyzer);
          ICHECK(!iter_var_shards_.count(var) || iter_var_shards_[var] == shard)
              << "A loop cannot have different sharding";
          iter_var_shards_[var] = shard;
        }
      }
    }

    Array<IterVar> new_iter_vars;
    for (const auto& iter_var : block->iter_vars) {
      if (iter_var_shards_.count(iter_var->var)) {
        int shard = iter_var_shards_[iter_var->var];
        if (shard > 1) {
          Range dom = iter_var->dom;
          ICHECK(is_zero(dom->min));
          arith::Analyzer analyzer;
          ICHECK(analyzer.CanProve(floormod(dom->extent, shard) == 0));
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

  Buffer ShardBuffer(Buffer buffer) {
    if (buffer_shards_.count(buffer) == 0) {
      return buffer;
    }
    DimShard dim_shards = buffer_shards_[buffer];
    Array<PrimExpr> shape;
    for (int i = 0; i < static_cast<int>(buffer->shape.size()); i++) {
      if (dim_shards.count(i)) {
        shape.push_back(floordiv(buffer->shape[i], dim_shards[i]));
      } else {
        shape.push_back(buffer->shape[i]);
      }
    }
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
    new_buffer->shape = shape;
    return Buffer(new_buffer);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    DistBlockInfoCollector collector;
    collector(block);
    Array<IterVar> new_iter_vars = ShardIterVar(block, collector.buffer_access_indices);
    Array<Buffer> new_alloc_buffers;
    Map<Buffer, Buffer> buffer_map;
    for (const Buffer& buffer : block->alloc_buffers) {
      Buffer sharded_buffer = ShardBuffer(buffer);
      if (!sharded_buffer.same_as(buffer)) {
        buffer_map.Set(buffer, sharded_buffer);
      }
      new_alloc_buffers.push_back(sharded_buffer);
    }
    // condition for adding allreduce:
    // sharding on reduction axis
    for (const IterVar& iter_var : new_iter_vars) {
      if (iter_var->iter_type == kCommReduce && iter_var_shards_.count(iter_var->var)) {
        ICHECK(add_allreduce_kind_ == "");
        AddAllReduceBlock(collector.reduce_kind);
        break;
      }
    }
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block.operator->());
    new_block->iter_vars = new_iter_vars;
    new_block->alloc_buffers = new_alloc_buffers;
    if (new_block->name_hint == "root") {
      new_block->alloc_buffers.insert(new_block->alloc_buffers.end(),
                                      allocated_buffer_under_root.begin(),
                                      allocated_buffer_under_root.end());
    }
    new_block->body = DistBufferReplacer::BufferReplace(block->body, buffer_map);
    return Block(new_block);
  }

  void AddAllReduceBlock(std::string reduce_kind) { add_allreduce_kind_ = reduce_kind; }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));

    for (int i = 0; i < static_cast<int>(realize->iter_values.size()); i++) {
      PrimExpr iter_value = realize->iter_values[i];
      IterVar iter_var = realize->block->iter_vars[i];
      if (!iter_var_shards_.count(iter_var->var)) {
        continue;
      }
      ICHECK(iter_value.as<VarNode>());
      loop_var_shards_[Downcast<Var>(iter_value)] = iter_var_shards_[iter_var->var];
    }
    return realize;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For new_loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (loop_var_shards_.count(op->loop_var)) {
      int shard = loop_var_shards_[op->loop_var];
      if (shard > 1) {
        arith::Analyzer analyzer;
        ICHECK(analyzer.CanProve(floormod(new_loop->extent, shard) == 0));
        return For(new_loop->loop_var, new_loop->min, floordiv(new_loop->extent, shard),
                   new_loop->kind, new_loop->body, new_loop->thread_binding, new_loop->annotations);
      }
    }
    return new_loop;
  }

  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> iter_var_shards_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> loop_var_shards_;
  Array<Buffer> allocated_buffer_under_root;
  BufferAxisGraphExtractor extractor_;
  std::vector<ShardingSpec> sharding_specs_;
  std::unordered_map<Buffer, DimShard, ObjectPtrHash, ObjectPtrEqual> buffer_shards_;
  std::string add_allreduce_kind_;
};

}  // namespace tir
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
      if (func_ == nullptr || !IsDistIRFunc(GetRef<Function>(func_))) {
        continue;
      }
      Expr new_func_body = this->VisitExpr(func_->body);
      ObjectPtr<FunctionNode> new_func = make_object<FunctionNode>(*func_);
      new_func->body = new_func_body;
      builder_->UpdateFunction(gv, Function(new_func));
    }
    return builder_->GetContextIRModule();
  }

 private:
  inline Array<DTensorStructInfo> ExtractDTensorStructInfo(Var var) {
    if (const auto* dtensor_sinfo = GetStructInfoAs<DTensorStructInfoNode>(var)) {
      return {GetRef<DTensorStructInfo>(dtensor_sinfo)};
    } else if (const auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(var)) {
      Array<DTensorStructInfo> ret;
      for (const auto& field : tuple_sinfo->fields) {
        ret.push_back(Downcast<DTensorStructInfo>(field));
      }
      return ret;
    } else {
      LOG(FATAL) << "The output of a call_tir should be a DTensorStructInfo or TupleStructInfo";
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (!val->op.same_as(call_tir_op)) {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    std::vector<ShardingSpec> sharding_specs;
    Array<Expr> args = Downcast<Tuple>(val->args[1])->fields;
    for (const auto& arg : args) {
      const auto* sinfo = GetStructInfoAs<DTensorStructInfoNode>(arg);
      ICHECK(sinfo);
      sharding_specs.push_back(ShardingSpec(sinfo->device_mesh, sinfo->placement));
    }
    Var output_var = binding->var;
    Array<DTensorStructInfo> output_sinfos = ExtractDTensorStructInfo(output_var);
    for (const auto& sinfo : output_sinfos) {
      sharding_specs.push_back(ShardingSpec(sinfo->device_mesh, sinfo->placement));
    }
    GlobalVar gvar = Downcast<GlobalVar>(val->args[0]);
    tir::PrimFunc prim_func = MatchPrimFunc(builder_->GetContextIRModule(), gvar).value();
    tir::PrimFunc new_prim_func;
    std::string allreduce_kind;
    std::tie(new_prim_func, allreduce_kind) =
        tir::DistributedBufferCompactor::DistBufferCompact(sharding_specs, prim_func);
    auto new_gvar = builder_->AddFunction(new_prim_func, gvar->name_hint);
    Call call = Downcast<Call>(this->VisitExpr(binding->value));
    ObjectPtr<CallNode> new_call_node = make_object<CallNode>(*call.get());
    new_call_node->op = Op::Get("relax.dist.call_tir_local_view");
    new_call_node->args.Set(0, new_gvar);
    Call new_call(new_call_node);
    if (allreduce_kind != "") {
      ObjectPtr<AllReduceAttrs> attrs = make_object<AllReduceAttrs>();
      attrs->op_type = allreduce_kind;
      new_call = Call(Op::Get("relax.ccl.allreduce"), {new_call}, Attrs(attrs), {});
    }
    ReEmitBinding(binding, this->builder_->Normalize(new_call));
  }
};

namespace transform {

Pass LowerGlobalViewToLocalView() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return LowerTIRToLocalView(m).Lower(); };
  return CreateModulePass(pass_func, 1, "LowerGlobalViewToLocalView", {});
}
TVM_REGISTER_GLOBAL("relax.distributed.transform.LowerGlobalViewToLocalView")
    .set_body_typed(LowerGlobalViewToLocalView);
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
