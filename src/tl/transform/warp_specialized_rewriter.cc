/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

enum class Role { kConsumer, kProducer, kBoth };

class WarpSpecializedRoleMarker : public StmtVisitor {
 public:
  WarpSpecializedRoleMarker(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  Role GetRole(const StmtNode* stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end());
    return it->second;
  }

  Role GetRole(const Stmt& stmt) const { return GetRole(stmt.get()); }

  void VisitStmt_(const EvaluateNode* op) final {
    Role role = Role::kConsumer;
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
        role = Role::kProducer;
        has_bulk_copy_ = true;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    bool is_shared_store = op->buffer.scope() == "shared.dyn" || op->buffer.scope() == "shared";
    if (!is_shared_store) {
      SetRole(op, Role::kConsumer);
      return;
    }

    // Check reads from global
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ GetRef<Stmt>(op));
    auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto reads = access[0];
    Role role = Role::kProducer;
    for (auto read : reads) {
      if (read->buffer.scope() != "global") {
        role = Role::kConsumer;
        break;
      }
    }
    if (role == Role::kProducer) has_simt_copy_ = true;
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (auto stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined()) {
      auto role_else = GetRole(op->else_case.value());
      if (role != role_else) role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType>
  void HandleBodyStmt(const NodeType* op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }

  bool HasProducer() { return has_simt_copy_ || has_bulk_copy_; }

  bool HasSimtCopy() { return has_simt_copy_; }

 private:
  void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const StmtNode*, Role> map_;
  bool has_simt_copy_ = false;
  bool has_bulk_copy_ = false;
};

static PrimExpr makeGetBarrier(PrimExpr barrier_id) {
  return Call(DataType::Handle(), GetMBarrierOp(), {barrier_id});
}

static Stmt makeExpectTX(PrimExpr barrier_id, PrimExpr bytes) {
  auto call = Call(DataType::Handle(), MBarrierExpectTX(), {makeGetBarrier(barrier_id), bytes});
  return Evaluate(call);
}

static Stmt makeArriveBarrier(PrimExpr barrier_id) {
  auto call = Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {makeGetBarrier(barrier_id)});
  return Evaluate(call);
}

static Stmt makeCpAsyncBarrier(PrimExpr barrier_id) {
  auto call =
      Call(DataType::Handle(), builtin::ptx_cp_async_barrier(), {makeGetBarrier(barrier_id)});
  return Evaluate(call);
}

static Stmt makeParityWait(PrimExpr barrier_id, PrimExpr parity) {
  auto call = Call(DataType::Handle(), MBarrierWaitParity(), {makeGetBarrier(barrier_id), parity});
  return Evaluate(call);
}

// static bool isGemm(Stmt stmt) {
//   bool is_gemm = false;
//   if (stmt.as<EvaluateNode>()) {
//     auto call = Downcast<Evaluate>(stmt)->value.as<CallNode>();
//     if (call && call->op.same_as(Op::Get("tir.call_extern"))) {
//       if (call->args[0].as<StringImmNode>()) {
//         std::string name = Downcast<StringImm>(call->args[0])->value;
//         if (name.find("gemm") != std::string::npos) {
//           is_gemm = true;
//         }
//       }
//     }
//   }
//   return is_gemm;
// }

class ProducerTraitsCollector : public StmtExprVisitor {
 public:
  ProducerTraitsCollector() { Clear(); }

  void Clear() {
    bulk_copy_bytes = 0;
    loop_extents = 1;
    has_simt_copy = false;
  }

  void Collect(Stmt stmt) { VisitStmt(stmt); }

  bool HasSimtCopy() { return has_simt_copy; }

  PrimExpr BulkCopyBytes() { return bulk_copy_bytes; }

 private:
  void VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
      Call access_ptr = Downcast<Call>(call->args[2]);
      ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
      int type_bytes = access_ptr->args[0]->dtype.bytes();
      bulk_copy_bytes += access_ptr->args[3] * loop_extents * type_bytes;
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void VisitStmt_(const ForNode* op) final {
    PrimExpr old_loop_evtents = loop_extents;
    loop_extents *= op->extent;
    StmtExprVisitor::VisitStmt_(op);
    loop_extents = old_loop_evtents;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    has_simt_copy = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  bool has_simt_copy;
  PrimExpr bulk_copy_bytes;
  PrimExpr loop_extents;
};

// Rewrite the producer Stmt to use the correct barrier index
class MbarrierRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(Stmt stmt, PrimExpr barrier_id) {
    MbarrierRewriter rewriter;
    rewriter.producer_barrier_idx_ = barrier_id;
    return rewriter(stmt);
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
      Call access_ptr = Downcast<Call>(call->args[2]);
      ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
      call.CopyOnWrite()->args.Set(1, makeGetBarrier(producer_barrier_idx_));
    }
    return call;
  }
  PrimExpr producer_barrier_idx_;
};


class ThreadIdxRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced) {
    auto rewriter = ThreadIdxRewriter(thread_var, replaced);
    return rewriter(stmt);
  }

 private:
  ThreadIdxRewriter(Var thread_var, PrimExpr replaced)
      : thread_var_(thread_var), replaced_(replaced) {}

  PrimExpr VisitExpr_(const VarNode* var) final {
    if (var == thread_var_.get()) {
      return replaced_;
    } else {
      return StmtExprMutator::VisitExpr_(var);
    }
  }

  Var thread_var_;
  PrimExpr replaced_;
};

Block MakeGroupBlock(const Stmt& stmt, const Map<String, ObjectRef>& annotations) {
  Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"", /*body*/ stmt,
              /*init=*/{}, /*alloc_buffers=*/{}, /*match_buffers=*/{}, /*annotations=*/annotations);
  return block;
}

struct OpInfo {
  int group_size, order, stage;
  std::vector<int> group;
};
struct PipelineInfo {
  std::vector<OpInfo> op_infos;

  PipelineInfo() = default;
  PipelineInfo(
    Array<Array<Integer>> group_info,
    Array<Integer> order_info,
    Array<Integer> stage_info
  ) {
    int n = static_cast<int>(group_info.size());
    ICHECK(n == static_cast<int>(order_info.size()));
    ICHECK(n == static_cast<int>(stage_info.size()));
    // int cur_id = 0;
    for (int i = 0; i < n; i++) {
      OpInfo op_info;
      op_info.group_size = group_info[i].size();
      for (int j = 0; j < op_info.group_size; j++) {
        op_info.group.push_back(group_info[i][j].as<IntImmNode>()->value);
      }
      op_info.order = order_info[i].as<IntImmNode>()->value;
      op_info.stage = stage_info[i].as<IntImmNode>()->value;
      op_infos.push_back(op_info);
    }
  }

  PipelineInfo(const PipelineInfo& other) {
    for (auto op_info : other.op_infos) {
      op_infos.push_back(op_info);
    }
  }

  std::pair<int, int> FindStmt(int stmt_idx) {
    for (size_t i = 0; i < op_infos.size(); i++) {
      for (size_t j = 0; j < op_infos[i].group.size(); j++) {
        if (op_infos[i].group[j] == stmt_idx) {
          return std::make_pair(i, j);
        }
      }
    }
    return std::make_pair(-1, -1);
  }

  void UpdateOrder(int order) {
    for (int i = 0; i < static_cast<int>(op_infos.size()); i++) {
      if (op_infos[i].order >= order && op_infos[i].order > 0) {
        op_infos[i].order++;
      }
    }
  }

  int SplitOp(int stmt_idx) {
    auto pair = FindStmt(stmt_idx);
    int op_idx = pair.first;
    int inner_idx = pair.second;
    ICHECK(op_idx != -1);
    ICHECK(inner_idx != -1);
    OpInfo half0;
    OpInfo half1;
    // The order to do sync
    int sync_order = op_infos[op_idx].order + 1;
    UpdateOrder(sync_order);

    half0.group_size = inner_idx + 1;
    half0.order = op_infos[op_idx].order;
    half0.stage = op_infos[op_idx].stage;
    for (int i = 0; i <= inner_idx; i++) {
      half0.group.push_back(op_infos[op_idx].group[i]);
    }
    half1.group_size = op_infos[op_idx].group_size - inner_idx - 1;
    half1.order = op_infos[op_idx].order + 2;
    half1.stage = op_infos[op_idx].stage;
    for (int i = inner_idx + 1; i < op_infos[op_idx].group_size; i++) {
      half1.group.push_back(op_infos[op_idx].group[i]);
    }
    op_infos.erase(op_infos.begin() + op_idx);
    if (half0.group_size > 0) {
      op_infos.insert(op_infos.begin() + op_idx, half0);
    }
    if (half1.group_size > 0) {
      UpdateOrder(half1.order);
      op_infos.insert(op_infos.begin() + op_idx + 1, half1);
    }
    return sync_order;
  }

  void PrintPipelineInfo() {
    std::cout << "Print op_infos:" << std::endl;
    for (size_t i = 0; i < op_infos.size(); i++) {
      std::cout << i << " " << op_infos[i].group_size << " " << op_infos[i].order << " " << op_infos[i].stage << std::endl;
    }
    std::cout << "End of print" << std::endl;
  }
};

class GroupOpRewriter : public StmtExprMutator {
 public:
  GroupOpRewriter(PipelineInfo pipeline_info) : pipeline_info_(pipeline_info) {}

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    Map<String, ObjectRef> annotations;
    annotations.Set(String("stmt_group"), Integer(1));
    auto original_node = (op->body).as<SeqStmtNode>();
    if (!original_node) {
      return GetRef<For>(op);
    }
    Array<Stmt> new_body;
    int cur_id = 0;
    for (int i = 0; i < static_cast<int>(pipeline_info_.op_infos.size()); i++) {
      if (pipeline_info_.op_infos[i].group_size == 0) continue;
      Array<Stmt> block_stmt;
      for (int j = 0; j < static_cast<int>(pipeline_info_.op_infos[i].group_size); j++) {
        // ICHECK(group_info_[i][j].as<IntImmNode>());
        // int index = static_cast<int>(group_info_[i][j].as<IntImmNode>()->value);
        ICHECK(original_node->seq[cur_id].as<BlockNode>());
        auto block = original_node->seq[cur_id].as<BlockNode>();
        // TODO: handle nested seqstmt
        block_stmt.push_back(block->body);
        cur_id++;
      }
      new_body.push_back(
        MakeGroupBlock(block_stmt.size() == 1 ? block_stmt[0] : SeqStmt(std::move(block_stmt)), annotations));
    }
    Array<Integer> order_anno;
    Array<Integer> stage_anno;
    for (auto op_info : pipeline_info_.op_infos) {
      order_anno.push_back(Integer(op_info.order));
      stage_anno.push_back(Integer(op_info.stage));
    }
    Map<String, ObjectRef> for_annotations = op->annotations;
    for_annotations.erase("tl_pipeline_group");
    for_annotations.Set("software_pipeline_order", order_anno);
    for_annotations.Set("software_pipeline_stage", stage_anno);
    For new_for = For(op->loop_var, op->min, op->extent, op->kind, new_body.size() == 1 ? new_body[0] : SeqStmt(std::move(new_body)), op->thread_binding, for_annotations);
    return new_for;
  }

  PipelineInfo pipeline_info_;
};
class WSCodeEmitter : public StmtMutator {
 public:
  WSCodeEmitter(bool is_emitting_producer, IterVar thread_iv,
                Map<Var, Buffer> buffer_data_to_buffer, const WarpSpecializedRoleMarker& marker)
      : is_emitting_producer_(is_emitting_producer),
        buffer_data_to_buffer_(buffer_data_to_buffer),
        marker_(marker),
        thread_var_(thread_iv->var) {}

 private:
  template <typename NodeType>
  Stmt FilterByRole(const NodeType* op) {
    Role role = marker_.GetRole(op);
    if (role == Role::kBoth)
      return StmtMutator::VisitStmt_(op);
    else if ((role == Role::kProducer) == is_emitting_producer_)
      return GetRef<Stmt>(op);
    else
      return Evaluate(0);
  }

  // TODO: only need to add block for ops in the loop
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    bool has_producer = false;
    for (auto stmt : op->seq) {
      if (marker_.GetRole(stmt) == Role::kProducer) {
        has_producer = true;
        break;
      }
    }
    bool need_producer_sync = has_producer && marker_.GetRole(op) == Role::kBoth;
    if (!need_producer_sync) return FilterByRole(op);

    auto seq_transformed = op->seq.Map([&](Stmt stmt) { return VisitStmt(stmt); });

    auto map = ExtractSyncPattern(op->seq);
    // std::cout << "Print ExtractSyncPattern" << std::endl;
    // for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
    //   std::cout << i << " " << map.acquire[i] << " " << map.release[i] << " " << map.release_after[i] << std::endl;
    // }
    // std::cout << "Print sync pattern" << std::endl;
    // for (auto pattern : map.patterns) {
    //   std::cout << pattern.release_idx << " " << pattern.acquire_idx << std::endl;
    // }
    // std::cout << "End of ExtractSyncPattern" << std::endl;
    // pipeline_info_.PrintPipelineInfo();
    Array<Stmt> new_body;
    Map<String, ObjectRef> annotations;
    annotations.Set(String("stmt_group"), Integer(1));

    if (is_emitting_producer_) {  // producer case
      ProducerTraitsCollector collector;
      for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
        Array<Stmt> block_stmt = {};
        if (marker_.GetRole(op->seq[i]) == Role::kConsumer) continue;
        if (marker_.GetRole(op->seq[i]) == Role::kBoth) {
          block_stmt.push_back(seq_transformed[i]);
          new_body.push_back(MakeGroupBlock(block_stmt.size() == 1 ? block_stmt[0] : SeqStmt(std::move(block_stmt)), annotations));
          continue;
        }
        if (map.acquire[i] != -1) {
          PrimExpr acquire_barrier_id = stage_ + num_barriers_ + num_stages_ * map.acquire[i];
          PrimExpr parity =
              map.is_loop_dependency(map.acquire[i]) ? bitwise_xor(parity_, 1) : parity_;
          block_stmt.push_back(makeParityWait(acquire_barrier_id, parity));
        }
        ICHECK(map.release[i] >= 0);
        PrimExpr release_barrier_id = stage_ + num_barriers_ + num_stages_ * map.release[i];
        auto stmt = MbarrierRewriter::Rewrite(seq_transformed[i], release_barrier_id);
        collector.Collect(stmt);
        if (!is_zero(collector.BulkCopyBytes())) {
          auto expect_tx = IfThenElse(EQ(thread_var_, 0),
                                      makeExpectTX(release_barrier_id, collector.BulkCopyBytes()));
          block_stmt.push_back(expect_tx);
        }
        block_stmt.push_back(stmt);
        if (collector.HasSimtCopy() > 0) {
          block_stmt.push_back(makeCpAsyncBarrier(release_barrier_id));
        }
        if (map.release_after[i]) {
          block_stmt.push_back(makeArriveBarrier(release_barrier_id));
          for (int j = 0; j < num_stages_; j++) {
            released_barrier_.insert(j + num_barriers_ + num_stages_ * map.release[i]);
          }
        }
        collector.Clear();
        new_body.push_back(MakeGroupBlock(block_stmt.size() == 1 ? block_stmt[0] : SeqStmt(std::move(block_stmt)), annotations));
      }
    } else {  // consumer case
      for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
        Array<Stmt> block_stmt = {};
        if (marker_.GetRole(op->seq[i]) == Role::kProducer) continue;
        if (map.acquire[i] != -1) {
          PrimExpr acquire_barrier_id = stage_ + num_barriers_ + num_stages_ * map.acquire[i];
          PrimExpr parity =
              map.is_loop_dependency(map.acquire[i]) ? bitwise_xor(parity_, 1) : parity_;
          block_stmt.push_back(makeParityWait(acquire_barrier_id, parity));
        }
        block_stmt.push_back(seq_transformed[i]);
        // new_body.push_back(MakeGroupBlock(block_stmt.size() == 1 ? block_stmt[0] : SeqStmt(std::move(block_stmt)), annotations));
        if (map.release_after[i]) {
          PrimExpr release_barrier_id = stage_ + num_barriers_ + num_stages_ * map.release[i];
          block_stmt.push_back(makeArriveBarrier(release_barrier_id));
          for (int j = 0; j < num_stages_; j++) {
            released_barrier_.insert(j + num_barriers_ + num_stages_ * map.release[i]);
          }
          // Update the pipeline info
          // Todo: handle sync
        }
        new_body.push_back(MakeGroupBlock(block_stmt.size() == 1 ? block_stmt[0] : SeqStmt(std::move(block_stmt)), annotations));
      }
      // Filter out the producer stmts
      int cur_id = 0;
      PipelineInfo new_pipeline_info;
      for (int i = 0; i < static_cast<int>(pipeline_info_.op_infos.size()); i++) {
        auto op_info = pipeline_info_.op_infos[i];
        bool is_producer = false;
        for (int j = 0; j < op_info.group_size; j++) {
          if (marker_.GetRole(op->seq[cur_id]) == Role::kProducer) {
            is_producer = true;
          }
          cur_id++;
        }
        if (is_producer) {
          ICHECK(op_info.group_size == 1);
        } else {
          new_pipeline_info.op_infos.push_back(op_info);
        }
      }
      pipeline_info_ = new_pipeline_info;
    }

    num_barriers_ += map.patterns.size() * num_stages_;

    ICHECK(new_body.size() > 0);
    return new_body.size() == 1 ? new_body[0] : SeqStmt(std::move(new_body));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    int num_stages = 1;
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (num_stages_anno.defined()) {
      ICHECK(num_stages_anno.as<IntImmNode>());
      num_stages = static_cast<int>(num_stages_anno.as<IntImmNode>()->value);
      ICHECK(num_stages_ == 1) << "Nested pipeline not supported.";
    }

    Array<Array<Integer>> group_info_array;
    Array<Integer> order_info_array;
    Array<Integer> stage_info_array;
   
    auto group_anno = op->annotations.Get("tl_pipeline_group");
    if (group_anno.defined()) {
      group_info_array = Downcast<Array<Array<Integer>>>(group_anno);
    }
    auto order_anno = op->annotations.Get("tl_pipeline_order");
    if (order_anno.defined()) {
      order_info_array = Downcast<Array<Integer>>(order_anno);
    }
    auto stage_anno = op->annotations.Get("tl_pipeline_stage");
    if (stage_anno.defined()) {
      stage_info_array = Downcast<Array<Integer>>(stage_anno);
    }

    PipelineInfo pipeline_info(group_info_array, order_info_array, stage_info_array);
    if (pipeline_info.op_infos.size() > 0) {
      ICHECK(pipeline_info_.op_infos.size() == 0) << "Nested pipeline not supported.";
    }

    PrimExpr parity_before = std::move(parity_);
    PrimExpr stage_before = std::move(stage_);
    int num_stages_before = num_stages_;
    PipelineInfo pipeline_info_before = pipeline_info_;

    num_stages_ = num_stages;
    pipeline_info_ = pipeline_info;
    stage_ = FloorMod(op->loop_var - op->min, num_stages);
    parity_ =
        FloorMod(parity_before * op->extent + FloorDiv(op->loop_var - op->min, num_stages), 2);

    auto result = FilterByRole(op);

    Stmt grouped_for_node;
    if (result.as<ForNode>() && group_anno.defined() && group_info_array.size() > 0 && !is_emitting_producer_) {
      GroupOpRewriter group_op_rewriter(pipeline_info_);
      auto for_node = Downcast<For>(result);
      grouped_for_node = group_op_rewriter(for_node);
    }

    parity_ = std::move(parity_before);
    stage_ = std::move(stage_before);
    num_stages_ = num_stages_before;
    pipeline_info_ = pipeline_info_before;

    // remove pipeline annotation
    auto for_node = result.as<For>();
    if (result.as<ForNode>()) {
      auto for_node = Downcast<For>(result);
      for_node.CopyOnWrite()->annotations.erase("num_stages");
      if (is_emitting_producer_ || group_info_array.size() == 0) {
        for_node.CopyOnWrite()->annotations.erase("tl_pipeline_order");
        for_node.CopyOnWrite()->annotations.erase("tl_pipeline_stage");
      }
      if (is_emitting_producer_ || !group_anno.defined() ||group_info_array.size() == 0) {
        return for_node;
      }
      return grouped_for_node;
    }
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const EvaluateNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const AttrStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const BufferStoreNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const LetStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const AssertStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK(0);
    return Stmt();
  }
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    ICHECK(0);
    return Stmt();
  }

  struct SyncPattern {
    int release_idx, acquire_idx;
  };

  struct SyncPatternMap {
    std::vector<int> acquire;
    std::vector<int> release;
    std::vector<bool> release_after;
    std::vector<SyncPattern> patterns;
    bool is_loop_dependency(int i) {
      // return if the acquire is based on release in the previous iteration
      return patterns[i].release_idx > patterns[i].acquire_idx;
    }
  };

  std::vector<SyncPattern> CreateBaseSyncPairs(Array<Stmt> seq_stmt,
                                               const std::vector<bool>& is_producer) {
    const int n = seq_stmt.size();
    std::vector<std::set<const BufferNode*>> reads, writes;
    reads.reserve(n);
    writes.reserve(n);
    for (int i = 0; i < n; i++) {
      Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                  /*body*/ seq_stmt[i]);
      auto access = GetBlockAccessRegion(block, buffer_data_to_buffer_);
      std::set<const BufferNode*> read_set, write_set;
      for (auto region : access[0]) read_set.insert(region->buffer.get());
      for (auto region : access[1]) write_set.insert(region->buffer.get());
      reads.push_back(std::move(read_set));
      writes.push_back(std::move(write_set));
    }

    auto intersect_fn = [](const std::set<const BufferNode*>& lhs,
                           const std::set<const BufferNode*>& rhs) {
      for (auto ptr : lhs)
        if (rhs.count(ptr)) return true;
      return false;
    };

    std::vector<SyncPattern> sync_patterns;
    // producer_release consumer_acquire,
    // inject before the first consumer stmt for each producer
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (is_producer[i] != is_producer[j] &&
            (intersect_fn(writes[i], reads[j]) || intersect_fn(reads[i], writes[j]))) {
          sync_patterns.push_back({i, j});
          break;
        }
      }
    }

    // consumer_release producer_acquire
    // valid when is_loop is true
    // inject before the earlest producer stmt for each consumer
    bool in_loop = !is_zero(parity_);
    if (in_loop) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          if (is_producer[i] != is_producer[j] &&
              (intersect_fn(writes[i], reads[j]) || intersect_fn(reads[i], writes[j]))) {
            sync_patterns.push_back({i, j});
            break;
          }
        }
      }
    }

    return sync_patterns;
  }

  static std::vector<SyncPattern> RemoveUnusedSyncPatterns(
      const std::vector<SyncPattern>& sync_patterns, const std::vector<bool>& is_producer) {
    /*
      Simplify multiple release-acquire pairs into one
      ------------------
        Produce(A)
        Produce(B)
        Consume(A, B)
      ------------------
      [(0, 2), (1, 2), (2, 0)] -> [(1, 2), (2, 0)]

      Or
      ------------------
        Produce(A, B)
        Consume(A)
        Consume(B)
      ------------------
      [(0, 1), (1, 0), (2, 0)] -> [(0, 1), (2, 0)]
    */
    int M = sync_patterns.size();
    std::vector<bool> removed(M, false);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++) {
        if (is_producer[sync_patterns[i].acquire_idx] ==
                is_producer[sync_patterns[j].acquire_idx] &&
            sync_patterns[i].acquire_idx >= sync_patterns[j].acquire_idx &&
            sync_patterns[i].release_idx < sync_patterns[j].release_idx)
          removed[i] = true;
      }
    }

    std::vector<SyncPattern> sync_pattern_cleaned;
    sync_pattern_cleaned.reserve(M);
    for (int i = 0; i < M; i++)
      if (!removed[i]) sync_pattern_cleaned.push_back(sync_patterns[i]);

    return sync_pattern_cleaned;
  }

  SyncPatternMap ExtractSyncPattern(Array<Stmt> seq_stmt) {
    size_t num_stmts = seq_stmt.size();
    std::vector<bool> is_producer;
    is_producer.reserve(num_stmts);
    for (auto stmt : seq_stmt) {
      is_producer.push_back(marker_.GetRole(stmt) == Role::kProducer);
    }

    auto sync_patterns_base = CreateBaseSyncPairs(seq_stmt, is_producer);
    auto sync_patterns = RemoveUnusedSyncPatterns(sync_patterns_base, is_producer);

    // for (auto pattern : sync_patterns) {
    //   std::cout << pattern.release_idx << " " << pattern.acquire_idx << std::endl;
    // }

    SyncPatternMap map;
    map.patterns = sync_patterns;
    map.acquire.resize(num_stmts, -1);
    map.release.resize(num_stmts, -1);
    map.release_after.resize(num_stmts, false);
    for (size_t i = 0; i < sync_patterns.size(); i++) {
      map.acquire[sync_patterns[i].acquire_idx] = i;
      map.release[sync_patterns[i].release_idx] = i;
      map.release_after[sync_patterns[i].release_idx] = true;
    }

    int cur_consumer_barrier = -1, cur_producer_barrier = -1;
    for (int i = num_stmts - 1; i >= 0; i--) {
      if (is_producer[i]) {
        if (map.release[i] == -1) {
          map.release[i] = cur_producer_barrier;
        } else {
          cur_producer_barrier = map.release[i];
        }
      } else {
        if (map.release[i] == -1) {
          map.release[i] = cur_consumer_barrier;
        } else {
          cur_consumer_barrier = map.release[i];
        }
      }
    }
    return map;
  }

  const bool is_emitting_producer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<int> released_barrier_;
  const WarpSpecializedRoleMarker& marker_;

  int num_barriers_ = 0;
  PrimExpr parity_ = 0;
  PrimExpr stage_ = 0;
  int num_stages_ = 1;
  Var thread_var_;
  PipelineInfo pipeline_info_;
  friend class WarpSpecializedRewriter;
};

class WarpSpecializedRewriter : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    auto T = WarpSpecializedRewriter();
    T.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : T.buffer_lca_) T.buffer_data_to_buffer_.Set(buffer->data, buffer);
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (updated_thread_extent_.defined()) {
        thread_iv_.CopyOnWrite()->dom = {0, updated_thread_extent_.value()};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_.value();
      }
      thread_iv_ = {};
      return attr_stmt;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  // If users define a thread binding, we will replace the thread binding with threadIdx.x
  // We require the thread binding is threadIdx.x, and the extent is the same as the thread extent
  Stmt VisitStmt_(const ForNode* op) final {
    ICHECK(thread_iv_.defined());
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (for_node->kind == ForKind::kThreadBinding) {
      ICHECK(for_node->thread_binding.defined());
      String thread_tag = for_node->thread_binding.value()->thread_tag;
      ICHECK(thread_tag == "threadIdx.x") << "Only support threadIdx.x";
      Var thread_iv = Downcast<Var>(for_node->loop_var);
      Stmt new_body = ThreadIdxRewriter::Rewrite(for_node->body, thread_iv, thread_iv_);
      return new_body;
    }
    return for_node;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize block_realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    if (!thread_iv_.defined()) {
      return block_realize;
    }
    ICHECK(!updated_thread_extent_.defined());

    Block block = block_realize->block;
    WarpSpecializedRoleMarker marker(buffer_data_to_buffer_);
    marker(block);
    if (!marker.HasProducer()) {
      // Cannot detect any producer here, directly return.
      return block_realize;
    }

    WSCodeEmitter producer(true, thread_iv_, buffer_data_to_buffer_, marker);
    WSCodeEmitter consumer(false, thread_iv_, buffer_data_to_buffer_, marker);
    Stmt producer_code = producer(block->body);
    Stmt consumer_code = consumer(block->body);

    PrimExpr consumer_thread_extent = thread_iv_->dom->extent;
    PrimExpr producer_thread_extent = thread_iv_->dom->extent;
    // Need one warp-group for bulk-copy only case
    if (!marker.HasSimtCopy()) producer_thread_extent = 128;

    // TODO: estimate the correct reg usage.
    auto inc_reg_stmt = Evaluate(Call(DataType::Handle(), SetMaxNReg(), {240, 1}));
    auto dec_reg_stmt = Evaluate(Call(DataType::Handle(), SetMaxNReg(), {24, 0}));

    producer_code = SeqStmt({dec_reg_stmt, producer_code});
    consumer_code = SeqStmt({inc_reg_stmt, consumer_code});

    producer_code = ThreadIdxRewriter::Rewrite(producer_code, thread_iv_->var,
                                               thread_iv_->var - consumer_thread_extent);
    updated_thread_extent_ = consumer_thread_extent + producer_thread_extent;

    ICHECK(producer.num_barriers_ == consumer.num_barriers_)
        << producer.num_barriers_ << " " << consumer.num_barriers_;
    int num_barriers = consumer.num_barriers_;
    Array<PrimExpr> barrier_num_threads;
    barrier_num_threads.reserve(num_barriers);
    for (int i = 0; i < num_barriers; i++) {
      PrimExpr arrive_thread_count =
          producer.released_barrier_.count(i) ? producer_thread_extent : consumer_thread_extent;
      barrier_num_threads.push_back(arrive_thread_count);
    }

    Stmt init_barrier =
        Evaluate(Call(DataType::Handle(), CreateListofMBarrierOp(), barrier_num_threads));
    Stmt body =
        IfThenElse(GE(thread_iv_->var, consumer_thread_extent), producer_code, consumer_code);
    // Add an attr here to handle the partial thread count in THreadSync pass.
    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_thread_extent),
                                  Downcast<IntImm>(consumer_thread_extent)};
    body = AttrStmt(ws_partition, "kWarpSpecializationScope", 0, body);

    block.CopyOnWrite()->body = SeqStmt({init_barrier, body});
    block_realize.CopyOnWrite()->block = block;
    return block_realize;
  }

  WarpSpecializedRewriter() = default;

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Buffer, Buffer> buffer_remap_;
  IterVar thread_iv_;
  Optional<PrimExpr> updated_thread_extent_;
};

using namespace tir::transform;

tvm::transform::Pass WarpSpecialized() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return WarpSpecializedRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.WarpSpecialized", {});
}

TVM_REGISTER_GLOBAL("tl.WarpSpecialized").set_body_typed(WarpSpecialized);

}  // namespace tl
}  // namespace tvm
