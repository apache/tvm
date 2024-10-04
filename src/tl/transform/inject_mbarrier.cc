// /*
//  * Licensed to the Apache Software Foundation (ASF) under one
//  * or more contributor license agreements.  See the NOTICE file
//  * distributed with this work for additional information
//  * regarding copyright ownership. The ASF licenses this file
//  * to you under the Apache License, Version 2.0 (the
//  * "License"); you may not use this file except in compliance
//  * with the License.  You may obtain a copy of the License at
//  *
//  *   http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing,
//  * software distributed under the License is distributed on an
//  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//  * KIND, either express or implied.  See the License for the
//  * specific language governing permissions and limitations
//  * under the License.
//  */

// /*!
//  * \file warp_specialized_pipeline.cc
//  * \brief Warp specialized Pipeline for cuda GPU (sm90+)
//  */

// #include <tvm/tir/analysis.h>
// #include <tvm/tir/builtin.h>
// #include <tvm/tir/op.h>
// #include <tvm/tir/stmt_functor.h>
// #include <tvm/tir/transform.h>

// #include "../op/builtin.h"

// namespace tvm {
// namespace tl {

// using namespace tir;

// enum class Role { kConsumer, kProducer, kBoth };

// class WarpSpecializedRoleMarker : public StmtVisitor {
//  public:
//   WarpSpecializedRoleMarker(Map<Var, Buffer> buffer_data_to_buffer)
//       : buffer_data_to_buffer_(buffer_data_to_buffer) {}

//   Role GetRole(const StmtNode* stmt) const {
//     auto it = map_.find(stmt);
//     ICHECK(it != map_.end());
//     return it->second;
//   }

//   Role GetRole(const Stmt& stmt) const { return GetRole(stmt.get()); }

//   void VisitStmt_(const EvaluateNode* op) final {
//     Role role = Role::kConsumer;
//     if (auto call = op->value.as<CallNode>()) {
//       if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
//         role = Role::kProducer;
//         has_bulk_copy_ = true;
//       }
//     }
//     SetRole(op, role);
//   }

//   void VisitStmt_(const BufferStoreNode* op) final {
//     bool is_shared_store = op->buffer.scope() == "shared.dyn" || op->buffer.scope() == "shared";
//     if (!is_shared_store) {
//       SetRole(op, Role::kConsumer);
//       return;
//     }

//     // Check reads from global
//     Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
//                 /*body*/ GetRef<Stmt>(op));
//     auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
//     auto reads = access[0];
//     Role role = Role::kProducer;
//     for (auto read : reads) {
//       if (read->buffer.scope() != "global") {
//         role = Role::kConsumer;
//         break;
//       }
//     }
//     if (role == Role::kProducer) has_simt_copy_ = true;
//     SetRole(op, role);
//   }

//   void VisitStmt_(const SeqStmtNode* op) final {
//     StmtVisitor::VisitStmt_(op);
//     auto role = GetRole(op->seq[0]);
//     for (auto stmt : op->seq) {
//       if (role != GetRole(stmt)) {
//         role = Role::kBoth;
//         break;
//       }
//     }
//     SetRole(op, role);
//   }

//   void VisitStmt_(const IfThenElseNode* op) final {
//     StmtVisitor::VisitStmt_(op);
//     auto role = GetRole(op->then_case);
//     if (op->else_case.defined()) {
//       auto role_else = GetRole(op->else_case.value());
//       if (role != role_else) role = Role::kBoth;
//     }
//     SetRole(op, role);
//   }

//   void VisitStmt_(const BlockRealizeNode* op) final {
//     StmtVisitor::VisitStmt_(op);
//     SetRole(op, GetRole(op->block));
//   }

//   template <class NodeType>
//   void HandleBodyStmt(const NodeType* op) {
//     StmtVisitor::VisitStmt_(op);
//     SetRole(op, GetRole(op->body));
//   }

//   void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
//   void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
//   void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
//   void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
//   void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }

//   bool HasProducer() { return has_simt_copy_ || has_bulk_copy_; }

//   bool HasSimtCopy() { return has_simt_copy_; }

//  private:
//   void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }
//   Map<Var, Buffer> buffer_data_to_buffer_;
//   std::unordered_map<const StmtNode*, Role> map_;
//   bool has_simt_copy_ = false;
//   bool has_bulk_copy_ = false;
// };

// static PrimExpr makeGetBarrier(PrimExpr barrier_id) {
//   return Call(DataType::Handle(), GetMBarrierOp(), {barrier_id});
// }

// static Stmt makeExpectTX(PrimExpr barrier_id, PrimExpr bytes) {
//   auto call = Call(DataType::Handle(), MBarrierExpectTX(), {makeGetBarrier(barrier_id), bytes});
//   return Evaluate(call);
// }

// static Stmt makeArriveBarrier(PrimExpr barrier_id) {
//   auto call = Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {makeGetBarrier(barrier_id)});
//   return Evaluate(call);
// }

// static Stmt makeCpAsyncBarrier(PrimExpr barrier_id) {
//   auto call =
//       Call(DataType::Handle(), builtin::ptx_cp_async_barrier(), {makeGetBarrier(barrier_id)});
//   return Evaluate(call);
// }

// static Stmt makeParityWait(PrimExpr barrier_id, PrimExpr parity) {
//   auto call = Call(DataType::Handle(), MBarrierWaitParity(), {makeGetBarrier(barrier_id), parity});
//   return Evaluate(call);
// }

// class ProducerTraitsCollector : public StmtExprVisitor {
//  public:
//   ProducerTraitsCollector() { Clear(); }

//   void Clear() {
//     bulk_copy_bytes = 0;
//     loop_extents = 1;
//     has_simt_copy = false;
//   }

//   void Collect(Stmt stmt) { VisitStmt(stmt); }

//   bool HasSimtCopy() { return has_simt_copy; }

//   PrimExpr BulkCopyBytes() { return bulk_copy_bytes; }

//  private:
//   void VisitExpr_(const CallNode* call) final {
//     if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
//       Call access_ptr = Downcast<Call>(call->args[2]);
//       ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
//       int type_bytes = access_ptr->args[0]->dtype.bytes();
//       bulk_copy_bytes += access_ptr->args[3] * loop_extents * type_bytes;
//     }
//     StmtExprVisitor::VisitExpr_(call);
//   }

//   void VisitStmt_(const ForNode* op) final {
//     PrimExpr old_loop_evtents = loop_extents;
//     loop_extents *= op->extent;
//     StmtExprVisitor::VisitStmt_(op);
//     loop_extents = old_loop_evtents;
//   }

//   void VisitExpr_(const BufferLoadNode* op) final {
//     has_simt_copy = true;
//     StmtExprVisitor::VisitExpr_(op);
//   }

//   bool has_simt_copy;
//   PrimExpr bulk_copy_bytes;
//   PrimExpr loop_extents;
// };

// // Rewrite the producer Stmt to use the correct barrier index
// class MbarrierRewriter : public StmtExprMutator {
//  public:
//   static Stmt Rewrite(Stmt stmt, PrimExpr barrier_id) {
//     MbarrierRewriter rewriter;
//     rewriter.producer_barrier_idx_ = barrier_id;
//     return rewriter(stmt);
//   }

//  private:
//   PrimExpr VisitExpr_(const CallNode* op) final {
//     auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
//     if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
//       Call access_ptr = Downcast<Call>(call->args[2]);
//       ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
//       call.CopyOnWrite()->args.Set(1, makeGetBarrier(producer_barrier_idx_));
//     }
//     return call;
//   }
//   PrimExpr producer_barrier_idx_;
// };


// using namespace tir::transform;

// tvm::transform::Pass InjectMbarrier() {
//   auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
//     return WarpSpecializedRewriter::Substitute(f);
//   };
//   return CreatePrimFuncPass(pass_func, 0, "tl.WarpSpecialized", {});
// }

// TVM_REGISTER_GLOBAL("tl.InjectMbarrier").set_body_typed(InjectMbarrier);

// }  // namespace tl
// }  // namespace tvm
