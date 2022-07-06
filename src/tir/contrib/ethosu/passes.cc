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
 * \file tir/contrib/ethosu/passes.cc
 *
 * \brief Passes used in TIR lowering for the microNPU compiler.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>

namespace tvm {

/*!
 * \brief The maximum number of movements allowed for a copy in the CopyComputeReordering pass.
 */
constexpr const char* kCopyComputeReorderingMaxCopyMovements =
    "tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements";
TVM_REGISTER_PASS_CONFIG_OPTION(kCopyComputeReorderingMaxCopyMovements, Integer);

namespace tir {
namespace contrib {
namespace ethosu {

/*!
 * \brief This mutator moves allocates to the top of the body of the main
 * function.
 *
 * Note: This pass can currently only be run in conjunction with the
 * LowerToTIR() pass as it expects a single primitive function called
 * "main" that is being offloaded to the NPU.
 *
 * For example,
 * Before:
 *   allocate {
 *       extern_call(...)
 *           allocate {
 *               extern_call(...)
 *           }
 *   }
 *
 * After:
 *   allocate {
 *       allocate {
 *           extern_call(...)
 *           extern_call(...)
 *       }
 *  }
 */
class HoistAllocatesMutator : public StmtExprMutator {
 public:
  HoistAllocatesMutator() {}

  PrimFunc operator()(PrimFunc main_func) {
    Stmt new_main_func_body = SeqStmt::Flatten(this->VisitStmt(main_func->body));

    // Write all allocates that were removed in reverse order
    for (auto it = allocates_.rbegin(); it != allocates_.rend(); it++) {
      Allocate current_alloc = *it;
      if (it != allocates_.rbegin()) {
        new_main_func_body = SeqStmt({new_main_func_body});
      }
      new_main_func_body =
          Allocate(current_alloc->buffer_var, current_alloc->dtype, current_alloc->extents,
                   current_alloc->condition, new_main_func_body, current_alloc->annotations,
                   current_alloc->span);
    }

    PrimFunc new_main_func =
        PrimFunc(main_func->params, new_main_func_body, main_func->ret_type, main_func->buffer_map,
                 main_func->preflattened_buffer_map, main_func->attrs);
    return new_main_func;
  }

 private:
  Stmt VisitStmt_(const AllocateNode* op) override {
    allocates_.push_back(GetRef<Allocate>(op));
    return VisitStmt(op->body);
  }

  /*! A stack to store allocates as they are visited. */
  std::vector<Allocate> allocates_;
};

/*!
 * \brief A pass to hoist allocate nodes to the top of the body of the main function.
 *
 * \return tvm::transform::Pass
 */
tvm::transform::Pass HoistAllocates() {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    ICHECK(mod->GetGlobalVars().size() == 1 && mod->ContainGlobalVar("main"))
        << "Expected a single primitive function called 'main'. Please run the HoistAllocates pass "
           "in conjunction with the LowerToTIR() pass.";
    return HoistAllocatesMutator()(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.contrib.ethos-u.HoistAllocates",
                                                 {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.HoistAllocates").set_body_typed(HoistAllocates);

/*!
 * \brief Reorders copy and compute nodes in such a way that independent DMA copies,
 * and computes happen in parallel.
 * Copies to buffers with local scope are not reordered, indeed they copy LUT
 * into the SHRAM which already happens in parallel with copying weights into
 * the weights encoder.
 */
class CopyComputeReorderingMutator : public StmtExprMutator {
 public:
  explicit CopyComputeReorderingMutator(int max_copy_movements)
      : _max_copy_movements{max_copy_movements} {}

  PrimFunc operator()(PrimFunc main_func) {
    if (_max_copy_movements > 0) {
      auto prim_func_node{main_func.CopyOnWrite()};
      prim_func_node->body = this->VisitStmt(main_func->body);
      return GetRef<PrimFunc>(prim_func_node);
    }
    return main_func;
  }

 private:
  Stmt VisitStmt_(const SeqStmtNode* op) override {
    if (op->size() <= 1) {
      return StmtExprMutator::VisitStmt_(op);
    }

    auto seq_stmt{GetRef<SeqStmt>(op)};
    std::vector<Stmt> new_seq(seq_stmt->size());
    std::copy(seq_stmt->seq.begin(), seq_stmt->seq.end(), new_seq.begin());

    // Each copy statement to a buffer with global scope is moved up
    // at most `_max_copy_movements` times.
    for (size_t index = 0; index < new_seq.size(); ++index) {
      if (stmt_is_global_copy(new_seq[index])) {
        int lower = std::max(0, static_cast<int>(index) - _max_copy_movements);
        for (int i = index; i > lower && !stmt_is_copy(new_seq[i - 1]); --i) {
          std::swap(new_seq[i - 1], new_seq[i]);
        }
      }
    }

    auto seq_stmt_node{CopyOnWrite(op)};
    seq_stmt_node->seq = std::move(new_seq);
    return Stmt{seq_stmt_node};
  }

  tvm::runtime::Array<tvm::PrimExpr> get_stmt_args(const Stmt& stmt) {
    Stmt eval_stmt = stmt;
    if (const auto* attr_stmt = eval_stmt.as<AttrStmtNode>()) {
      eval_stmt = attr_stmt->body;
    }

    auto eval_node{eval_stmt.as<EvaluateNode>()};
    ICHECK(eval_node) << "Expected statement to be an evaluate node, but was "
                      << eval_stmt->GetTypeKey();
    auto call_node{eval_node->value.as<CallNode>()};
    ICHECK(call_node) << "Expected expression to be a call node, but was "
                      << eval_node->value->GetTypeKey();
    return call_node->args;
  }

  bool stmt_is_copy(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    return args[0].as<StringImmNode>()->value == "ethosu_copy";
  }

  bool stmt_is_global_copy(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    return args[0].as<StringImmNode>()->value == "ethosu_copy" &&
           args[3].as<BufferLoadNode>()->buffer.scope() == "global";
  }

  /*! The maximum number of movements allowed for a copy. */
  int _max_copy_movements;
};

/*!
 * \brief A pass to reorder copy and compute nodes in such a way that independent DMA copies,
 * and computes happen in parallel.
 *
 * \param max_copy_movements: The maximum number of movements allowed for a copy.
 *  If None, the pass context option tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements
 *  is used if provided, otherwise the default value will be 1.
 * \return tvm::transform::Pass
 */
tvm::transform::Pass CopyComputeReordering(Optional<Integer> max_copy_movements) {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    ICHECK(mod->GetGlobalVars().size() == 1 && mod->ContainGlobalVar("main"))
        << "Expected a single primitive function called 'main'. Please run the "
           "CopyComputeReordering "
           "pass in conjunction with the LowerToTIR() pass.";
    auto value = max_copy_movements.value_or(
        ctx->GetConfig(kCopyComputeReorderingMaxCopyMovements, Integer(1)).value());
    return CopyComputeReorderingMutator(value.IntValue())(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tir.contrib.ethos-u.CopyComputeReordering", {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.CopyComputeReordering")
    .set_body_typed(CopyComputeReordering);

}  // namespace ethosu
}  // namespace contrib
}  // namespace tir
}  // namespace tvm
