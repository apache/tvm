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
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "../../transforms/ir_utils.h"

namespace tvm {

/*!
 * \brief The maximum number of movements allowed for a copy in the CopyComputeReordering pass.
 */
constexpr const char* kCopyComputeReorderingMaxCopyMovements =
    "tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements";
TVM_REGISTER_PASS_CONFIG_OPTION(kCopyComputeReorderingMaxCopyMovements, Integer);

/*!
 * \brief Whether to reorder copies and computes based on cycle count.
 */
constexpr const char* kCopyComputeReorderingReorderByCycles =
    "tir.contrib.ethos-u.copy_compute_reordering_reorder_by_cycles";
TVM_REGISTER_PASS_CONFIG_OPTION(kCopyComputeReorderingReorderByCycles, Bool);

namespace tir {
namespace contrib {
namespace ethosu {

namespace {

struct FlattenUnwrapResult {
  std::vector<Stmt> seq;
  std::vector<Stmt> rewrap_nest;
};

/*! \brief Utility function to flatten SeqStmt
 *
 * An AttrStmt or DeclBuffer may internally contain SeqStmt nodes that
 * we want to flatten.  Unlike SeqStmt::Flatten, this function unwraps
 * these node types when encountered.
 *
 * \param stmt The tir::Stmt to be flattened.
 * \return The flattened statements
 */
FlattenUnwrapResult FlattenUnwrap(const Stmt& stmt) {
  std::vector<Stmt> seq_stmt;
  std::vector<Stmt> rewrap_nest;
  std::function<void(const Stmt&)> flatten_unwrap = [&](const Stmt& stmt) {
    if (auto* ptr = stmt.as<DeclBufferNode>()) {
      rewrap_nest.push_back(DeclBuffer(ptr->buffer, Evaluate(0)));
      flatten_unwrap(ptr->body);
    } else if (auto* ptr = stmt.as<SeqStmtNode>()) {
      for (const auto& sub_stmt : ptr->seq) {
        flatten_unwrap(sub_stmt);
      }
    } else if (auto* ptr = stmt.as<EvaluateNode>(); ptr && ptr->value.as<IntImmNode>()) {
      // Skip
    } else {
      seq_stmt.push_back(stmt);
    }
  };
  flatten_unwrap(stmt);
  return FlattenUnwrapResult{seq_stmt, rewrap_nest};
}

/*! Returns the arguments of the given statement */
Array<PrimExpr> GetStmtArgs(Stmt stmt) {
  while (auto* ptr = stmt.as<DeclBufferNode>()) {
    stmt = ptr->body;
  }

  auto attr{stmt.as<AttrStmtNode>()};
  Stmt eval_stmt{attr ? attr->body : stmt};
  auto eval{eval_stmt.as<EvaluateNode>()};
  ICHECK(eval) << "Expected statement to be an evaluate node, but was " << eval_stmt->GetTypeKey();
  auto call{eval->value.as<CallNode>()};
  ICHECK(call) << "Expected expression to be a call node, but was " << eval->value->GetTypeKey();
  return call->args;
}

enum class StmtType { global_copy, local_copy, compute };

/*! Returns the type of the given statement */
StmtType GetStmtType(const Stmt& stmt) {
  Array<PrimExpr> args{GetStmtArgs(stmt)};
  if (args[0].as<StringImmNode>()->value == "ethosu_copy") {
    if (args[3].as<BufferLoadNode>()->buffer.scope() == "global") {
      return StmtType::global_copy;
    } else {
      return StmtType::local_copy;
    }
  }
  return StmtType::compute;
}
/*! Returns the buffer read my the given copy statement */
Buffer GetCopyReadBuffer(const Stmt& stmt) {
  Array<PrimExpr> args{GetStmtArgs(stmt)};
  return args[1].as<BufferLoadNode>()->buffer;
}

/*! Returns the buffer written my the given copy statement */
Buffer GetCopyWriteBuffer(const Stmt& stmt) {
  Array<PrimExpr> args{GetStmtArgs(stmt)};
  return args[3].as<BufferLoadNode>()->buffer;
}

/*! Returns the length of the given copy statement */
int64_t GetCopyLength(const Stmt& stmt) {
  Array<PrimExpr> args{GetStmtArgs(stmt)};
  return args[2].as<IntImmNode>()->value;
}

/*! Returns the cycles of the given statement */
int64_t GetStmtCycles(const Stmt& stmt) {
  auto attr{stmt.as<AttrStmtNode>()};
  if (attr && attr->attr_key == "pragma_compute_cycles_hint") {
    int64_t cycles{Downcast<Integer>(attr->value)->value};
    return cycles;
  }
  return 0;
}
}  // namespace

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
        new_main_func_body = SeqStmt::Flatten(new_main_func_body);
      }
      new_main_func_body =
          Allocate(current_alloc->buffer_var, current_alloc->dtype, current_alloc->extents,
                   current_alloc->condition, new_main_func_body, current_alloc->annotations,
                   current_alloc->span);
    }

    PrimFunc new_main_func = PrimFunc(main_func->params, new_main_func_body, main_func->ret_type,
                                      main_func->buffer_map, main_func->attrs);
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
 * \brief Reorders copy and compute nodes in such a way that independent DMA copies
 * and computes happen in parallel.
 * Copies to buffers with local scope are not reordered since they copy LUT
 * into the SHRAM and that already happens in parallel with copying weights into
 * the weights encoder.
 */
class CopyComputeReorderingMutator : public StmtExprMutator {
 public:
  explicit CopyComputeReorderingMutator(int max_copy_movements, bool reorder_by_cycles)
      : _max_copy_movements{max_copy_movements}, _reorder_by_cycles{reorder_by_cycles} {}

  PrimFunc operator()(PrimFunc main_func) {
    if (_max_copy_movements > 0) {
      auto prim_func_node{main_func.CopyOnWrite()};
      prim_func_node->body = this->VisitStmt(main_func->body);
      return GetRef<PrimFunc>(prim_func_node);
    }
    return main_func;
  }

 private:
  // A structure to hold a compute op with the corresponding weights/bias copy and LUT copy
  struct OpWithCopies {
    Stmt compute_op{};
    Stmt global_copy{};
    Stmt local_copy{};
  };

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    auto [seq, rewrap_nest] = FlattenUnwrap(GetRef<Stmt>(op));

    if (seq.size() <= 1) {
      return StmtExprMutator::VisitStmt_(op);
    }

    std::vector<Stmt> new_seq(seq.begin(), seq.end());

    // Reorder the copies and computes based on the cycle count
    if (_reorder_by_cycles) {
      // We can't hide the first copy, so ignore it for the purpose of hiding copies
      Stmt first_copy{};
      if (stmt_is_global_copy(new_seq[0]) ||
          (stmt_is_local_copy(new_seq[0]) && stmt_is_global_copy(new_seq[1]))) {
        auto copy_position = stmt_is_global_copy(new_seq[0]) ? 0 : 1;
        first_copy = new_seq[copy_position];
        new_seq.erase(new_seq.begin() + copy_position);
      }

      // Build up a list of cells with the compute op and the copy ops that directly preceed it
      std::vector<OpWithCopies> ops{};
      for (size_t idx = 0; idx < new_seq.size(); ++idx) {
        if (stmt_is_compute_op(new_seq[idx])) {
          OpWithCopies new_op;
          new_op.compute_op = new_seq[idx];
          if (idx > 0) {
            auto prev_op = new_seq[idx - 1];
            if (!stmt_is_compute_op(prev_op)) {
              if (stmt_is_local_copy(prev_op)) {
                new_op.local_copy = prev_op;
              } else {
                new_op.global_copy = prev_op;
              }
              if (idx > 1) {
                auto prev_prev_op = new_seq[idx - 2];
                if (!stmt_is_compute_op(prev_prev_op)) {
                  if (stmt_is_local_copy(prev_prev_op)) {
                    new_op.local_copy = prev_prev_op;
                  } else {
                    new_op.global_copy = prev_prev_op;
                  }
                }
              }
            }
          }
          ops.push_back(new_op);
        }
      }

      // Move the global copies up by one. If in general the computes take longer than the copies,
      // that should be good enough
      for (size_t idx = 1; idx < ops.size(); ++idx) {
        if (ops[idx].global_copy.as<AttrStmtNode>()) {
          ops[idx - 1].global_copy = ops[idx].global_copy;
          ops[idx].global_copy = {};
        }
      }

      // If there are long copies, try to hide them further
      for (size_t idx = ops.size() - 1; idx > 0; --idx) {
        if (ops[idx].global_copy.as<AttrStmtNode>()) {
          // Check whether the copy is hidden
          int64_t copy_cycles{GetStmtCycles(ops[idx].global_copy)};
          int64_t compute_cycles{GetStmtCycles(ops[idx].compute_op)};
          bool is_hidden = compute_cycles >= copy_cycles;

          // If the previous compute op is not already hiding another copy, move the copy back, so
          // that it would be hidden by multiple computes
          while (!is_hidden && !ops[idx - 1].global_copy.as<AttrStmtNode>() && (idx > 0)) {
            int64_t new_compute_cycles{GetStmtCycles(ops[idx - 1].compute_op)};
            ops[idx - 1].global_copy = ops[idx].global_copy;
            ops[idx].global_copy = {};
            compute_cycles += new_compute_cycles;
            is_hidden = compute_cycles >= copy_cycles;
            --idx;
          }
        }
      }

      // Reconstruct the op sequence from the vector of OpWithCopies
      new_seq.clear();
      if (first_copy.as<AttrStmtNode>()) {
        new_seq.push_back(first_copy);
      }
      for (auto& op : ops) {
        if (op.global_copy.as<AttrStmtNode>()) {
          new_seq.push_back(op.global_copy);
        }
        if (op.local_copy.as<EvaluateNode>()) {
          new_seq.push_back(op.local_copy);
        }
        if (op.compute_op.as<AttrStmtNode>()) {
          new_seq.push_back(op.compute_op);
        }
      }
    } else {
      // Each copy statement to a buffer with global scope is moved up
      // at most `_max_copy_movements` times.
      for (size_t index = 0; index < new_seq.size(); ++index) {
        if (GetStmtType(new_seq[index]) == StmtType::global_copy) {
          int lower = std::max(0, static_cast<int>(index) - _max_copy_movements);
          for (int i = index; i > lower && (GetStmtType(new_seq[i - 1]) == StmtType::compute);
               --i) {
            std::swap(new_seq[i - 1], new_seq[i]);
          }
        }
      }
    }

    return MergeNest(rewrap_nest, SeqStmt::Flatten(new_seq));
  }

  bool stmt_is_global_copy(const Stmt& stmt) { return GetStmtType(stmt) == StmtType::global_copy; }

  bool stmt_is_local_copy(const Stmt& stmt) { return GetStmtType(stmt) == StmtType::local_copy; }

  bool stmt_is_compute_op(const Stmt& stmt) { return GetStmtType(stmt) == StmtType::compute; }

  /*! The maximum number of movements allowed for a copy. */
  int _max_copy_movements;
  /*! Whether we use the cycle hint to determine the reordering. */
  bool _reorder_by_cycles;
};

/*!
 * \brief A pass to reorder copy and compute nodes in such a way that independent DMA copies
 * and computes happen in parallel. If reorder_by_cycles is set, we will ignore the
 * max_copy_movements value.
 *
 *  \param max_copy_movements: The maximum number of movements allowed for a copy.
 *  If None, the pass context option tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements
 *  is used if provided, otherwise the default value will be 1.
 *
 * \param reorder_by_cycles: Whether to reorder copies and computes by cycles.
 *  If None, the pass context option tir.contrib.ethos-u.copy_compute_reordering_reorder_by_cycles
 *  is used if provided, otherwise the default value will be False. If the value is True,
 *  max_copy_movements will be ignored.
 * \return tvm::transform::Pass
 */
tvm::transform::Pass CopyComputeReordering(Optional<Integer> max_copy_movements,
                                           Optional<Bool> reorder_by_cycles) {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    ICHECK(mod->GetGlobalVars().size() == 1 && mod->ContainGlobalVar("main"))
        << "Expected a single primitive function called 'main'. Please run the "
           "CopyComputeReordering "
           "pass in conjunction with the LowerToTIR() pass.";

    auto copy_movements = max_copy_movements.value_or(
        ctx->GetConfig(kCopyComputeReorderingMaxCopyMovements, Integer(1)).value());
    auto reorder = reorder_by_cycles.value_or(
        ctx->GetConfig(kCopyComputeReorderingReorderByCycles, Bool(false)).value());
    return CopyComputeReorderingMutator(copy_movements.IntValue(), reorder)(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tir.contrib.ethos-u.CopyComputeReordering", {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.CopyComputeReordering")
    .set_body_typed(CopyComputeReordering);

/*!
 * \brief This mutator removes all allocates.
 */
class RemoveAllocatesMutator : public StmtExprMutator {
 public:
  PrimFunc operator()(PrimFunc main_func) {
    auto prim_func_node{main_func.CopyOnWrite()};
    prim_func_node->body = this->VisitStmt(main_func->body);
    return GetRef<PrimFunc>(prim_func_node);
  }

 private:
  Stmt VisitStmt_(const AllocateNode* op) override { return VisitStmt(op->body); }
};

/*!
 * \brief This extractor collects information used by the MergeConstantsMutator
 */
class MergeConstantsInfoExtractor : public StmtExprVisitor {
 public:
  class Info {
   public:
    /*! A stack to store allocates as they are visited. */
    std::vector<Allocate> allocates{};

    /*! A list that contains in the i-th position the write buffer of the i-th statement
     * if that statement is a copy to a buffer with global scope  */
    std::vector<Optional<Buffer>> copy_write_buffers{};

    /*! Maps a copy's write buffer to an index representing the
     * new buffer and an offset in that buffer */
    std::unordered_map<const BufferNode*, std::pair<int /* new buffer index */, int /* offset */>>
        old_to_new_write_buffer{};

    /*! Maps an index representing a new buffer to the length of that buffer */
    std::unordered_map<int /* new buffer index */, int /* length */> new_buffers_length{};

    /*! Maps an index representing a new buffer to the cycless needed to copy that buffer */
    std::unordered_map<int /* new buffer index */, int64_t> cycless{};
  };

  Info operator()(PrimFunc main_func) {
    this->VisitStmt(main_func->body);
    return std::move(_info);
  }

 private:
  /*! The information collected by this extractor */
  Info _info{};

  void VisitStmt_(const AllocateNode* op) override {
    _info.allocates.push_back(GetRef<Allocate>(op));
    VisitStmt(op->body);
  }

  void VisitStmt_(const SeqStmtNode* op) override {
    std::vector<Stmt> seq_stmt = FlattenUnwrap(GetRef<Stmt>(op)).seq;

    if (seq_stmt.size() <= 1) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    for (size_t i = 0; i < seq_stmt.size(); ++i) {
      Stmt stmt{seq_stmt[i]};
      switch (GetStmtType(stmt)) {
        case StmtType::global_copy: {
          Buffer write_buffer{GetCopyWriteBuffer(stmt)};
          _info.copy_write_buffers.push_back(write_buffer);
          _info.old_to_new_write_buffer[write_buffer.as<BufferNode>()] = std::make_pair(-1, -1);
          break;
        }
        case StmtType::local_copy: {
          _info.copy_write_buffers.push_back(Optional<Buffer>{});
          break;
        }
        case StmtType::compute: {
          _info.copy_write_buffers.push_back(Optional<Buffer>{});
          std::vector<Buffer> buffers{GetCopiedBuffersUsedByStmt(stmt)};
          if (buffers.empty()) {
            continue;
          }
          _info.new_buffers_length[i] = 0;
          for (Buffer buffer : buffers) {
            for (size_t j{i - 1}; j >= 0; --j) {
              if (_info.copy_write_buffers[j] == buffer) {
                _info.old_to_new_write_buffer[buffer.as<BufferNode>()] =
                    std::make_pair(i, _info.new_buffers_length[i]);
                _info.new_buffers_length[i] += GetCopyLength(seq_stmt[j]);
                _info.cycless[i] += GetStmtCycles(seq_stmt[j]);
                break;
              }
            }
          }
          break;
        }
      }
    }
  }

  /*! Get all buffers written by copies and used by a given statement */
  std::vector<Buffer> GetCopiedBuffersUsedByStmt(const Stmt& stmt) {
    std::vector<Buffer> buffers{};
    for (PrimExpr arg : GetStmtArgs(stmt)) {
      if (auto buffer_load = arg.as<BufferLoadNode>()) {
        Buffer buffer{buffer_load->buffer};
        // Check if the buffer has already been added
        if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
          // Check if the buffer is copied
          if (_info.old_to_new_write_buffer.count(buffer.as<BufferNode>())) {
            buffers.push_back(buffer);
          }
        }
      }
    }
    return buffers;
  }
};

/*!
 * \brief This mutator looks for the constants used by each compute operator
 * and merges them into a single buffer.
 * Constants written to a buffer with local scope are not merged.
 */
class MergeConstantsMutator : public StmtExprMutator {
 public:
  explicit MergeConstantsMutator(MergeConstantsInfoExtractor::Info info) : _info{std::move(info)} {}

  PrimFunc operator()(PrimFunc main_func, const Map<IntImm, runtime::NDArray>& const_dict) {
    // Rewrite
    Stmt new_body = RewritePrimFuncBody(main_func->body);
    std::unordered_set<const VarNode*> params_to_delete{};
    Map<Var, Buffer> new_buffer_map{MakeNewBufferMap(main_func->buffer_map, &params_to_delete)};
    Array<Var> new_params{MakeNewParams(main_func->params, params_to_delete)};

    // Make the new const dict
    Array<Array<IntImm>> args_to_merge{GetArgsToMerge(main_func->buffer_map, main_func->params)};
    Map<IntImm, Array<IntImm>> buffers_to_merge{
        GetArgsToMergeWithoutArgsNotInConstDict(args_to_merge, const_dict)};
    Map<IntImm, runtime::NDArray> new_const_dict{MakeNewConstDict(buffers_to_merge, const_dict)};

    // Make the new prim func
    auto prim_func_node{main_func.CopyOnWrite()};
    prim_func_node->body = std::move(new_body);
    prim_func_node->buffer_map = std::move(new_buffer_map);
    prim_func_node->params = std::move(new_params);
    PrimFunc f{GetRef<PrimFunc>(prim_func_node)};

    // Add the new const dict as an attribute
    f = WithAttr(std::move(f), "ethos-u.const_dict", new_const_dict);

    return f;
  }

 private:
  /*! The information collected by the MergeConstantsInfoExtractor */
  MergeConstantsInfoExtractor::Info _info;

  /*! Maps an index representing a new buffer to the new buffer */
  std::unordered_map<int /* new buffer index */, Buffer> new_buffers{};

  /*! Maps a copy's read buffer to the new copy's read buffer */
  std::unordered_map<const BufferNode*, Buffer> old_to_new_read_buffers{};

  /*! Maps an index representing a new buffer to the list of buffers to be merged in the new buffer
   */
  std::unordered_map<int /* new buffer index */, std::vector<Buffer>> buffers_to_merge{};

  /*! A set of buffers to delete */
  std::unordered_set<const BufferNode*> buffers_to_delete{};

  Stmt RewritePrimFuncBody(Stmt body) {
    std::unordered_map<const VarNode*, Allocate> var_to_allocate{};

    // Rewrite old allocates
    std::unordered_set<const VarNode*> buffer_vars{GetVarsForWrittenCopyBuffers()};
    for (auto it{_info.allocates.rbegin()}; it != _info.allocates.rend(); ++it) {
      Allocate alloc{*it};
      var_to_allocate[alloc->buffer_var.get()] = alloc;
      if (buffer_vars.count(alloc->buffer_var.as<VarNode>()) == 0) {
        body = Allocate(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->condition, body,
                        alloc->annotations, alloc->span);
      }
    }

    // Rewrite new allocates
    for (auto it{_info.copy_write_buffers.rbegin()}; it != _info.copy_write_buffers.rend(); ++it) {
      if (Optional<Buffer> buffer_opt = *it) {
        Buffer old_write_buffer{buffer_opt.value()};
        int new_buffer_index{
            _info.old_to_new_write_buffer[old_write_buffer.as<BufferNode>()].first};

        // Check if the allocate has already been created
        if (new_buffers.count(new_buffer_index) == 0) {
          BufferNode* new_buffer{old_write_buffer.CopyOnWrite()};
          new_buffer->shape = {_info.new_buffers_length[new_buffer_index]};

          new_buffers[new_buffer_index] = GetRef<Buffer>(new_buffer);

          Allocate old_allocate{var_to_allocate[old_write_buffer->data.get()]};
          body = Allocate(new_buffer->data, new_buffer->dtype, new_buffer->shape, tir::const_true(),
                          body, old_allocate->annotations, old_allocate->span);
        }
      }
    }

    // Rewrite operators
    return this->VisitStmt(body);
  }

  Stmt VisitStmt_(const AllocateNode* op) override {
    auto allocate{CopyOnWrite(op)};
    allocate->body = this->VisitStmt(op->body);
    return Stmt(allocate);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    std::vector<Stmt> seq_stmt = FlattenUnwrap(GetRef<Stmt>(op)).seq;

    if (seq_stmt.size() <= 1) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Array<Stmt> new_seq{};
    for (size_t i{0}; i < seq_stmt.size(); ++i) {
      Stmt stmt{seq_stmt[i]};

      switch (GetStmtType(stmt)) {
        case StmtType::global_copy: {
          Buffer old_write_buffer{_info.copy_write_buffers[i].value()};
          std::pair<int, int> pair{
              _info.old_to_new_write_buffer[old_write_buffer.as<BufferNode>()]};
          int new_buffer_index{pair.first};
          int new_buffer_offset{pair.second};
          UpdateBuffersToMergeAndDelete(stmt, new_buffer_index, new_buffer_offset);

          if (!IsCopyToBeDeleted(new_buffer_offset)) {
            Optional<PrimExpr> cycless{GetMergedCycles(new_buffer_index)};
            new_seq.push_back(MakeNewStmt(
                stmt, MakeNewCopyArgs(stmt, old_write_buffer, new_buffer_index), cycless));
          }
          break;
        }
        case StmtType::local_copy: {
          new_seq.push_back(stmt);
          break;
        }
        case StmtType::compute: {
          new_seq.push_back(MakeNewStmt(stmt, MakeNewComputeArgs(stmt)));
          break;
        }
      }
    }
    return SeqStmt::Flatten(new_seq);
  }

  /*! Returns the variables of the buffers written by copies */
  std::unordered_set<const VarNode*> GetVarsForWrittenCopyBuffers() {
    std::unordered_set<const VarNode*> buffer_vars{};
    std::transform(_info.old_to_new_write_buffer.begin(), _info.old_to_new_write_buffer.end(),
                   std::inserter(buffer_vars, buffer_vars.begin()),
                   [](std::pair<const BufferNode*, std::pair<int, int>> pair) -> const VarNode* {
                     return pair.first->data.as<VarNode>();
                   });
    return buffer_vars;
  }

  /*! Returns the cycles of the new buffer at the given index */
  Optional<PrimExpr> GetMergedCycles(int new_buffer_index) {
    auto it = _info.cycless.find(new_buffer_index);
    if (it != _info.cycless.end()) {
      return Integer(it->second);
    }
    return Optional<PrimExpr>{};
  }

  /*! Returns true if a copy must be deleted, false otherwise */
  bool IsCopyToBeDeleted(int new_buffer_offset) { return new_buffer_offset > 0; }

  Array<PrimExpr> MakeNewCopyArgs(const Stmt& stmt, const Buffer& old_write_buffer,
                                  int new_buffer_index) {
    Array<PrimExpr> args{GetStmtArgs(stmt)};
    int new_length{_info.new_buffers_length[new_buffer_index]};

    Array<PrimExpr> new_args{};
    for (size_t i = 0; i < args.size(); ++i) {
      switch (i) {
        case 1: /* read_address */ {
          auto buffer_load = args[1].as<BufferLoadNode>();
          Buffer buffer{buffer_load->buffer};
          Buffer new_buffer{buffer->data,
                            buffer->dtype,
                            {new_length},
                            buffer->strides,
                            buffer->elem_offset,
                            buffer->name,
                            buffer->data_alignment,
                            buffer->offset_factor,
                            buffer->buffer_type,
                            buffer->axis_separators,
                            buffer->span};
          old_to_new_read_buffers[buffer.as<BufferNode>()] = new_buffer;
          new_args.push_back(BufferLoad(new_buffer, buffer_load->indices, buffer_load->span));
          break;
        }
        case 2: /* length */ {
          new_args.push_back(new_length);
          break;
        }
        case 3: /* write_address */ {
          new_args.push_back(MakeNewBufferLoad(old_write_buffer, 0, true).value());
          break;
        }
        default:
          new_args.push_back(args[i]);
          break;
      }
    }
    return new_args;
  }

  Array<PrimExpr> MakeNewComputeArgs(const Stmt& stmt) {
    Array<PrimExpr> args{GetStmtArgs(stmt)};
    Array<PrimExpr> new_args{};
    for (size_t i = 0; i < args.size(); ++i) {
      if (auto buffer_load = args[i].as<BufferLoadNode>()) {
        BufferLoad new_buffer_load{
            MakeNewBufferLoad(buffer_load->buffer, buffer_load->indices[0], false)
                .value_or(GetRef<BufferLoad>(buffer_load))};
        new_args.push_back(new_buffer_load);
      } else {
        new_args.push_back(args[i]);
      }
    }
    return new_args;
  }

  Stmt MakeNewStmt(const Stmt& stmt, const Array<PrimExpr>& new_args,
                   Optional<PrimExpr> cycless = Optional<PrimExpr>{}) {
    auto attr{stmt.as<AttrStmtNode>()};
    Stmt eval_stmt{attr ? attr->body : stmt};
    auto eval{eval_stmt.as<EvaluateNode>()};
    ICHECK(eval) << "Expected statement to be an evaluate node, but was "
                 << eval_stmt->GetTypeKey();
    auto call{eval->value.as<CallNode>()};
    ICHECK(call) << "Expected expression to be a call node, but was " << eval->value->GetTypeKey();

    Call new_call{call->dtype, call->op, new_args, call->span};
    Evaluate new_eval{new_call, eval->span};

    if (attr) {
      ICHECK(attr->attr_key == "pragma_compute_cycles_hint");
      PrimExpr value = cycless.value_or(attr->value);
      return AttrStmt{attr->node, attr->attr_key, value, new_eval, attr->span};
    } else {
      return std::move(new_eval);
    }
  }

  Optional<BufferLoad> MakeNewBufferLoad(const Buffer& write_buffer, const PrimExpr& old_index,
                                         bool only_old_index) {
    auto it = _info.old_to_new_write_buffer.find(write_buffer.as<BufferNode>());
    if (it != _info.old_to_new_write_buffer.end()) {
      std::pair<int, int> pair{it->second};
      int new_buffer_index{pair.first};
      PrimExpr new_index{only_old_index ? old_index : (pair.second + old_index)};
      return BufferLoad{new_buffers[new_buffer_index], {new_index}};
    }
    return Optional<BufferLoad>{};
  }

  Map<tir::Var, Buffer> MakeNewBufferMap(const Map<tir::Var, Buffer>& buffer_map,
                                         std::unordered_set<const VarNode*>* params_to_delete) {
    Map<tir::Var, Buffer> new_buffer_map{};
    for (std::pair<Var, Buffer> pair : buffer_map) {
      Var var{pair.first};
      Buffer buffer{pair.second};

      if (buffers_to_delete.count(buffer.as<BufferNode>()) == 1) {
        params_to_delete->insert(var.as<VarNode>());
      } else if (old_to_new_read_buffers.count(buffer.as<BufferNode>()) == 1) {
        new_buffer_map.Set(var, old_to_new_read_buffers[buffer.as<BufferNode>()]);
      } else {
        new_buffer_map.Set(var, buffer);
      }
    }
    return new_buffer_map;
  }

  Array<tir::Var> MakeNewParams(const Array<tir::Var>& params,
                                const std::unordered_set<const VarNode*>& params_to_delete) {
    std::vector<Var> new_params{};
    for (Var var : params) {
      if (params_to_delete.count(var.as<VarNode>()) == 0) {
        new_params.push_back(var);
      }
    }
    return new_params;
  }

  void UpdateBuffersToMergeAndDelete(const Stmt& stmt, int new_buffer_index,
                                     int new_buffer_offset) {
    Array<PrimExpr> args{GetStmtArgs(stmt)};
    Buffer read_buffer{GetCopyReadBuffer(stmt)};

    if (buffers_to_merge.count(new_buffer_index) == 0) {
      buffers_to_merge[new_buffer_index] = std::vector<Buffer>{read_buffer};
    } else {
      buffers_to_merge[new_buffer_index].push_back(read_buffer);
    }

    if (new_buffer_offset > 0) {
      buffers_to_delete.insert(read_buffer.as<BufferNode>());
    }
  }

  /*! Returns an array whose elements are the indices of the function arguments to be merged.
   * Example: if a function has three arguments and the second and the third ones must
   * be merged then the array is: [[0], [1, 2], [3]] */
  Array<Array<IntImm>> GetArgsToMerge(const Map<Var, Buffer>& buffer_map,
                                      const Array<Var>& params) {
    std::unordered_map<const BufferNode*, Var> buffer_to_var{};
    for (std::pair<Var, Buffer> var_buffer : buffer_map) {
      buffer_to_var[var_buffer.second.as<BufferNode>()] = var_buffer.first;
    }

    std::unordered_map<const VarNode*, int> var_to_index{};
    for (int i = 0; i < static_cast<int>(params.size()); ++i) {
      var_to_index[params[i].as<VarNode>()] = i;
    }

    std::vector<Array<IntImm>> vector{};
    for (std::pair<int, std::vector<Buffer>> index_vector : buffers_to_merge) {
      std::vector<IntImm> indices{};
      for (Buffer buffer : index_vector.second) {
        const VarNode* var{buffer_to_var[buffer.as<BufferNode>()].as<VarNode>()};
        IntImm index{DataType::Int(64), var_to_index[var]};
        var_to_index.erase(var);
        auto it = std::find_if(indices.begin(), indices.end(),
                               [&](IntImm value) { return value->value == index->value; });
        if (it == indices.end()) {
          indices.push_back(index);
        }
      }
      vector.push_back(Array<IntImm>{indices});
    }

    for (std::pair<const VarNode*, int> var_index : var_to_index) {
      vector.push_back(Array<IntImm>{IntImm(DataType::Int(64), var_index.second)});
    }
    std::sort(vector.begin(), vector.end(),
              [](Array<IntImm> a, Array<IntImm> b) { return a[0]->value < b[0]->value; });
    return vector;
  }

  Map<IntImm, Array<IntImm>> GetArgsToMergeWithoutArgsNotInConstDict(
      const Array<Array<IntImm>>& args_to_merge, const Map<IntImm, runtime::NDArray>& const_dict) {
    Map<IntImm, Array<IntImm>> new_args_to_merge{};
    bool first_arg_found = false;
    int64_t new_arg_key = 0;  // the updated key of the merged const_dict
    for (Array<IntImm> args : args_to_merge) {
      IntImm key{args[0]};
      auto it = std::find_if(const_dict.begin(), const_dict.end(),
                             [&](std::pair<tvm::IntImm, runtime::NDArray> pair) {
                               return pair.first->value == key->value;
                             });
      if (it != const_dict.end()) {
        if (first_arg_found == false) {
          first_arg_found = true;
          new_arg_key = key->value;
        }
        new_args_to_merge.Set(IntImm(DataType::Int(64), new_arg_key), args);
      }
      if (first_arg_found) {
        new_arg_key++;
      }
    }
    return new_args_to_merge;
  }

  Map<IntImm, runtime::NDArray> MakeNewConstDict(const Map<IntImm, Array<IntImm>>& args_to_merge,
                                                 Map<IntImm, runtime::NDArray> const_dict) {
    Map<IntImm, runtime::NDArray> new_const_dict{};
    if (args_to_merge.size() == 0) {
      return new_const_dict;
    }

    for (auto const& elem : args_to_merge) {
      IntImm key = elem.first;
      Array<IntImm> args = elem.second;
      int64_t size = 0;
      for (IntImm arg : args) {
        auto it = std::find_if(const_dict.begin(), const_dict.end(),
                               [&](auto pair) { return pair.first->value == arg->value; });
        runtime::NDArray arg_constant{(*it).second};
        size += runtime::GetDataSize(*arg_constant.operator->());
      }

      runtime::NDArray constant = runtime::NDArray::Empty({size}, DataType::UInt(8), {kDLCPU, 0});

      size_t offset = 0;
      for (IntImm arg : args) {
        auto it = std::find_if(const_dict.begin(), const_dict.end(),
                               [&](auto pair) { return pair.first->value == arg->value; });
        runtime::NDArray arg_constant{(*it).second};
        size_t nbytes = runtime::GetDataSize(*arg_constant.operator->());
        arg_constant.CopyToBytes(static_cast<uint8_t*>(constant->data) + offset, nbytes);
        offset += nbytes;
      }
      new_const_dict.Set(key, constant);
    }
    return new_const_dict;
  }
};

/*!
 * \brief This pass looks for the constants used by each compute operator
 * and merges them into a single buffer.
 * Constants written to a buffer with local scope are not merged.
 * \return tvm::transform::Pass
 */
tvm::transform::Pass MergeConstants() {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    ICHECK(mod->GetGlobalVars().size() == 1 && mod->ContainGlobalVar("main"))
        << "Expected a single primitive function called 'main'. Please run the "
           "MergeConstants pass in conjunction with the LowerToTIR() pass.";
    Optional<Map<IntImm, runtime::NDArray>> const_dict{
        f->attrs.GetAttr("ethos-u.const_dict", Optional<Map<IntImm, runtime::NDArray>>{})};
    ICHECK(const_dict) << "Expected a ethos-u.const_dict attribute";

    MergeConstantsInfoExtractor::Info info{MergeConstantsInfoExtractor()(f)};
    f = RemoveAllocatesMutator()(f);
    return MergeConstantsMutator(info)(f, const_dict.value());
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.contrib.ethos-u.MergeConstants",
                                                 {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.MergeConstants").set_body_typed(MergeConstants);

/*!
 * \brief This pass removes the ethos-u.const_dict attribute
 * \return tvm::transform::Pass
 */
class RemoveConstDictAttributeMutator : public StmtExprMutator {
 public:
  RemoveConstDictAttributeMutator() {}

  PrimFunc operator()(PrimFunc main_func) {
    return WithoutAttr(std::move(main_func), "ethos-u.const_dict");
  }
};

tvm::transform::Pass RemoveConstDictAttribute() {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    return RemoveConstDictAttributeMutator()(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tir.contrib.ethos-u.RemoveConstDictAttribute", {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.RemoveConstDictAttribute")
    .set_body_typed(RemoveConstDictAttribute);

}  // namespace ethosu
}  // namespace contrib
}  // namespace tir
}  // namespace tvm
