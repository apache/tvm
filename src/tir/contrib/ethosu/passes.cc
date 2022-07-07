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
    return CopyComputeReorderingMutator(value)(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tir.contrib.ethos-u.CopyComputeReordering", {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.CopyComputeReordering")
    .set_body_typed(CopyComputeReordering);

/*!
 * \brief This pass looks for the constants used by each compute operator
 * and merges them into a single buffer.
 * Constants written to a buffer with local scope are not merged.
 */
class MergeConstantsMutator : public StmtExprMutator {
 public:
  MergeConstantsMutator() {}

  PrimFunc operator()(PrimFunc main_func, const Map<IntImm, runtime::NDArray>& const_dict) {
    // Analyze
    Stmt new_body{this->VisitStmt(main_func->body)};

    // Rewrite
    analyze = false;
    new_body = rewrite_prim_func_body(new_body);
    std::set<ObjectRef> params_to_delete{};
    auto new_buffer_map{make_new_buffer_map(main_func->buffer_map, &params_to_delete)};
    auto new_params{make_new_params(main_func->params, params_to_delete)};

    // Make the new const dict
    auto args_to_merge{get_args_to_merge(main_func->buffer_map, main_func->params)};
    auto buffers_to_merge{
        get_args_to_merge_without_args_not_in_const_dict(args_to_merge, const_dict)};
    auto new_const_dict{make_new_const_dict(buffers_to_merge, const_dict)};

    // Make the new prim func
    auto prim_func_node{main_func.CopyOnWrite()};
    prim_func_node->body = std::move(new_body);
    prim_func_node->buffer_map = std::move(new_buffer_map);
    prim_func_node->params = std::move(new_params);
    prim_func_node->preflattened_buffer_map = {};
    PrimFunc f{GetRef<PrimFunc>(prim_func_node)};

    // Add the new const dict as an attribute
    f = WithAttr(std::move(f), "ethos-u.const-dict", new_const_dict);

    return f;
  }

 private:
  /*! Indicates whether the pass is analyzing or rewriting */
  bool analyze = true;

  /*! A stack to store allocates as they are visited. */
  std::vector<Allocate> allocates{};

  /*! A list that contains in the i-th position the write buffer of the i-th statement
   * if that statement is a copy to a buffer with global scope  */
  std::vector<Optional<Buffer>> copy_write_buffers{};

  /*! Maps a copy's write buffer to an index representing the
   * new buffer and an offset in that buffer */
  std::map<Buffer, std::pair<int /* new buffer index */, int /* offset */>>
      old_to_new_write_buffer{};

  /*! Maps an index representing a new buffer to the length of that buffer */
  std::map<int /* new buffer index */, int /* length */> new_buffers_length{};

  /*! Maps an index representing a new buffer to the new buffer */
  std::map<int /* new buffer index */, Buffer> new_buffers{};

  /*! Maps an index representing a new buffer to the cycle_counts needed to copy that buffer */
  std::map<int /* new buffer index */, int64_t> cycle_counts{};

  /*! Maps a copy's read buffer to the new copy's read buffer */
  std::map<Buffer, Buffer> old_to_new_read_buffers{};

  /*! Maps an index representing a new buffer to the list of buffers to be merged in the new buffer
   */
  std::map<int /* new buffer index */, std::vector<Buffer>> buffers_to_merge{};

  /*! A set of buffers to delete */
  std::set<Buffer> buffers_to_delete{};

  // Visit

  Stmt VisitStmt_(const AllocateNode* op) override {
    if (analyze) {
      allocates.push_back(GetRef<Allocate>(op));
      return VisitStmt(op->body);
    } else {
      auto allocate{CopyOnWrite(op)};
      allocate->body = this->VisitStmt(op->body);
      return Stmt(allocate);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    if (op->size() <= 1) {
      return StmtExprMutator::VisitStmt_(op);
    }
    return analyze ? analyze_seq_stmt(op) : rewrite_seq_stmt(op);
  }

  Stmt analyze_seq_stmt(const SeqStmtNode* op) {
    auto seq_stmt{GetRef<SeqStmt>(op)};

    for (size_t i = 0; i < seq_stmt.size(); ++i) {
      Stmt stmt{seq_stmt[i]};

      switch (get_stmt_type(stmt)) {
        case StmtType::global_copy: {
          Buffer write_buffer{get_copy_write_buffer(stmt)};
          copy_write_buffers.push_back(write_buffer);
          old_to_new_write_buffer[write_buffer] = std::make_pair(-1, -1);
          break;
        }
        case StmtType::local_copy: {
          copy_write_buffers.push_back(Optional<Buffer>{});
          break;
        }
        case StmtType::compute: {
          copy_write_buffers.push_back(Optional<Buffer>{});
          auto buffers{get_copied_buffers_used_by_stmt(stmt)};
          if (buffers.empty()) {
            continue;
          }
          new_buffers_length[i] = 0;
          for (auto buffer : buffers) {
            for (size_t j{i - 1}; j >= 0; --j) {
              if (copy_write_buffers[j] == buffer) {
                old_to_new_write_buffer[buffer] = std::make_pair(i, new_buffers_length[i]);
                new_buffers_length[i] += get_copy_length(seq_stmt[j]);
                cycle_counts[i] += get_stmt_cycle_counts(seq_stmt[j]);
                break;
              }
            }
          }
          break;
        }
      }
    }
    return std::move(seq_stmt);
  }

  Stmt rewrite_prim_func_body(Stmt body) {
    std::map<const VarNode*, Allocate> var_to_allocate{};

    // Rewrite old allocates
    std::set<ObjectRef> buffer_vars{get_vars_for_written_copy_buffers()};
    for (auto it{allocates.rbegin()}; it != allocates.rend(); ++it) {
      Allocate alloc{*it};
      var_to_allocate[alloc->buffer_var.get()] = alloc;
      if (buffer_vars.count(alloc->buffer_var) == 0) {
        body = Allocate(alloc->buffer_var, alloc->dtype, alloc->extents, alloc->condition, body,
                        alloc->annotations, alloc->span);
      }
    }

    // Rewrite new allocates
    for (auto it{copy_write_buffers.rbegin()}; it != copy_write_buffers.rend(); ++it) {
      if (auto buffer_opt = *it) {
        Buffer old_write_buffer{buffer_opt.value()};
        int new_buffer_index{old_to_new_write_buffer[old_write_buffer].first};

        // Check if the allocate has already been created
        if (new_buffers.count(new_buffer_index) == 0) {
          BufferNode* new_buffer{old_write_buffer.CopyOnWrite()};
          new_buffer->shape = {new_buffers_length[new_buffer_index]};

          new_buffers[new_buffer_index] = GetRef<Buffer>(new_buffer);

          auto old_allocate{var_to_allocate[old_write_buffer->data.get()]};
          body = Allocate(new_buffer->data, new_buffer->dtype, new_buffer->shape, tir::const_true(),
                          body, old_allocate->annotations, old_allocate->span);
        }
      }
    }

    // Rewrite operators
    return this->VisitStmt(body);
  }

  Stmt rewrite_seq_stmt(const SeqStmtNode* op) {
    Array<Stmt> new_seq{};

    auto seq_stmt{GetRef<SeqStmt>(op)};
    for (size_t i{0}; i < seq_stmt.size(); ++i) {
      Stmt stmt{seq_stmt[i]};

      switch (get_stmt_type(stmt)) {
        case StmtType::global_copy: {
          Buffer old_write_buffer{copy_write_buffers[i].value()};
          auto pair{old_to_new_write_buffer[old_write_buffer]};
          auto new_buffer_index{pair.first};
          auto new_buffer_offset{pair.second};
          update_buffers_to_merge_and_delete(stmt, new_buffer_index, new_buffer_offset);

          if (!is_copy_to_be_deleted(new_buffer_offset)) {
            auto cycle_counts{get_merged_cycle_counts(new_buffer_index)};
            new_seq.push_back(make_new_stmt(
                stmt, make_new_copy_args(stmt, old_write_buffer, new_buffer_index), cycle_counts));
          }
          break;
        }
        case StmtType::local_copy: {
          new_seq.push_back(stmt);
          break;
        }
        case StmtType::compute: {
          new_seq.push_back(make_new_stmt(stmt, make_new_compute_args(stmt)));
          break;
        }
      }
    }
    return SeqStmt(new_seq, op->span);
  }

  enum class StmtType { global_copy, local_copy, compute };

  StmtType get_stmt_type(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    if (args[0].as<StringImmNode>()->value == "ethosu_copy") {
      if (args[3].as<BufferLoadNode>()->buffer.scope() == "global") {
        return StmtType::global_copy;
      } else {
        return StmtType::local_copy;
      }
    }
    return StmtType::compute;
  }

  Buffer get_copy_read_buffer(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    return args[1].as<BufferLoadNode>()->buffer;
  }

  Buffer get_copy_write_buffer(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    return args[3].as<BufferLoadNode>()->buffer;
  }

  int64_t get_copy_length(const Stmt& stmt) {
    auto args{get_stmt_args(stmt)};
    return args[2].as<IntImmNode>()->value;
  }

  int64_t get_stmt_cycle_counts(const Stmt& stmt) {
    auto attr{stmt.as<AttrStmtNode>()};
    if (attr && attr->attr_key == "pragma_compute_cycles_hint") {
      int64_t cycle_count{Downcast<Integer>(attr->value)->value};
      return cycle_count;
    }
    return 0;
  }

  std::vector<Buffer> get_copied_buffers_used_by_stmt(const Stmt& stmt) {
    std::vector<Buffer> buffers{};
    for (auto arg : get_stmt_args(stmt)) {
      if (auto buffer_load = arg.as<BufferLoadNode>()) {
        auto buffer{buffer_load->buffer};
        // Check if the buffer has already been added
        if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
          // Check if the buffer is copied
          if (old_to_new_write_buffer.count(buffer)) {
            buffers.push_back(buffer);
          }
        }
      }
    }
    return buffers;
  }

  std::set<ObjectRef> get_vars_for_written_copy_buffers() {
    std::set<ObjectRef> buffer_vars{};
    std::transform(old_to_new_write_buffer.begin(), old_to_new_write_buffer.end(),
                   std::inserter(buffer_vars, buffer_vars.begin()),
                   [](auto pair) -> Var { return pair.first->data; });
    return buffer_vars;
  }

  tvm::runtime::Array<tvm::PrimExpr> get_stmt_args(const Stmt& stmt) {
    auto attr{stmt.as<AttrStmtNode>()};
    Stmt eval_stmt{attr ? attr->body : stmt};
    auto eval{eval_stmt.as<EvaluateNode>()};
    ICHECK(eval) << "Expected statement to be an evaluate node, but was "
                 << eval_stmt->GetTypeKey();
    auto call{eval->value.as<CallNode>()};
    ICHECK(call) << "Expected expression to be a call node, but was " << eval->value->GetTypeKey();
    return call->args;
  }

  Optional<PrimExpr> get_merged_cycle_counts(int new_buffer_index) {
    auto it = cycle_counts.find(new_buffer_index);
    if (it != cycle_counts.end()) {
      return Integer(it->second);
    }
    return Optional<PrimExpr>{};
  }

  bool is_copy_to_be_deleted(int new_buffer_offset) { return new_buffer_offset > 0; }

  Array<PrimExpr> make_new_copy_args(const Stmt& stmt, const Buffer& old_write_buffer,
                                     int new_buffer_index) {
    Array<PrimExpr> args{get_stmt_args(stmt)};
    auto new_length{new_buffers_length[new_buffer_index]};

    Array<PrimExpr> new_args{};
    for (size_t i = 0; i < args.size(); ++i) {
      switch (i) {
        case 1: /* read_address */ {
          auto buffer_load = args[1].as<BufferLoadNode>();
          auto buffer{buffer_load->buffer};
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
          old_to_new_read_buffers[buffer] = new_buffer;
          new_args.push_back(BufferLoad(new_buffer, buffer_load->indices, buffer_load->span));
          break;
        }
        case 2: /* length */ {
          new_args.push_back(new_length);
          break;
        }
        case 3: /* write_address */ {
          new_args.push_back(make_new_buffer_load(old_write_buffer, 0, true).value());
          break;
        }
        default:
          new_args.push_back(args[i]);
          break;
      }
    }
    return new_args;
  }

  Array<PrimExpr> make_new_compute_args(const Stmt& stmt) {
    Array<PrimExpr> args{get_stmt_args(stmt)};
    Array<PrimExpr> new_args{};
    for (size_t i = 0; i < args.size(); ++i) {
      if (auto buffer_load = args[i].as<BufferLoadNode>()) {
        auto new_buffer_load{
            make_new_buffer_load(buffer_load->buffer, buffer_load->indices[0], false)
                .value_or(GetRef<BufferLoad>(buffer_load))};
        new_args.push_back(new_buffer_load);
      } else {
        new_args.push_back(args[i]);
      }
    }
    return new_args;
  }

  Stmt make_new_stmt(const Stmt& stmt, const Array<PrimExpr>& new_args,
                     Optional<PrimExpr> cycle_counts = Optional<PrimExpr>{}) {
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
      PrimExpr value = cycle_counts.value_or(attr->value);
      return AttrStmt{attr->node, attr->attr_key, value, new_eval, attr->span};
    } else {
      return std::move(new_eval);
    }
  }

  Optional<BufferLoad> make_new_buffer_load(const Buffer& write_buffer, const PrimExpr& old_index,
                                            bool only_old_index) {
    auto it = old_to_new_write_buffer.find(write_buffer);
    if (it != old_to_new_write_buffer.end()) {
      auto pair{it->second};
      auto new_buffer_index{pair.first};
      auto new_index{only_old_index ? old_index : (pair.second + old_index)};
      return BufferLoad{new_buffers[new_buffer_index], {new_index}};
    }
    return Optional<BufferLoad>{};
  }

  Map<tir::Var, Buffer> make_new_buffer_map(const Map<tir::Var, Buffer>& buffer_map,
                                            std::set<ObjectRef>* params_to_delete) {
    Map<tir::Var, Buffer> new_buffer_map{};
    for (auto pair : buffer_map) {
      Var var{pair.first};
      Buffer buffer{pair.second};

      if (buffers_to_delete.count(buffer) == 1) {
        params_to_delete->insert(var);
      } else if (old_to_new_read_buffers.count(buffer) == 1) {
        new_buffer_map.Set(var, old_to_new_read_buffers[buffer]);
      } else {
        new_buffer_map.Set(var, buffer);
      }
    }
    return new_buffer_map;
  }

  Array<tir::Var> make_new_params(const Array<tir::Var>& params,
                                  const std::set<ObjectRef>& params_to_delete) {
    std::vector<Var> new_params{};
    for (auto var : params) {
      if (params_to_delete.count(var) == 0) {
        new_params.push_back(var);
      }
    }
    return new_params;
  }

  void update_buffers_to_merge_and_delete(const Stmt& stmt, int new_buffer_index,
                                          int new_buffer_offset) {
    Array<PrimExpr> args{get_stmt_args(stmt)};
    Buffer read_buffer{get_copy_read_buffer(stmt)};

    if (buffers_to_merge.count(new_buffer_index) == 0) {
      buffers_to_merge[new_buffer_index] = std::vector<Buffer>{read_buffer};
    } else {
      buffers_to_merge[new_buffer_index].push_back(read_buffer);
    }

    if (new_buffer_offset > 0) {
      buffers_to_delete.insert(read_buffer);
    }
  }

  /*! Returns an array whose elements are the indices of the function arguments to be merged.
   * Example: if a function has three arguments and the second and the third ones must
   * be merged then the array is: [[0], [1, 2], [3]] */
  Array<Array<IntImm>> get_args_to_merge(const Map<Var, Buffer>& buffer_map,
                                         const Array<Var>& params) {
    std::map<Buffer, Var> buffer_to_var{};
    for (auto var_buffer : buffer_map) {
      buffer_to_var[var_buffer.second] = var_buffer.first;
    }

    std::map<ObjectRef, int> var_to_index{};
    for (int i = 0; i < static_cast<int>(params.size()); ++i) {
      var_to_index[params[i]] = i;
    }

    std::vector<Array<IntImm>> vector{};
    for (auto index_vector : buffers_to_merge) {
      std::vector<IntImm> indices{};
      for (auto buffer : index_vector.second) {
        auto var{buffer_to_var[buffer]};
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

    for (auto var_index : var_to_index) {
      vector.push_back(Array<IntImm>{IntImm(DataType::Int(64), var_index.second)});
    }
    std::sort(vector.begin(), vector.end(),
              [](Array<IntImm> a, Array<IntImm> b) { return a[0]->value < b[0]->value; });
    return vector;
  }

  Array<Array<IntImm>> get_args_to_merge_without_args_not_in_const_dict(
      const Array<Array<IntImm>>& args_to_merge, const Map<IntImm, runtime::NDArray>& const_dict) {
    Array<Array<IntImm>> new_args_to_merge{};
    for (auto args : args_to_merge) {
      IntImm key{args[0]};
      auto it = std::find_if(const_dict.begin(), const_dict.end(),
                             [&](std::pair<tvm::IntImm, runtime::NDArray> pair) {
                               return pair.first->value == key->value;
                             });
      if (it != const_dict.end()) {
        new_args_to_merge.push_back(args);
      }
    }
    return new_args_to_merge;
  }

  Map<IntImm, runtime::NDArray> make_new_const_dict(const Array<Array<IntImm>>& args_to_merge,
                                                    Map<IntImm, runtime::NDArray> const_dict) {
    Map<IntImm, runtime::NDArray> new_const_dict{};
    if (args_to_merge.size() == 0) {
      return new_const_dict;
    }

    int64_t key = args_to_merge[0][0]->value;
    for (auto args : args_to_merge) {
      int64_t size = 0;
      for (auto arg : args) {
        auto it = std::find_if(const_dict.begin(), const_dict.end(),
                               [&](auto pair) { return pair.first->value == arg->value; });
        auto arg_constant{(*it).second};
        size += runtime::GetDataSize(*arg_constant.operator->());
      }

      runtime::NDArray constant = runtime::NDArray::Empty({size}, DataType::UInt(8), {kDLCPU, 0});

      size_t offset = 0;
      for (auto arg : args) {
        auto it = std::find_if(const_dict.begin(), const_dict.end(),
                               [&](auto pair) { return pair.first->value == arg->value; });
        auto arg_constant{(*it).second};
        size_t nbytes = runtime::GetDataSize(*arg_constant.operator->());
        arg_constant.CopyToBytes(static_cast<uint8_t*>(constant->data) + offset, nbytes);
        offset += nbytes;
      }
      new_const_dict.Set(IntImm(DataType::Int(64), key), constant);
      key += 1;
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
    auto const_dict{
        f->attrs.GetAttr("ethos-u.const-dict", Optional<Map<IntImm, runtime::NDArray>>{})};
    ICHECK(const_dict) << "Expected a ethos-u.const-dict attribute";
    return MergeConstantsMutator()(f, const_dict.value());
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.contrib.ethos-u.MergeConstants",
                                                 {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.MergeConstants").set_body_typed(MergeConstants);

/*!
 * \brief This pass removes the ethos-u.const-dict attribute
 * \return tvm::transform::Pass
 */
class RemoveConstDictAttributeMutator : public StmtExprMutator {
 public:
  RemoveConstDictAttributeMutator() {}

  PrimFunc operator()(PrimFunc main_func) {
    return WithoutAttr(std::move(main_func), "ethos-u.const-dict");
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
