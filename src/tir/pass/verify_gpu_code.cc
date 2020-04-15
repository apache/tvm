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
 * \file verify_gpu_code.cc
 * \brief Verify the correctness of a GPU IR.
 *        It will check the whether the amount of memory usage or the number of threads
 *        in a block exceeds the limit
 */

#include <tvm/runtime/registry.h>

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class GPUCodeVerifier : public StmtVisitor {
 public:
  bool Verify(Stmt stmt,
              int64_t max_local_memory_per_block,
              int64_t max_shared_memory_per_block,
              int64_t max_threads_per_block,
              int64_t max_thread_x,
              int64_t max_thread_y,
              int64_t max_thread_z) {
    max_local_memory_per_block_ = static_cast<size_t>(max_local_memory_per_block);
    max_shared_memory_per_block_ = static_cast<size_t>(max_shared_memory_per_block);
    max_threads_per_block_ = static_cast<size_t>(max_threads_per_block);
    max_thread_x_ = static_cast<size_t>(max_thread_x);
    max_thread_y_ = static_cast<size_t>(max_thread_y);
    max_thread_z_ = static_cast<size_t>(max_thread_z);

    Reset_();

    this->VisitStmt(stmt);

    return valid_;
  }

  void VisitStmt_(const AllocateNode* op) final {
    StmtVisitor::VisitStmt_(op);
    // visit an allocation of a buffer in shared memory, record its size
    if (visited_local_buffers_.count(op->buffer_var.get()) != 0) {
      size_t size = static_cast<size_t>(op->constant_allocation_size());
      local_memory_per_block_ += size * op->dtype.bytes() * op->dtype.lanes();
    } else if (visited_shared_buffers_.count(op->buffer_var.get()) != 0) {
      size_t size = static_cast<size_t>(op->constant_allocation_size());
      shared_memory_per_block_ += size * op->dtype.bytes() * op->dtype.lanes();
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::storage_scope) {
      std::string op_value = op->value.as<StringImmNode>()->value;
      if (op_value == "local") {
        visited_local_buffers_.insert(op->node.as<VarNode>());
      } else if (op_value == "shared") {
        visited_shared_buffers_.insert(op->node.as<VarNode>());
      }
      StmtVisitor::VisitStmt_(op);
    } else if (op->attr_key == attr::thread_extent) {
      if (nest_level_ == 0) {
        // enter a new kernel, reset statistics
        Reset_();
      }

      Var var = op->node.as<IterVarNode>()->var;
      const auto *extent = op->value.as<IntImmNode>();
      CHECK(extent);

      // record the number of threads in a block
      std::string name = var.get()->name_hint;
      if (name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z") {
        size_t length = static_cast<size_t>(extent->value);
        if (!visited_threads_.count(name)) {
          visited_threads_.insert(name);
          thread_per_block_ *= length;

          if (name == "threadIdx.x") {
            valid_ &= length <= max_thread_x_;
            thread_x_extent_ = length;
          } else if (name == "threadIdx.y") {
            valid_ &= length <= max_thread_y_;
            thread_y_extent_ = length;
          } else if (name == "threadIdx.z") {
            valid_ &= length <= max_thread_z_;
            thread_z_extent_ = length;
          }
        } else {
          // the thread should be bound to axes with the same length
          if (name == "threadIdx.x") {
            valid_ &= length == thread_x_extent_;
          } else if (name == "threadIdx.y") {
            valid_ &= length == thread_y_extent_;
          } else if (name == "threadIdx.z") {
            valid_ &= length == thread_z_extent_;
          }
        }
      }

      nest_level_++;
      StmtVisitor::VisitStmt_(op);
      nest_level_--;

      if (nest_level_ == 0) {
        // exit a kernel, check the validity
        valid_ &= thread_per_block_ <= max_threads_per_block_;

        valid_ &= local_memory_per_block_ <= max_local_memory_per_block_;
        valid_ &= shared_memory_per_block_ <= max_shared_memory_per_block_;
      }
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

 private:
  int nest_level_{0};

  std::unordered_set<const VarNode *> visited_local_buffers_;
  std::unordered_set<const VarNode *> visited_shared_buffers_;
  std::unordered_set<std::string> visited_threads_;

  size_t thread_x_extent_, thread_y_extent_, thread_z_extent_;

  size_t local_memory_per_block_;
  size_t shared_memory_per_block_;
  size_t thread_per_block_;

  size_t max_local_memory_per_block_;
  size_t max_shared_memory_per_block_;
  size_t max_threads_per_block_;
  size_t max_thread_x_, max_thread_y_, max_thread_z_;

  bool valid_{true};

  void Reset_() {
    visited_local_buffers_.clear();
    visited_shared_buffers_.clear();
    local_memory_per_block_ = 0;
    shared_memory_per_block_ = 0;

    visited_threads_.clear();
    thread_per_block_ = 1;
  }
};

bool VerifyGPUCode(Stmt stmt,
                   Map<std::string, PrimExpr> constraints) {
  GPUCodeVerifier verifier;

  int64_t max_local_memory_per_block = INT64_MAX;
  int64_t max_shared_memory_per_block = INT64_MAX;
  int64_t max_threads_per_block = INT64_MAX;
  int64_t max_thread_x = INT64_MAX;
  int64_t max_thread_y = INT64_MAX;
  int64_t max_thread_z = INT64_MAX;

  for (auto iter : constraints) {
    const IntImmNode* val = iter.second.as<IntImmNode>();
    if (iter.first == "max_local_memory_per_block")
      max_local_memory_per_block = val->value;
    else if (iter.first == "max_shared_memory_per_block")
      max_shared_memory_per_block = val->value;
    else if (iter.first == "max_threads_per_block")
      max_threads_per_block = val->value;
    else if (iter.first == "max_thread_x")
      max_thread_x = val->value;
    else if (iter.first == "max_thread_y")
      max_thread_y = val->value;
    else if (iter.first == "max_thread_z")
      max_thread_z = val->value;
    else
      LOG(FATAL) << "Invalid check item: " << iter.first;
  }

  return verifier.Verify(stmt,
                         max_local_memory_per_block,
                         max_shared_memory_per_block,
                         max_threads_per_block,
                         max_thread_x,
                         max_thread_y,
                         max_thread_z);
}

}  // namespace tir
}  // namespace tvm
