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
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class GPUCodeVerifier : public StmtExprVisitor {
 public:
  std::vector<String> Verify(Stmt stmt, int64_t max_local_memory_per_block,
                             int64_t max_shared_memory_per_block, int64_t max_threads_per_block,
                             int64_t max_thread_x, int64_t max_thread_y, int64_t max_thread_z,
                             int64_t max_vthread, int64_t max_vector_bytes) {
    max_local_memory_per_block_ = static_cast<size_t>(max_local_memory_per_block);
    max_shared_memory_per_block_ = static_cast<size_t>(max_shared_memory_per_block);
    max_threads_per_block_ = static_cast<size_t>(max_threads_per_block);
    max_thread_x_ = static_cast<size_t>(max_thread_x);
    max_thread_y_ = static_cast<size_t>(max_thread_y);
    max_thread_z_ = static_cast<size_t>(max_thread_z);
    max_vthread_ = static_cast<size_t>(max_vthread);
    max_vector_bytes_ = static_cast<size_t>(max_vector_bytes);

    Reset_();

    // TODO(jcf94): Add support of detecting CUDA Misaligned Address error
    this->VisitStmt(stmt);

    return errors_;
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
    if (op->dtype.lanes() > 1) {
      if (static_cast<size_t>(op->dtype.lanes() * op->dtype.bytes()) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << op->dtype.lanes() << ") times number of bytes ("
          << op->dtype.bytes() << ") for dtype " << op->dtype
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
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
    } else if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      if (nest_level_ == 0) {
        // enter a new kernel, reset statistics
        Reset_();
      }

      Var var = op->node.as<IterVarNode>()->var;
      const auto* extent = op->value.as<IntImmNode>();
      CHECK(extent);

      std::string name = var.get()->name_hint;
      // record the number of threads in a block
      if (name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z" ||
          name == "vthread") {
        size_t length = static_cast<size_t>(extent->value);
        if (!visited_threads_.count(name)) {
          visited_threads_.insert(name);
          thread_per_block_ *= length;

          auto err = [this](std::string id, size_t ext, size_t m) {
            if (ext > m) {
              std::stringstream s;
              s << "Extent of " << id << " (" << ext << ") is greater than maximum allowed (" << m
                << ");";
              errors_.push_back(s.str());
            }
          };

          if (name == "threadIdx.x") {
            err("threadIdx.x", length, max_thread_x_);
            thread_x_extent_ = length;
          } else if (name == "threadIdx.y") {
            err("threadIdx.y", length, max_thread_y_);
            thread_y_extent_ = length;
          } else if (name == "threadIdx.z") {
            err("threadIdx.z", length, max_thread_z_);
            thread_z_extent_ = length;
          } else if (name == "vthread") {
            err("vthread", length, max_vthread_);
          }
        } else {
          // the thread should be bound to axes with the same length
          auto err = [this, name](std::string id, size_t ext, size_t m) {
            if (name == id && ext != m) {
              std::stringstream s;
              s << "Extent of " << id << " (" << ext << ") does not match the bound " << m;
              errors_.push_back(s.str());
            }
          };
          err("threadIdx.x", length, thread_x_extent_);
          err("threadIdx.y", length, thread_y_extent_);
          err("threadIdx.z", length, thread_z_extent_);
        }
      }

      nest_level_++;
      StmtVisitor::VisitStmt_(op);
      nest_level_--;

      if (nest_level_ == 0) {
        // exit a kernel, check the validity
        auto err = [this](std::string id, size_t num, size_t m) {
          if (num > m) {
            std::stringstream s;
            s << "Used " << id << " (" << num << ") is greater than the allowed maximum (" << m
              << ")";
            errors_.push_back(s.str());
          }
        };
        err("threads per block", thread_per_block_, max_threads_per_block_);
        err("local memory per block", local_memory_per_block_, max_local_memory_per_block_);
        err("shared memory per block", shared_memory_per_block_, max_shared_memory_per_block_);
      }
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const ForNode* op) {
    if (op->loop_var->name_hint == "vthread.s") {
      const auto* extent = op->extent.as<IntImmNode>();
      CHECK(extent);

      size_t num_vthread = static_cast<size_t>(extent->value);
      if (num_vthread > max_vthread_) {
        std::stringstream s;
        s << "Number of vthreads (" << num_vthread << ") is greater than the allowed maximum ("
          << max_vthread_ << ")";
        errors_.push_back(s.str());
      }
    }

    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) {
    if (op->dtype.lanes() > 1) {
      if (static_cast<size_t>(op->dtype.lanes() * op->dtype.bytes()) > max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << op->dtype.lanes() << ") times number of bytes ("
          << op->dtype.bytes() << ") for dtype " << op->dtype
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const StoreNode* op) {
    if (op->index->dtype.lanes() > 1) {
      if (static_cast<size_t>(op->index->dtype.lanes() * op->index->dtype.bytes()) >
          max_vector_bytes_) {
        std::stringstream s;
        s << "Number of lanes (" << op->index->dtype.lanes() << ") times number of bytes ("
          << op->index->dtype.bytes() << ") for dtype " << op->index->dtype
          << " is greater than the maximum number of vector bytes (" << max_vector_bytes_ << ")";
        errors_.push_back(s.str());
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 private:
  int nest_level_{0};

  std::unordered_set<const VarNode*> visited_local_buffers_;
  std::unordered_set<const VarNode*> visited_shared_buffers_;
  std::unordered_set<std::string> visited_threads_;

  size_t thread_x_extent_, thread_y_extent_, thread_z_extent_;

  size_t local_memory_per_block_;
  size_t shared_memory_per_block_;
  size_t thread_per_block_;

  size_t max_local_memory_per_block_;
  size_t max_shared_memory_per_block_;
  size_t max_threads_per_block_;
  size_t max_thread_x_, max_thread_y_, max_thread_z_, max_vthread_;
  size_t max_vector_bytes_;

  std::vector<String> errors_;

  void Reset_() {
    visited_local_buffers_.clear();
    visited_shared_buffers_.clear();
    local_memory_per_block_ = 0;
    shared_memory_per_block_ = 0;

    visited_threads_.clear();
    thread_per_block_ = 1;
  }
};

std::vector<String> VerifyGPUCode_(const PrimFunc& func, Map<String, PrimExpr> constraints) {
  GPUCodeVerifier verifier;

  int64_t max_local_memory_per_block = INT64_MAX;
  int64_t max_shared_memory_per_block = INT64_MAX;
  int64_t max_threads_per_block = INT64_MAX;
  int64_t max_thread_x = INT64_MAX;
  int64_t max_thread_y = INT64_MAX;
  int64_t max_thread_z = INT64_MAX;
  int64_t max_vthread = INT64_MAX;
  int64_t max_vector_bytes = INT64_MAX;

  for (auto iter : constraints) {
    const IntImmNode* val = iter.second.as<IntImmNode>();
    if (iter.first == "max_local_memory_per_block") {
      max_local_memory_per_block = val->value;
    } else if (iter.first == "max_shared_memory_per_block") {
      max_shared_memory_per_block = val->value;
    } else if (iter.first == "max_threads_per_block") {
      max_threads_per_block = val->value;
    } else if (iter.first == "max_thread_x") {
      max_thread_x = val->value;
    } else if (iter.first == "max_thread_y") {
      max_thread_y = val->value;
    } else if (iter.first == "max_thread_z") {
      max_thread_z = val->value;
    } else if (iter.first == "max_vthread") {
      max_vthread = val->value;
    } else if (iter.first == "max_vector_bytes") {
      max_vector_bytes = val->value;
    } else {
      LOG(FATAL) << "Invalid check item: " << iter.first;
    }
  }

  return verifier.Verify(func->body, max_local_memory_per_block, max_shared_memory_per_block,
                         max_threads_per_block, max_thread_x, max_thread_y, max_thread_z,
                         max_vthread, max_vector_bytes);
}

bool VerifyGPUCode(const PrimFunc& func, Map<String, PrimExpr> constraints) {
  auto errs = VerifyGPUCode_(func, constraints);
  return errs.size() == 0;
}

TVM_REGISTER_GLOBAL("tir.analysis.verify_gpu_code").set_body_typed(VerifyGPUCode);

namespace transform {

Pass VerifyGPUCode(Map<String, PrimExpr> constraints) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        auto errs = VerifyGPUCode_(func, constraints);
        if (errs.size() != 0) {
          std::stringstream s;
          for (auto& err : errs) {
            s << "    " << err << std::endl;
          }
          LOG(FATAL) << "RuntimeError: GPU constraint(s) violated:\n"
                     << s.str() << "  In function\n"
                     << func;
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.VerifyGPUCode", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifyGPUCode").set_body_typed(VerifyGPUCode);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
