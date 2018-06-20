/*!
 *  Copyright (c) 2018 by Contributors
 * \file verify_gpu_code.cc
 * \brief Verify the correctness of a GPU IR.
 *        It will check the whether the amount of shared memory or
 *        the number of threads in a block exceeds the limit
 */

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

class GPUCodeVerifier : public IRVisitor {
 public:
  bool verify(tvm::Stmt stmt, int max_shared_memory_per_block, int max_thread_per_block) {
    max_shared_memory_per_block_ = static_cast<size_t>(max_shared_memory_per_block);
    max_thread_per_block_ = static_cast<size_t>(max_thread_per_block);

    this->Visit(stmt);

    return valid;
  }

  void Visit_(const ProducerConsumer *op) {
    if (nest_level_ == 0) {
      // enter a new kernel, reset statistics
      reset_();
    }

    if (op->is_producer) {
      nest_level_++;
      IRVisitor::Visit_(op);
      nest_level_--;
    } else {
      IRVisitor::Visit_(op);
    }

    if (nest_level_ == 0) {
      // exit a kernel, check the validity
      if (thread_per_block_ > max_thread_per_block_) {
        valid = false;
      }
      if (shared_memory_per_block_ > max_shared_memory_per_block_) {
        valid = false;
      }
    }
  }

  void Visit_(const Allocate *op) {
    IRVisitor::Visit_(op);
    // visit an allocation of a buffer in shared memory, record its size
    if (shared_buffers_.count(op->buffer_var.get()) != 0) {
      int64_t size = op->type.bytes();
      for (auto dim : op->extents) {
        size *= dim.as<IntImm>()->value;
      }
      shared_memory_per_block_ += size;
    }
  }

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == attr::storage_scope) {
      if (op->value.as<StringImm>()->value == "shared") {
        shared_buffers_.insert(op->node.as<tvm::Variable>());
      }
    } else if (op->attr_key == attr::thread_extent) {
      VarExpr var = op->node.as<tvm::IterVarNode>()->var;
      const auto *extent = op->value.as<IntImm>();
      CHECK(extent);

      // record the number of threads in a block
      std::string name = var.get()->name_hint;
      if (name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z") {
        if (visited_threads_.find(name) == visited_threads_.end()) {
          visited_threads_.insert(name);
          thread_per_block_ *= extent->value;
        }
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  int nest_level_{0};

  std::unordered_set<const tvm::Variable *> shared_buffers_;
  std::unordered_set<std::string> visited_threads_;
  size_t shared_memory_per_block_;
  size_t thread_per_block_;

  size_t max_shared_memory_per_block_;
  size_t max_thread_per_block_;

  bool valid{true};

  void reset_() {
    shared_buffers_.clear();
    shared_memory_per_block_ = 0;
    thread_per_block_ = 1;
    visited_threads_.clear();
  }
};

bool VerifyGPUCode(Stmt stmt,
                   int max_shared_memory_per_block,
                   int max_thread_per_block) {
  GPUCodeVerifier verifier;
  return verifier.verify(stmt, max_shared_memory_per_block, max_thread_per_block);
}

}  // namespace ir
}  // namespace tvm
