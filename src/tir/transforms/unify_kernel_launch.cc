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
 * \file unify_kernel_launch.cc
 # \note This pass should be executed after the `UnifyThreadBinding`.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageScope;
using support::StartsWith;

static const std::vector<std::string> reverse_thread_emit_order = {
    "vthread.z",   "vthread.y",   "vthread.x",   "vthread",  //
    "threadIdx.z", "threadIdx.y", "threadIdx.x",             //
    "blockIdx.z",  "blockIdx.y",  "blockIdx.x",
};

/*! \brief The helper structure for the kernel infomation. */
struct KernelInfo {
  /*! \brief The map from thread name to its thread extents. */
  Map<String, Range> thread_extents;
  /*! \brief The map from thread name to its thread vars. */
  Map<String, Var> thread_vars;
  /*! \brief The kernel body without thread binding loops or attrs. */
  Stmt body;
  /*! \brief If the kernel use shared memory. */
  bool use_shared_memory = false;

  bool same_thread_config_as(const KernelInfo& other, arith::Analyzer* analyzer) const {
    // Check if the threads number are the same.
    if (thread_extents.size() != other.thread_extents.size()) {
      return false;
    }
    // Check if the threads extents are the same.
    for (const auto& kv : thread_extents) {
      const String& thread_tag = kv.first;
      const Range& dom = kv.second;
      auto it = other.thread_extents.find(thread_tag);
      if (it == other.thread_extents.end()) {
        return false;
      }
      if (!analyzer->CanProveEqual(dom->min, (*it).second->min) ||
          !analyzer->CanProveEqual(dom->extent, (*it).second->extent)) {
        return false;
      }
    }
    return true;
  }
};

/*! \brief The stmt visitor who collect the kernel launch infomation. */
class KernelInfoCollector : public StmtMutator {
 public:
  static std::vector<KernelInfo> Collect(const Stmt& stmt) {
    KernelInfoCollector collector;
    collector(stmt);
    return collector.kernel_info_;
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // If this AttrStmt is not thread binding attribute, return as usual.
    if (op->attr_key != attr::thread_extent && op->attr_key != attr::virtual_thread) {
      return StmtMutator::VisitStmt_(op);
    } else {
      const IterVar& iter_var = Downcast<IterVar>(op->node);
      return CollectKernelInfoImpl(op->body, iter_var->thread_tag, iter_var->var, iter_var->dom);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // If this For is not thread binding attribute, return as usual.
    if (op->kind != ForKind::kThreadBinding) {
      return StmtMutator::VisitStmt_(op);
    } else {
      return CollectKernelInfoImpl(op->body, op->thread_binding.value()->thread_tag, op->loop_var,
                                   Range::FromMinExtent(op->min, op->extent));
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const Buffer& buf : op->alloc_buffers) {
      if (IsSharedMemory(buf->data)) {
        kernel_info_.back().use_shared_memory = true;
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (IsSharedMemory(op->buffer_var)) {
      kernel_info_.back().use_shared_memory = true;
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt CollectKernelInfoImpl(const Stmt& stmt, const String& thread_tag, const Var& var,
                             const Range& dom) {
    // Step 1: Enter a kernel launch scope if the thread tag starts with "blockIdx".
    bool in_kernel_launch_scope = in_kernel_launch_scope_;
    if (!in_kernel_launch_scope_) {
      // Create a new Kernel info
      kernel_info_.push_back(KernelInfo());
      // It must be in a kernel launch scope if we call `CollectKernelInfoImpl`.
      in_kernel_launch_scope_ = true;
    }

    // Step 2. Get the current kernel info.
    ICHECK(!kernel_info_.empty());
    KernelInfo& info = kernel_info_.back();

    // Step 3. Update the thread extents.
    CHECK(info.thread_extents.count(thread_tag) == 0)
        << "Duplicate thread tag " << thread_tag << ". Please run pass `UnifyThreadBinding` first.";
    info.thread_extents.Set(thread_tag, dom);
    info.thread_vars.Set(thread_tag, var);

    // Step 4. Recursive visit
    Stmt new_stmt = StmtMutator::VisitStmt(stmt);

    // Step 5. Exit the kernel launch scope.
    if (!in_kernel_launch_scope) {
      in_kernel_launch_scope_ = false;
      info.body = new_stmt;
    }

    return new_stmt;
  }

  inline bool IsSharedMemory(const Var& var) {
    StorageScope scope = StorageScope::Create(GetPtrStorageScope(var));
    return scope.rank == runtime::StorageRank::kShared;
  }

 private:
  /*! \brief The collected kernel info. */
  std::vector<KernelInfo> kernel_info_;
  /*! \brief A flag indicates if current is inside a kernel launch scope. */
  bool in_kernel_launch_scope_ = false;
};

/*!
 * \brief Fuse the given kernels to the first one, and set the rest kernel to nop.
 * \param kernel_info The kernel info.
 * \param start_pos The start position of the kernels to be fused.
 * \param end_pos The end position of the kernels to be fused. (the `end_pos` is exclusive)
 */
void FuseKernel(std::vector<KernelInfo>* kernel_info, size_t start_pos, size_t end_pos) {
  // Use the first kernels thread vars as the fused kernel's thread vars.
  std::vector<Stmt> bodies;
  const KernelInfo& major_kernel = (*kernel_info)[start_pos];
  bodies.push_back(major_kernel.body);

  // Step 2. Replace the rest kernels iter_vars
  for (size_t i = start_pos + 1; i < end_pos; ++i) {
    const KernelInfo& current_kernel = (*kernel_info)[i];
    // Step 2.1. Create var maps
    Map<Var, PrimExpr> var_map;
    for (const auto& kv : major_kernel.thread_vars) {
      const String& thread_tag = kv.first;
      const Var& thread_var = kv.second;
      auto it = current_kernel.thread_vars.find(thread_tag);
      ICHECK(it != current_kernel.thread_vars.end());
      var_map.Set((*it).second, cast((*it).second.dtype(), thread_var));
    }

    // Step 2.2. Create the var map for the thread vars
    bodies.push_back(Substitute((*kernel_info)[i].body, var_map));
  }

  // Step 3. Replace the major kernel body.
  (*kernel_info)[start_pos].body = SeqStmt::Flatten(bodies);

  // Step 4. Set all other kernels' body to nop
  for (size_t i = start_pos + 1; i < end_pos; ++i) {
    (*kernel_info)[i].body = Evaluate(0);
  }
}

/*!
 * \brief Fuse the given kernels to the first one, and set the rest kernel to nop.
 * \param kernel_info The kernel info.
 * \note The result will write back to the `kernel_info` list.
 */
void UnifyKernels(std::vector<KernelInfo>* kernel_info) {
  arith::Analyzer analyzer;
  for (size_t start_pos = 0; start_pos < kernel_info->size();) {
    for (size_t end_pos = start_pos + 1; end_pos <= kernel_info->size(); ++end_pos) {
      if (end_pos == kernel_info->size() ||             //
          (*kernel_info)[end_pos].use_shared_memory ||  //
          !(*kernel_info)[start_pos].same_thread_config_as((*kernel_info)[end_pos], &analyzer)) {
        // We do not allowed to fuse kernels using shared memory or with different thread config.
        // But it's fine that only the first kernel uses shared memory.
        FuseKernel(kernel_info, start_pos, end_pos);
        start_pos = end_pos;
      }
    }
  }
}

/*! \brief The rewriter to update the AST according to the fused kernel info. */
class UnifyKernelRewriter : public StmtMutator {
 public:
  static Stmt Rewrite(Stmt stmt, const std::vector<KernelInfo>& kernel_info) {
    return UnifyKernelRewriter(kernel_info)(std::move(stmt));
  }

 private:
  explicit UnifyKernelRewriter(const std::vector<KernelInfo>& kernel_info)
      : kernel_info_(kernel_info) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // If this AttrStmt is not thread binding attribute, return as usual.
    if (op->attr_key != attr::thread_extent && op->attr_key != attr::virtual_thread) {
      return StmtMutator::VisitStmt_(op);
    } else {
      const IterVar& iter_var = Downcast<IterVar>(op->node);
      return RewriteImpl(iter_var->thread_tag, iter_var->var);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // If this For is not thread binding attribute, return as usual.
    if (op->kind != ForKind::kThreadBinding) {
      return StmtMutator::VisitStmt_(op);
    } else {
      return RewriteImpl(op->thread_binding.value()->thread_tag, op->loop_var);
    }
  }

  Stmt RewriteImpl(const String& thread_tag, const Var& var) {
    const KernelInfo& cur_kernel = kernel_info_[kernel_idx_++];

    // Double check the kernel index is matched.
    ICHECK(StartsWith(thread_tag, "blockIdx."));
    auto it = cur_kernel.thread_vars.find(thread_tag);
    ICHECK(it != cur_kernel.thread_vars.end());
    ICHECK((*it).second.same_as(var));
    Stmt body = cur_kernel.body;
    if (!is_no_op(body)) {
      // Re-emit thread bindings.
      for (const std::string& thread_tag : reverse_thread_emit_order) {
        auto it = cur_kernel.thread_vars.find(thread_tag);
        if (it != cur_kernel.thread_vars.end()) {
          const Range& dom = cur_kernel.thread_extents.at(thread_tag);
          body = For((*it).second, dom->min, dom->extent, ForKind::kThreadBinding, body,
                     IterVar(NullValue<Range>(), Var(""), IterVarType::kThreadIndex, thread_tag));
        }
      }
    }
    return body;
  }

 private:
  /*! \brief The kernel info*/
  const std::vector<KernelInfo>& kernel_info_;
  /*! \brief The current kernel index. */
  size_t kernel_idx_ = 0;
};

PrimFunc UnifyKernelLaunch(PrimFunc f) {
  if (IsFromLegacyTESchedule(f)) {
    return f;
  }
  // Step 1. Collect all thread bindings of all kernels.
  std::vector<KernelInfo> kernel_infos = KernelInfoCollector::Collect(f->body);
  // Fast pass: no need to fuse kernels if there is less than 2 kernels.
  if (kernel_infos.size() <= 1) {
    return f;
  }
  // Step 2. Try fuse nearby kernels with same thread bindings
  UnifyKernels(&kernel_infos);
  // Step 3. Replace the IR
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = UnifyKernelRewriter::Rewrite(std::move(f->body), kernel_infos);
  return f;
}

namespace transform {

Pass UnifyKernelLaunch() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return UnifyKernelLaunch(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.UnifyKernelLaunch", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UnifyKernelLaunch").set_body_typed(UnifyKernelLaunch);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
