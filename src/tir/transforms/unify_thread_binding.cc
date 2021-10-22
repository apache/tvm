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
 * \file unify_thread_binding.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"
#include "../../support/utils.h"

namespace tvm {
namespace tir {

using support::StartsWith;

/*!
 * \brief A mutator which searches AttrStmts of thread bindings and changes the `node` field IterVar
 * of the AttrStmts, so that for one kind of thread binding, all such thread bindings use the same
 * IterVar
 */
class ThreadBindingUnifier : public StmtExprMutator {
 public:
  static Stmt Unify(Stmt stmt) { return ThreadBindingUnifier()(std::move(stmt)); }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind != ForKind::kThreadBinding || StartsWith(op->thread_binding.value()->thread_tag, "vthread")) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Step 1. Fetch the old IterVar and the thread tag.
    IterVar old_iter_var = op->thread_binding.value();
    IterVar new_iter_var{nullptr};
    const String& thread_tag = op->thread_binding.value()->thread_tag;

    // Step 2: Increase `thread_block_depth_` if the thread tag starts with "blockIdx". If the
    // thread block depth is 0 before the increasement, it means we are entering a new kernel, and
    // therefore we need to make `thread_tag2iter_var_map_` empty, as different kernels can have
    // thread axes with different extents.
    bool is_kernel_launch_scope = false;
    int old_thread_block_depth = thread_block_depth_;
    if (StartsWith(thread_tag, "blockIdx.") || (StartsWith(thread_tag, "threadIdx.") && !thread_block_depth_)) {
      if (!thread_block_depth_) {
        thread_tag2iter_var_map_.clear();
        is_kernel_launch_scope = true;
      }
      ++thread_block_depth_;
    }

    // Step 3. See if an IterVar for this kind of thread binding was created before. If so, we use
    // the created IterVar. Otherwise, we create a new IterVar for this thread binding and store the
    // IterVar in mapping `thread_tag2iter_var_map_`.
    Map<String, IterVar>::iterator it = thread_tag2iter_var_map_.find(thread_tag);
    if (it != thread_tag2iter_var_map_.end()) {
      new_iter_var = (*it).second;
      CHECK(ana.CanProveEqual(op->extent, new_iter_var->dom->extent))
          << "ValueError: All loops that are bound to `" << thread_tag
          << "` should have the same extent. However, there are two loops with extent "
          << new_iter_var->dom->extent << " and " << op->extent
          << ", which are not equal";
    } else {
      ObjectPtr<IterVarNode> p_new_iter_var = make_object<IterVarNode>(*old_iter_var.get());
      p_new_iter_var->var = Var(thread_tag);
      p_new_iter_var->dom = Range::FromMinExtent(op->min, op->extent);
      new_iter_var = IterVar(p_new_iter_var);
      thread_tag2iter_var_map_.Set(thread_tag, new_iter_var);
      launch_threads_.push_back(new_iter_var);
    }

    // Step 4. We will substitute the occurrences of the old variable in the old IterVar with the
    // new variable in further mutation. Thus, we store the mapping entry.
    var_substitution_map_.Set(op->loop_var, new_iter_var->var);

    // Step 5. Mutate recursively, update the AttrStmt with the new IterVar, and decrease the depth
    // counter if the thread tag starts with "blockIdx".
    For for_node = Downcast<For>(StmtMutator::VisitStmt_(op));
    thread_block_depth_ = old_thread_block_depth;
    if (is_kernel_launch_scope) {
      Stmt result = for_node->body;
      while (!launch_threads_.empty()) {
        const IterVar& thread_binding = launch_threads_.back();
        result = For(thread_binding->var, thread_binding->dom->min, thread_binding->dom->extent, ForKind::kThreadBinding, result, thread_binding);
        launch_threads_.pop_back();
      }
      return result;
    } else {
      return for_node->body;
    }
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    // If this variable appears as a key in `var_substitution_map_`, we substitute it with its
    // corresponding value in the mapping.
    Map<Var, Var>::iterator it = var_substitution_map_.find(GetRef<Var>(var));
    return it != var_substitution_map_.end() ? (*it).second : GetRef<Var>(var);
  }

  /*!
   * \brief A mapping from a thread tag to its corresponding launching for-loop that is shared by all
   * occurrences of the thread tag
   * */
  Map<String, IterVar> thread_tag2iter_var_map_;
  Array<IterVar> launch_threads_;
  /*! \brief A mapping from old variables to new variables, which is used for substitution */
  Map<Var, Var> var_substitution_map_;
  /*! \brief A integer counter storing the depth of thread bindings of "blockIdx.x/y/z" */
  int thread_block_depth_ = 0;
  /*! \brief An analyzer used for equality proof */
  arith::Analyzer ana;
};

PrimFunc UnifyThreadBinding(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = ThreadBindingUnifier::Unify(std::move(f->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass UnifyThreadBinding() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return UnifyThreadBinding(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.UnifyThreadBinding", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UnifyThreadBinding").set_body_typed(UnifyThreadBinding);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
