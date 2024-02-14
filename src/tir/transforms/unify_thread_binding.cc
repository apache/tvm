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

#include "../../support/utils.h"
#include "ir_utils.h"

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
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // If this AttrStmt is not thread binding attribute, return as usual.
    if (op->attr_key != attr::thread_extent && op->attr_key != attr::virtual_thread) {
      return StmtMutator::VisitStmt_(op);
    }
    IterVar old_iter_var = Downcast<IterVar>(op->node);
    return UnifyThreadBindingImpl(op, old_iter_var->var, old_iter_var,
                                  Range::FromMinExtent(IntImm(op->value->dtype, 0), op->value));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // If this For is not thread binding attribute, return as usual.
    if (op->kind != ForKind::kThreadBinding) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Map<String, ObjectRef> annotations = op->annotations;
    Stmt stmt = UnifyThreadBindingImpl(op, op->loop_var, op->thread_binding.value(),
                                       Range::FromMinExtent(op->min, op->extent));
    if (annotations.empty()) {
      return stmt;
    }
    if (const auto* loop = stmt.as<ForNode>()) {
      For new_loop = GetRef<For>(loop);
      new_loop.CopyOnWrite()->annotations = std::move(annotations);
      return std::move(new_loop);
    } else {
      // Create a new unit loop with the annotation.
      DataType dtype = op->loop_var->dtype;
      return For(/*loop_var=*/Var("var", dtype),   //
                 /*min=*/IntImm(dtype, 0),         //
                 /*extent=*/IntImm(dtype, 1),      //
                 /*kind=*/ForKind::kSerial, stmt,  //
                 /*thread_binding=*/NullOpt,       //
                 /*annotation=*/std::move(annotations));
    }
  }

  template <typename Node>
  Stmt UnifyThreadBindingImpl(const Node* op, const Var& old_var, const IterVar& old_iter_var,
                              const Range& dom) {
    // Step 1. Fetch the thread tag.
    IterVar new_iter_var{nullptr};
    const String& thread_tag = old_iter_var->thread_tag;

    // Step 2: Increase `thread_block_depth_` if the thread tag starts with "blockIdx". If the
    // thread block depth is 0 before the increment, it means we are entering a new kernel, and
    // therefore we need to make `thread_tag2iter_var_map_` empty, as different kernels can have
    // thread axes with different extents.
    bool is_kernel_launch_scope = false;
    int old_thread_block_depth = thread_block_depth_;
    if (StartsWith(thread_tag, "blockIdx.") || !thread_block_depth_) {
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
      ICHECK(ana.CanProveEqual(dom->min, new_iter_var->dom->min));
      CHECK(ana.CanProveEqual(dom->extent, new_iter_var->dom->extent))
          << "ValueError: All loops that are bound to `" << thread_tag
          << "` should have the same extent. However, there are two loops with extent "
          << new_iter_var->dom->extent << " and " << dom->extent << ", which are not equal";
    } else {
      new_iter_var = IterVar(dom, Var(thread_tag, dom->extent.dtype()), old_iter_var->iter_type,
                             old_iter_var->thread_tag);
      thread_tag2iter_var_map_.Set(thread_tag, new_iter_var);
      launch_threads_.push_back(new_iter_var);
    }

    // Step 4. We will substitute the occurrences of the old variable in the old IterVar with the
    // new variable in further mutation. Thus, we store the mapping entry. Cast to old dtype if
    // needed (we assume both old and new dtype are valid for the range of the thread extent).
    var_substitution_map_.Set(old_var, cast(old_var.dtype(), new_iter_var->var));

    // Step 5. Mutate recursively, update the body with the new IterVar, and restore the depth
    // counter. Emit for-loops to launch threads if current statement is the outermost thread
    // binding of the kernel.
    Stmt new_stmt = StmtMutator::VisitStmt_(op);
    auto* new_node = new_stmt.as<Node>();
    ICHECK(new_node);
    thread_block_depth_ = old_thread_block_depth;
    if (is_kernel_launch_scope) {
      return EmitLaunchThreads(new_node->body);
    } else {
      return new_node->body;
    }
  }

  /*!
   * \brief Emit loop nests representing all thread bindings of the kernel
   * \param body The body of the innermost loop of the thread bindings.
   * \return The loop nests of the thread bindings.
   */
  Stmt EmitLaunchThreads(const Stmt& body) {
    Stmt result = body;
    while (!launch_threads_.empty()) {
      const IterVar& thread_binding = launch_threads_.back();
      // Recreate the IterVar as we don't duplicate `dom` in both For and IterVar. This is
      // necessary for unit tests.
      result = For(thread_binding->var, thread_binding->dom->min, thread_binding->dom->extent,
                   ForKind::kThreadBinding, result,
                   IterVar(NullValue<Range>(), Var(""), IterVarType::kThreadIndex,
                           thread_binding->thread_tag));
      launch_threads_.pop_back();
    }
    return result;
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    // If this variable appears as a key in `var_substitution_map_`, we substitute it with its
    // corresponding value in the mapping.
    Map<Var, PrimExpr>::iterator it = var_substitution_map_.find(GetRef<Var>(var));
    return it != var_substitution_map_.end() ? (*it).second : GetRef<Var>(var);
  }

  /*!
   * \brief A mapping from a thread tag to its corresponding IterVar that is shared by all
   * occurrences of the thread tag
   */
  Map<String, IterVar> thread_tag2iter_var_map_;
  /*!
   * \brief A list of IterVar corresponding to threads in current kernel. This will be used to
   * generate for-loops to launch threads.
   */
  Array<IterVar> launch_threads_;
  /*! \brief A mapping from old variables to new variables, which is used for substitution */
  Map<Var, PrimExpr> var_substitution_map_;
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
