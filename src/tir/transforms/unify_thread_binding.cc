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

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief A mutator which searches AttrStmts of thread bindings and changes the `node` field IterVar
 * of the AttrStmts, so that for one kind of thread binding (except "vthread"), all such thread
 * bindings use the same IterVar
 */
class ThreadBindingUnifier : public StmtExprMutator {
 public:
  static Stmt Unify(const PrimFunc& f) { return ThreadBindingUnifier().VisitStmt(f->body); }

 private:
  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    // If this AttrStmt is not thread binding attribute, return as usual.
    if (attr->attr_key != attr::thread_extent && attr->attr_key != attr::virtual_thread) {
      return StmtMutator::VisitStmt_(attr);
    }

    // Step 1. Fetch the old IterVar.
    IterVar old_iter_var = Downcast<IterVar>(attr->node);
    IterVar new_iter_var;

    // Step 2. See if an IterVar for this kind of thread binding was created before. If so, we use
    // the created IterVar. Otherwise, we create a new IterVar for this thread binding and store the
    // IterVar in mapping `thread_tag2iter_var_map_`.
    Map<String, IterVar>::iterator it = thread_tag2iter_var_map_.find(old_iter_var->thread_tag);
    if (it != thread_tag2iter_var_map_.end()) {
      new_iter_var = (*it).second;
    } else {
      ObjectPtr<IterVarNode> p_new_iter_var = make_object<IterVarNode>(*old_iter_var.get());
      p_new_iter_var->var = Var(old_iter_var->thread_tag);
      new_iter_var = IterVar(p_new_iter_var);
      // We don't unify thread bindings of "vthread".
      if (old_iter_var->thread_tag != "vthread") {
        thread_tag2iter_var_map_.Set(old_iter_var->thread_tag, new_iter_var);
      }
    }

    // Step 3. We will substitute the occurrences of the old variable in the old IterVar with the
    // new variable in further mutation. Thus, we store the mapping entry.
    var_substitution_map_.Set(old_iter_var->var, new_iter_var->var);

    // Step 4. Mutate recursively, and update the AttrStmt with the new IterVar.
    AttrStmt new_attr = Downcast<AttrStmt>(StmtMutator::VisitStmt_(attr));
    ObjectPtr<AttrStmtNode> p_new_attr = CopyOnWrite(new_attr.get());
    p_new_attr->node = new_iter_var;
    return Stmt(p_new_attr);
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    // If this variable appears as a key in `var_substitution_map_`, we substitute it with its
    // corresponding value in the mapping.
    Map<Var, Var>::iterator it = var_substitution_map_.find(GetRef<Var>(var));
    return it != var_substitution_map_.end() ? (*it).second : GetRef<Var>(var);
  }

  /*!
   * \brief A mapping from a thread tag to its corresponding IterVar that is shared by all
   * occurrences of the thread tag
   * */
  Map<String, IterVar> thread_tag2iter_var_map_;
  /*! \brief A mapping from old variables to new variables, which is used for substitution */
  Map<Var, Var> var_substitution_map_;
};

PrimFunc UnifyThreadBinding(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = ThreadBindingUnifier::Unify(f);
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
