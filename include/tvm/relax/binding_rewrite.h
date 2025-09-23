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
 * \file tvm/relax/binding_rewrite.h
 * \brief An IR rewriter to easily add/remove/replace bindings (statements).
 */

#ifndef TVM_RELAX_BINDING_REWRITE_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>

#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/*! \brief Statement rewriter for relax.DataflowBlock. */
class DataflowBlockRewriteNode : public Object {
 public:
  /*! \brief Replace all uses of old_var with new_var. */
  void ReplaceAllUses(Var old_var, Var new_var);
  /*! \brief Insert a Binding statement. */
  void Add(Binding binding);
  /*! \brief Insert an expression as VarBinding with variable name. */
  void Add(ffi::String var_name, Expr expr, bool is_dfvar = false) {
    auto var = is_dfvar ? DataflowVar(var_name, GetStructInfo(expr))  //
                        : Var(var_name, GetStructInfo(expr));
    Add(VarBinding(std::move(var), std::move(expr)));
  }
  /*! \brief Insert an expression as VarBinding with automatic variable name. */
  void Add(Expr expr, bool is_dfvar = false) {
    Add(name_supply_->FreshName("tmp"), expr, is_dfvar);
  }
  /*! \brief Remove the definition statement of an unused variable. */
  void RemoveUnused(Var unused, bool allow_undef = false);
  /*! \brief Remove the definition statements of all unused variables. */
  void RemoveAllUnused();

  /*! \brief The rewritten dataflow block. */
  DataflowBlock MutatedDataflowBlock() { return dfb_; }
  /*! \brief The rewritten function. */
  Function MutatedFunc() { return root_fn_.value(); }
  /*! \brief The rewritten IRModule. */
  IRModule MutateIRModule(IRModule irmod);

  /*! \brief Visit attributes. */
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DataflowBlockRewriteNode>()
        .def_ro("dfb", &DataflowBlockRewriteNode::dfb_)
        .def_ro("root_fn", &DataflowBlockRewriteNode::root_fn_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DataflowBlockRewrite", DataflowBlockRewriteNode, Object);

 protected:
  friend class DataflowBlockRewrite;

  DataflowBlock dfb_;                        //!< The rewritten dataflow block.
  ffi::Optional<Function> root_fn_;          //!< The rewritten function.
  const FunctionNode* original_fn_ptr_;      //!< Pointer to the original function.
  ffi::Map<Var, ffi::Array<Var>> to_users_;  //!< Map from variable to its users.
  ffi::Array<Var> fn_outputs_;               //!< Variables required by function outputs.

 private:
  NameSupply name_supply_;  //!< Name supply for tracking and generating unique names.
};

/*!
 * \brief A statement rewriter for relax.DataflowBlock.
 * \sa DataflowBlockRewriteNode
 */
class DataflowBlockRewrite : public ObjectRef {
 public:
  TVM_DLL explicit DataflowBlockRewrite(DataflowBlock dfb, Function root_fn);

  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  DataflowBlockRewriteNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<DataflowBlockRewriteNode*>(get_mutable());
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DataflowBlockRewrite, ObjectRef,
                                             DataflowBlockRewriteNode);
};

}  // namespace relax
}  // namespace tvm

#define TVM_RELAX_BINDING_REWRITE_H_
#endif  // TVM_RELAX_BINDING_REWRITE_H_
