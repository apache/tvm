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
 * \file codegen_stack_vm.h
 * \brief Codegen into Simple Stack VM.
 */
#ifndef TVM_TARGET_STACKVM_CODEGEN_STACKVM_H_
#define TVM_TARGET_STACKVM_CODEGEN_STACKVM_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/target/codegen.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "../../runtime/stackvm/stackvm.h"

namespace tvm {
namespace codegen {

using namespace tir;
using runtime::StackVM;

/*!
 * \brief A base class to generate a stack VM.
 *  This module is used to generate host wrapper
 *  into device function when only device JIT is available.
 */
class CodeGenStackVM
    : public ExprFunctor<void(const PrimExpr&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
 /*!
   * \brief Generate a stack VM representing
   * \param f The function to be compiled
   * \param device_funcs The extern device functions to be linked.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  StackVM Compile(const PrimFunc& f);
  /*! \brief Push stmt to generate new code */
  void Push(const Stmt& n);
  /*! \brief Push expr to generate new code */
  void Push(const PrimExpr& n) {
    VisitExpr(n);
  }
  /*!
   * \brief Push the opcode to the code.
   * \param opcode The code to be pushed.
   */
  void PushOp(StackVM::OpCode opcode);
  /*!
   * \brief Push the opcode and operand to the code.
   * \param opcode The opcode.
   * \param operand The operand to be pushed.
   * \return operand_index, indicating location of operand
   */
  int64_t PushOp(StackVM::OpCode opcode, int operand);
  /*!
   * \brief Set the relative jump offset to be offset.
   * \param operand_index The indexed returned by PushOp.
   * \param operand The operand to be set.
   */
  void SetOperand(int64_t operand_index, int64_t operand);
  /*! \return The current program pointer */
  int64_t GetPC() const {
    return static_cast<int64_t>(vm_.code.size());
  }
  /*!
   * \brief Get string id in vm
   * \param key The string to get id.
   * \return the id of the string.
   */
  int GetStrID(const std::string& key);
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the heap index of the var.
   */
  int AllocVarID(const VarNode* v);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the heap index of the var.
   */
  int GetVarID(const VarNode* v) const;
  // Push binary operator
  void PushBinary(StackVM::OpCode op_int64,
                  const PrimExpr& a,
                  const PrimExpr& b);
  // push cast;
  void PushCast(DataType dst, DataType src);
  // overloadable functions
  // expression
  void VisitExpr_(const VarNode* op) final;
  void VisitExpr_(const LoadNode* op) final;
  void VisitExpr_(const LetNode* op) final;
  void VisitExpr_(const CallNode* op) final;
  void VisitExpr_(const AddNode* op) final;
  void VisitExpr_(const SubNode* op) final;
  void VisitExpr_(const MulNode* op) final;
  void VisitExpr_(const DivNode* op) final;
  void VisitExpr_(const ModNode* op) final;
  void VisitExpr_(const MinNode* op) final;
  void VisitExpr_(const MaxNode* op) final;
  void VisitExpr_(const EQNode* op) final;
  void VisitExpr_(const NENode* op) final;
  void VisitExpr_(const LTNode* op) final;
  void VisitExpr_(const LENode* op) final;
  void VisitExpr_(const GTNode* op) final;
  void VisitExpr_(const GENode* op) final;
  void VisitExpr_(const AndNode* op) final;
  void VisitExpr_(const OrNode* op) final;
  void VisitExpr_(const CastNode* op) final;
  void VisitExpr_(const NotNode* op) final;
  void VisitExpr_(const SelectNode* op) final;
  void VisitExpr_(const RampNode* op) final;
  void VisitExpr_(const BroadcastNode* op) final;
  void VisitExpr_(const IntImmNode* op) final;
  void VisitExpr_(const FloatImmNode* op) final;
  void VisitExpr_(const StringImmNode* op) final;
  // statment
  void VisitStmt_(const LetStmtNode* op) final;
  void VisitStmt_(const StoreNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const IfThenElseNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;
  void VisitStmt_(const AssertStmtNode* op) final;
  void VisitStmt_(const EvaluateNode* op) final;
  void VisitStmt_(const SeqStmtNode* op) final;

 private:
  bool debug_{false};
  /*! \brief The vm to be generated */
  StackVM vm_;
  /*! \brief id of each variable */
  std::unordered_map<const VarNode*, int> var_idmap_;
  /*! \brief id of each string */
  std::unordered_map<std::string, int> str_idmap_;
  /*! \brief id of each global function */
  std::unordered_map<std::string, int> extern_fun_idmap_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_STACKVM_CODEGEN_STACKVM_H_
