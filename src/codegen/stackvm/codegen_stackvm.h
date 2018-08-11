/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_stack_vm.h
 * \brief Codegen into Simple Stack VM.
 */
#ifndef TVM_CODEGEN_STACKVM_CODEGEN_STACKVM_H_
#define TVM_CODEGEN_STACKVM_CODEGEN_STACKVM_H_

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/lowered_func.h>
#include <tvm/codegen.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "../../runtime/stackvm/stackvm.h"

namespace tvm {
namespace codegen {

using namespace ir;
using runtime::StackVM;

/*!
 * \brief A base class to generate a stack VM.
 *  This module is used to generate host wrapper
 *  into device function when only device JIT is available.
 */
class CodeGenStackVM
    : public ExprFunctor<void(const Expr&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
 /*!
   * \brief Generate a stack VM representing
   * \param f The function to be compiled
   * \param device_funcs The extern device functions to be linked.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  StackVM Compile(LoweredFunc f);
  /*! \brief Push stmt to generate new code */
  void Push(const Stmt& n);
  /*! \brief Push expr to generate new code */
  void Push(const Expr& n) {
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
  int AllocVarID(const Variable* v);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the heap index of the var.
   */
  int GetVarID(const Variable* v) const;
  // Push binary operator
  void PushBinary(StackVM::OpCode op_int64,
                  const Expr& a,
                  const Expr& b);
  // push cast;
  void PushCast(Type dst, Type src);
  // overloadable functions
  // expression
  void VisitExpr_(const Variable* op) final;
  void VisitExpr_(const Load* op) final;
  void VisitExpr_(const Let* op) final;
  void VisitExpr_(const Call* op) final;
  void VisitExpr_(const Add* op) final;
  void VisitExpr_(const Sub* op) final;
  void VisitExpr_(const Mul* op) final;
  void VisitExpr_(const Div* op) final;
  void VisitExpr_(const Mod* op) final;
  void VisitExpr_(const Min* op) final;
  void VisitExpr_(const Max* op) final;
  void VisitExpr_(const EQ* op) final;
  void VisitExpr_(const NE* op) final;
  void VisitExpr_(const LT* op) final;
  void VisitExpr_(const LE* op) final;
  void VisitExpr_(const GT* op) final;
  void VisitExpr_(const GE* op) final;
  void VisitExpr_(const And* op) final;
  void VisitExpr_(const Or* op) final;
  void VisitExpr_(const Cast* op) final;
  void VisitExpr_(const Not* op) final;
  void VisitExpr_(const Select* op) final;
  void VisitExpr_(const Ramp* op) final;
  void VisitExpr_(const Broadcast* op) final;
  void VisitExpr_(const IntImm* op) final;
  void VisitExpr_(const UIntImm* op) final;
  void VisitExpr_(const FloatImm* op) final;
  void VisitExpr_(const StringImm* op) final;
  // statment
  void VisitStmt_(const LetStmt* op) final;
  void VisitStmt_(const Store* op) final;
  void VisitStmt_(const For* op) final;
  void VisitStmt_(const IfThenElse* op) final;
  void VisitStmt_(const Allocate* op) final;
  void VisitStmt_(const AttrStmt* op) final;
  void VisitStmt_(const AssertStmt* op) final;
  void VisitStmt_(const Evaluate* op) final;
  void VisitStmt_(const Block* op) final;
  void VisitStmt_(const ProducerConsumer* op) final;

 private:
  bool debug_{false};
  /*! \brief The vm to be generated */
  StackVM vm_;
  /*! \brief id of each variable */
  std::unordered_map<const Variable*, int> var_idmap_;
  /*! \brief id of each string */
  std::unordered_map<std::string, int> str_idmap_;
  /*! \brief id of each global function */
  std::unordered_map<std::string, int> extern_fun_idmap_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_STACKVM_CODEGEN_STACKVM_H_
