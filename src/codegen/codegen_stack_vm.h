/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_stack_vm.h
 * \brief Codegen into Simple Stack VM.
 */
#ifndef TVM_CODEGEN_CODEGEN_STACK_VM_H_
#define TVM_CODEGEN_CODEGEN_STACK_VM_H_

#include <tvm/ir.h>
#include <tvm/lowered_func.h>
#include <tvm/codegen.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "../runtime/stack_vm/stack_vm.h"

namespace tvm {
namespace codegen {

using runtime::StackVM;

/*!
 * \brief A base class to generate a stack VM.
 *  This module is used to generate host wrapper
 *  into device function when only device JIT is available.
 */
class CodeGenStackVM {
 public:
 /*!
   * \brief Generate a stack VM representing
   * \param f The function to be compiled
   * \param device_funcs The extern device functions to be linked.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  StackVM Compile(
      LoweredFunc f,
      const std::unordered_map<LoweredFunc, PackedFunc>& device_funcs);
  /*! \brief Push stmt to generate new code */
  void Push(const Stmt& n);
    /*! \brief Push expr to generate new code */
  void Push(const Expr& n);
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
   * \brief Push a call packed function.
   * \param fid The function id.
   * \param arg_type_codes The type codes of arguments.
   */
  void PushCallPacked(int fid,
                      const std::vector<int>& arg_type_codes);
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
  // overloadable functions
  virtual void Push_(const ir::Load* op);
  virtual void Push_(const ir::Store* op);
  virtual void Push_(const ir::Allocate* op);
  virtual void Push_(const ir::Call* op);
  virtual void HandleUnknownCall(const ir::Call* op);
  /*! \brief function to to print normal code */
  using FType = IRFunctor<void(const NodeRef&, CodeGenStackVM *)>;
  // vtable to print code
  static FType& vtable();  // NOLINT(*)

 private:
  bool debug_{false};
  /*! \brief The vm to be generated */
  StackVM vm_;
  /*! \brief id of each variable */
  std::unordered_map<const Variable*, int> var_idmap_;
  /*! \brief id of each string */
  std::unordered_map<std::string, int> str_idmap_;
  /*! \brief id of each global function */
  std::unordered_map<std::string, int> global_fun_idmap_;
  /*! \brief id of device function */
  std::unordered_map<LoweredFunc, int> device_fun_idmap_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_STACK_VM_H_
