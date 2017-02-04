/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen_c.h
 * \brief Common utilities to generated C style code.
 */
#ifndef TVM_CODEGEN_CODEGEN_C_H_
#define TVM_CODEGEN_CODEGEN_C_H_

#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace codegen {

/*!
 * \brief A base class to generate C code.
 *
 *  CodeGenC have two modes: generate SSA formed C code or normal form.
 */
class CodeGenC {
 public:
  /*!
   * \brief Generate the C code of statement
   * \param f The function to be compiled
   * \param output_ssa Whether output ssa form.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  std::string Compile(LoweredFunc f,
                      bool output_ssa);
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n);
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const Expr& n, std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const Expr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  /*! \brief print the current indented value */
  void PrintIndent();
  /*!
   * \brief Register constant value appeared in expresion tree
   *  This avoid generated a ssa id for each appearance of the value
   * \param value The constant value.
   */
  void MarkConst(std::string value);
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the variable name.
   */
  std::string AllocVarID(const Variable* v);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the variable name.
   */
  std::string GetVarID(const Variable* v) const;
  // The following parts are overloadable print operations.
  /*!
   * Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(Type t, std::ostream& os) const; // NOLINT(*)
  /*!
   * \brief Print expr representing the thread tag
   * \param tag The tag in the thread.
   * \param  os The strean to output to
   */
  virtual void PrintThreadIndexExpr(
      std::string tag, std::ostream& os); // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*
  virtual void PrintStorageSync(const std::string& scope);  // NOLINT(*)

  virtual void PrintStmt(const ir::LetStmt* op);
  virtual void PrintStmt(const ir::Store* op);
  virtual void PrintStmt(const ir::For* op);
  virtual void PrintStmt(const ir::IfThenElse* op);
  virtual void PrintStmt(const ir::Allocate* op);
  virtual void PrintStmt(const ir::AttrStmt* op);
  virtual void PrintStmt(const ir::AssertStmt* op);
  virtual void PrintExpr(const ir::Load* op, std::ostream& os);  // NOLINT(*)
  virtual void PrintExpr(const ir::Call* op, std::ostream& os);  // NOLINT(*)
  virtual void PrintExpr(const ir::Let* op, std::ostream& os);  // NOLINT(*)
  virtual void PrintExpr(const ir::Ramp* op, std::ostream& os);  // NOLINT(*)
  virtual void PrintExpr(const ir::Broadcast* op, std::ostream& os);  // NOLINT(*)
  virtual void PrintExpr(const ir::Select* op, std::ostream& os);  // NOLINT(*)
  /*! \brief function print into the ostream */
  using FPrintExpr = IRFunctor<void(const NodeRef&, std::ostream& os, CodeGenC *)>; // NOLINT(*)
  /*! \brief function to to print normal code */
  using FPrintStmt = IRFunctor<void(const NodeRef&, CodeGenC *)>;
  // vtable to print code
  static FPrintStmt& vtable_print_stmt();
  // vtable to print code
  static FPrintExpr& vtable_print_expr();
  /*! \brief The current indentation value */
  int indent{0};
  /*! \brief the stream to be printed */
  std::ostringstream stream;

 protected:
  // additional string for arg addr_space.
  std::string arg_addr_space_;

 private:
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    std::string vid;
    /*! \brief The scope id */
    int scope_id;
  };
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  std::string SSAGetID(std::string src, Type t);
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope();
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id);
  /*!
   * \brief If buffer is allocated as type t.
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  bool HandleTypeMatch(const Variable* buf_var, Type t) const;
  /*!
   * \brief Register the data type of buf_var
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  void HandleTypeRegister(const Variable* buf_var, Type t);
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  std::string GetUniqueName(std::string prefix);
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{true};
  /*! \brief name of each variable */
  std::unordered_map<const Variable*, std::string> var_idmap_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief assignment map of ssa */
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_C_H_
