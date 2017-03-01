/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen_c.h
 * \brief Common utilities to generated C style code.
 */
#ifndef TVM_CODEGEN_CODEGEN_C_H_
#define TVM_CODEGEN_CODEGEN_C_H_

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace codegen {

using namespace ir;
/*!
 * \brief A base class to generate C code.
 *
 *  CodeGenC have two modes: generate SSA formed C code or normal form.
 */
class CodeGenC :
      public ExprFunctor<void(const Expr&, std::ostream&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init(bool output_ssa);
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   */
  void AddFunction(LoweredFunc f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  std::string Finish();
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) {
    VisitStmt(n);
  }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const Expr& n, std::ostream& os);
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
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(LoweredFunc f);
  // expression
  void VisitExpr_(const Variable* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Load* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Let* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Call* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Add* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Sub* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Mul* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Div* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Mod* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Min* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Max* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const EQ* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const NE* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LT* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LE* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const GT* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const GE* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const And* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Or* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Cast* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Not* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Select* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Ramp* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const Broadcast* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const IntImm* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const UIntImm* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const FloatImm* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const StringImm* op, std::ostream& os) override;  // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
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
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const std::string& scope);  // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(
      const std::string&op, Type op_type,
      Expr lhs, Expr rhs, std::ostream& os);  // NOLINT(*)
  // print vector load
  virtual void PrintVecLoad(const Variable* buffer,
                            Type t, Expr base,
                            std::ostream& os);  // NOLINT(*)
  // print vector store
  virtual void PrintVecStore(const Variable* buffer,
                             Type t, Expr base,
                             const std::string& value);  // NOLINT(*)
  // print load of single element
  virtual void PrintVecElemLoad(
      const std::string& vec, Type t, int i, std::ostream& os);  // NOLINT(*)
  // print store of single element.
  virtual void PrintVecElemStore(
      const std::string& vec, Type t, int i, const std::string& value);

 protected:
  /*! \brief the stream to be printed */
  std::ostringstream stream;
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    std::string vid;
    /*! \brief The scope id */
    int scope_id;
  };
  // print reference to a buffer as type t in index.
  void PrintBufferRef(const Variable* buffer,
                      Type t, Expr index,
                      std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  std::string SSAGetID(std::string src, Type t);
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  std::string GetUniqueName(std::string prefix);
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
  void RegisterHandleType(const Variable* buf_var, Type t);
  /*!
   * \brief Get the storage scope of buf_var.
   * \param buf_var The buf_var to be queryed.
   * \return The storage scope.
   */
  std::string GetStorageScope(const Variable* buf_var) const;
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;

 private:
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{true};
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief assignment map of ssa */
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_;
  /*! \brief name of each variable */
  std::unordered_map<const Variable*, std::string> var_idmap_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
  /*! \brief The current indentation value */
  int indent{0};
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_C_H_
