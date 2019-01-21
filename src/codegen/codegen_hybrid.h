/*!
 *  Copyright (c) 2019 by Contributors
 * \file codegen_hybrid.h
 * \brief Common utilities to generated C style code.
 */
#ifndef TVM_CODEGEN_CODEGEN_HYBRID_H_
#define TVM_CODEGEN_CODEGEN_HYBRID_H_

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

using namespace ir;
/*!
 * \brief A base class to generate Python script.
 *
 *  Unlike C-family generators, CodeGenHybrid does not generate SSA formed Python script.
 *  Only normal form is supported.
 *
 * **NOTE** CodeGenHybrid does not aim at generating Python scripts consumed by Python2/3.
 * For runtime support, please refer the decorator in ``tvm/python/hybrid/api.py``.
 */
class CodeGenHybrid :
      public ExprFunctor<void(const Expr&, std::ostream&)>,
      public StmtFunctor<void(const Stmt&)>,
      public CodeGenSourceBase {
 public:
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
   * \brief Print the Stmt n to CodeGenHybrid->stream
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
  virtual void PrintType(Type t, std::ostream& os); // NOLINT(*)
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv); // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os); // NOLINT(*)
  virtual void PrintStorageSync(const Call* op);  // NOLINT(*)
  // Get a cast type from to
  virtual std::string CastFromTo(std::string value, Type from, Type target);

 protected:
  std::string GetVarID(const Variable* v);
  // Print reference to struct location
  std::string GetStructRef(
      Type t, const Expr& buffer, const Expr& index, int kind);
  // print reference to a buffer as type t in index.
  virtual std::string GetBufferRef(
      Type t, const Variable* buffer, Expr index);
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
  // override
  void PrintSSAAssign(
      const std::string& target, const std::string& src, Type t) final;
  /*! \brief restrict keyword */
  std::string restrict_keyword_{""};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;
  /*! \brief reserves common C keywords */
  void ReserveKeywordsAsUnique();

 private:
  /*! \brief whether print a simple form */
  bool simple_mode_{true};
  /*! \brief set of volatile buf access */
  std::unordered_set<const Variable*> volatile_buf_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_HYBRID_H_
