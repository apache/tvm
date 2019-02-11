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
#include <tvm/schedule.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace contrib {

using namespace ir;
/*!
 * \brief A base class to generate Hybrid Script.
 *
 * **NOTE** CodeGenHybrid does not aim at generating Python scripts consumed by Python2/3.
 * For runtime support, please refer the decorator in ``tvm/python/hybrid/api.py``.
 */
class CodeGenHybrid :
      public ExprFunctor<void(const Expr&, std::ostream&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Dump the given schedule to hybrid script.
   * \param sch The schedule to be dumped to hybrid script.
   */
  void DumpSchedule(const Schedule &sch);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  std::string Finish();
  /*!
   * \brief Print the Stmt n to CodeGenHybrid->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt &n) {
    this->VisitStmt(n);
  }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const Expr &n, std::ostream &os) {
    this->VisitExpr(n, os);
  }
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const Expr &n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
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
  void VisitStmt_(const Provide* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const Realize* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
  /*!
   * \brief Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(Type t, std::ostream& os); // NOLINT(*)
  virtual std::string CastFromTo(std::string value, Type from, Type target);

 private:
  /*! \brief The current indent of the code dump. */
  int indent_{0};
  /*! \brief The tab size of code indent. */
  const int tab_{2};
  /*! \brief Print the current indent spaces. */
  inline void PrintIndent();
  /*! \brief Keys are ids allocated, and values are the suffix to prevent double-name.  */
  std::map<std::string, int> ids_allocated_;
  /*! \brief Keys are either tensors or variables. Values are the corresponding IDs.*/
  std::map<const Node *, std::string> id_map_;
  /*!
   * \brief Find an unallocated name for the given prefix.
   * \param prefix The given prefix.
   */
  std::string GetUniqueName(std::string prefix);
  /*! \brief The output code string builder. */
  std::stringstream stream;
  /*! 
   * \brief Get or allocate the ID for the given variable.
   * \param v The given variable.
   */
  std::string GetVarID(const Variable *v);
  /*! 
   * \brief Get or allocate the ID for the given tensor.
   * \param func The tensor to allocate a name.
   * \param value_index The value index of the given tensor.
   */
  std::string GetTensorID(const FunctionRef &func, int value_index);
  /*! \brief the storage scope of allocation */
  std::map<FunctionRef, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;
};

}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_HYBRID_H_
