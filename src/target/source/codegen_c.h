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
 * \file codegen_c.h
 * \brief Common utilities to generated C style code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_C_H_
#define TVM_TARGET_SOURCE_CODEGEN_C_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/lowered_func.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

using namespace tir;
/*!
 * \brief A base class to generate C code.
 *
 *  CodeGenC have two modes: generate SSA formed C code or normal form.
 *
 * **NOTE** CodeGenC does not aim at generating C codes consumed by MSVC or GCC,
 * Rather, it's providing infrastructural abstraction for C variants like CUDA
 * and OpenCL-C. You might find some odd variant features, e.g., type `int3` for
 * a vector of 3 `int`s. For native C code generator, see `CodeGenLLVM`.
 */
class CodeGenC :
      public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
      public StmtFunctor<void(const Stmt&)>,
      public CodeGenSourceBase {
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
  void PrintExpr(const PrimExpr& n, std::ostream& os);
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const PrimExpr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  // The following parts are overloadable print operations.
  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PreFunctionBody(LoweredFunc f) {}
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(LoweredFunc f);
  // expression
  void VisitExpr_(const VarNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LoadNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;  // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const ProducerConsumerNode* op) override;
  /*!
   * Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(DataType t, std::ostream& os); // NOLINT(*)
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv); // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os); // NOLINT(*)
  virtual void PrintStorageSync(const CallNode* op);  // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(
      const std::string&op, DataType op_type,
      PrimExpr lhs, PrimExpr rhs, std::ostream& os);  // NOLINT(*)
  // print vector load
  virtual std::string GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base);
  // print vector store
  virtual void PrintVecStore(const VarNode* buffer,
                             DataType t, PrimExpr base,
                             const std::string& value);  // NOLINT(*)
  // print load of single element
  virtual void PrintVecElemLoad(
      const std::string& vec, DataType t, int i, std::ostream& os);  // NOLINT(*)
  // print store of single element.
  virtual void PrintVecElemStore(
      const std::string& vec, DataType t, int i, const std::string& value);
  // Get a cast type from to
  virtual std::string CastFromTo(std::string value, DataType from, DataType target);

 protected:
  // Print reference to struct location
  std::string GetStructRef(
      DataType t, const PrimExpr& buffer, const PrimExpr& index, int kind);
  // print reference to a buffer as type t in index.
  virtual std::string GetBufferRef(
      DataType t, const VarNode* buffer, PrimExpr index);
  /*!
   * \brief If buffer is allocated as type t.
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  bool HandleTypeMatch(const VarNode* buf_var, DataType t) const;
  /*!
   * \brief Register the data type of buf_var
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  void RegisterHandleType(const VarNode* buf_var, DataType t);
  // override
  void PrintSSAAssign(
      const std::string& target, const std::string& src, DataType t) final;
  /*! \brief restrict keyword */
  std::string restrict_keyword_{""};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const VarNode*, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const VarNode*, DataType> handle_data_type_;
  /*! \brief reserves common C keywords */
  void ReserveKeywordsAsUnique();

 private:
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{false};
  /*! \brief set of volatile buf access */
  std::unordered_set<const VarNode*> volatile_buf_;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SOURCE_CODEGEN_C_H_
