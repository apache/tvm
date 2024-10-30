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
 * \file codegen_hybrid.h
 * \brief Common utilities to generated C style code.
 */
#ifndef TVM_CONTRIB_HYBRID_CODEGEN_HYBRID_H_
#define TVM_CONTRIB_HYBRID_CODEGEN_HYBRID_H_

#include <tvm/ir/name_supply.h>
#include <tvm/target/codegen.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace contrib {

using namespace te;
using namespace tir;
/*!
 * \brief A base class to generate Hybrid Script.
 *
 * **NOTE** CodeGenHybrid does not aim at generating Python scripts consumed by Python2/3.
 * For runtime support, please refer the decorator in ``tvm/python/hybrid/api.py``.
 */
class CodeGenHybrid : public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
                      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Dump the given function body to hybrid script.
   * \param stmt The function body to be dumped to hybrid script.
   * \param inputs Input tensors of this schedule.
   * \param outputs Output tensors of this schedule.
   * \param name The name of the function.
   */
  void DumpStmt(const Stmt& stmt, const Array<ObjectRef>& inputs, const Array<Tensor>& outputs,
                const std::string& name = "hybrid_func");
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  std::string Finish();
  /*! \brief Reserve keywords in avoid of name conflict. */
  void ReserveKeywords();
  /*!
   * \brief Print the Stmt n to CodeGenHybrid->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { this->VisitStmt(n); }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const PrimExpr& n, std::ostream& os) { this->VisitExpr(n, os); }
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const PrimExpr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  // expression
  void VisitExpr_(const VarNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;    // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const ProducerLoadNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const FloorDivNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const FloorModNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) override;            // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os) override;           // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;     // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ProducerStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const ProducerRealizeNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  /*!
   * \brief Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(DataType t, std::ostream& os);  // NOLINT(*)

 private:
  /*! \brief The current indent of the code dump. */
  int indent_{0};
  /*! \brief The tab size of code indent. */
  const int tab_{4};
  /*! \brief Print the current indent spaces. */
  inline void PrintIndent();
  /*! \brief NameSupply for allocated ids.  */
  NameSupply ids_allocated;
  /*!
   * \brief Keys are either (tensors, value_index) or (variables, 0).
   *        Values are the corresponding IDs.*/
  std::map<std::pair<const Object*, int>, std::string> id_map_;
  /*! \brief Variables (keys) binded to the threads (values). */
  std::map<const VarNode*, std::string> binds_;
  /*! \brief The output code string builder. */
  std::stringstream stream;
  /*!
   * \brief Get or allocate the ID for the given variable.
   * \param v The given variable.
   */
  std::string GetVarID(const VarNode* v);
  /*!
   * \brief Get or allocate the ID for the given tensor.
   * \param tensor The tensor to allocate a name.
   */
  std::string GetTensorID(const Tensor& tensor);
};

}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_HYBRID_CODEGEN_HYBRID_H_
