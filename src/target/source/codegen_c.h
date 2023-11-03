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

#include <tvm/ir/op.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../tir/transforms/ir_utils.h"
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
class CodeGenC : public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
                 public StmtFunctor<void(const Stmt&)>,
                 public CodeGenSourceBase {
 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init(bool output_ssa);

  /*!
   * \brief Add the function declaration to the generated module,
   * without defining it.
   *
   * \param gvar The GlobalVar representing the function.
   * \param func The function to be compiled.
   * \param whether to append return 0 in the end.
   */
  virtual void DeclareFunction(const GlobalVar& gvar, const PrimFunc& func);

  /*!
   * \brief Add the function to the generated module, including its
   * declaration and definition.
   *
   * \param gvar The GlobalVar representing the function.
   * \param func The function to be compiled.
   */
  virtual void AddFunction(const GlobalVar& gvar, const PrimFunc& func);

  /*!
   * \brief Get the name of a declared function
   * \param gvar The GlobalVar of the function
   * \returns The string name of the function
   */
  String GetFunctionName(const GlobalVar& gvar);

  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  virtual std::string Finish();
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { VisitStmt(n); }
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

  /*! \brief Print the function signature before the argument list
   *
   * The default implementation delegates out to PrintFuncPrefix and
   * PrintExtraAttrs.
   *
   * \param function_name The name of the function
   *
   * \param func The function whose signature should be printed
   *
   * \param os The output stream
   */
  virtual void PrintFunctionSignature(const String& function_name, const PrimFunc& func,
                                      std::ostream& os);

  /*!
   * \brief Print the function header before the argument list
   * \param os The output stream
   *
   *  Example: stream << "void";
   */
  virtual void PrintFuncPrefix(std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Print extra function attributes
   *
   *  Example: __launch_bounds__(256) for CUDA functions
   */
  virtual void PrintExtraAttrs(const PrimFunc& f, std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PreFunctionBody(const PrimFunc& f) {}
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(const PrimFunc& f);
  // expression
  void VisitExpr_(const VarNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) override;          // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;      // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;    // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;   // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const WhileNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const AllocateConstNode* op) override;
  void VisitStmt_(const DeclBufferNode* op) override;

  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv);                             // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const CallNode* op);                           // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(const std::string& op, DataType op_type, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os);  // NOLINT(*)
  // print vector load
  virtual std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base);
  // print vector store
  virtual void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                             const std::string& value);  // NOLINT(*)
  // print load of single element
  virtual void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os);  // NOLINT(*)
  // print store of single element.
  virtual void PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value);
  // print vector constructor
  virtual void PrintVecConstructor(DataType t, std::ostream& os);
  // Get a cast type from to
  virtual std::string CastFromTo(std::string value, DataType from, DataType target);
  // Get load of single element with expression
  virtual void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os);
  // Print restrict keyword for a given Var if applicable
  virtual void PrintRestrict(const Var& v, std::ostream& os);

  virtual void SetConstantsByteAlignment(Integer constants_byte_alignment) {
    constants_byte_alignment_ = constants_byte_alignment;
  }

 protected:
  // Print reference to struct location
  std::string GetStructRef(DataType t, const PrimExpr& buffer, const PrimExpr& index, int kind);
  // Print reference to a buffer as type t in index.
  virtual std::string GetBufferRef(DataType t, const BufferNode* buffer, PrimExpr index);

  /*!
   * \brief Handle volatile loads.
   *
   * This is to workaround a bug in CUDA cuda_fp16.h. Volatile accesses
   * to shared memory are required for reductions. However, __half class
   * does not implement volatile member functions. CUDA codegen will cast
   * away volatile qualifier from CUDA __half types.
   */
  virtual void HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                                   std::ostream& os) {
    // By default, do nothing but print the loaded value.
    os << value;
  }

  /*!
   * \brief Check if scope is part of type in the target language.
   *
   * **NOTE** In OpenCL, __local is part of type, so "__local int *"
   * is legal. This is not the case for CUDA, where "__shared__"
   * or "__constant__" is not part of type but a storage class (like
   * C/C++ static).
   */
  virtual bool IsScopePartOfType() const { return true; }

  /*!
   * \brief Generate forward function declarations.
   * \param global_symbol The symbolc of the target function.
   * \param arg_types The argument types to the function.
   * \param ret_type The return type of the function
   * \param os The output stream.
   */
  virtual void GenerateForwardFunctionDeclarations(String global_symbol,
                                                   const Array<Type>& arg_types,
                                                   const Type& ret_type) {}

  /*!
   * \brief Print external function call.
   * \param ret_type The return type.
   * \param global_symbol The symbolc of the target function.
   * \param args The arguments to the function.
   * \param skip_first_arg Whether to skip the first arguments.
   * \param os The output stream.
   */
  virtual void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                               bool skip_first_arg, std::ostream& os);  // NOLINT(*)
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
  void PrintSSAAssign(const std::string& target, const std::string& src, DataType t) override;
  /*! \brief reserves common C keywords */
  void ReserveKeywordsAsUnique();

  /*! \brief Check if buf_var is volatile or not. */
  bool IsVolatile(const VarNode* buf_var) const { return volatile_buf_.count(buf_var) != 0; }

  /*! \brief restrict keyword */
  std::string restrict_keyword_{""};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const VarNode*, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const VarNode*, DataType> handle_data_type_;
  /*! \brief Record of ops that have pre-defined global symbol. */
  OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
  // cache commonly used ops
  const Op& builtin_call_extern_ = builtin::call_extern();
  const Op& builtin_call_pure_extern_ = builtin::call_pure_extern();
  Integer constants_byte_alignment_ = 16;
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{false};

 private:
  /*! \brief set of volatile buf access */
  std::unordered_set<const VarNode*> volatile_buf_;

  // deep comparison of PrimExpr
  ExprDeepEqual deep_equal_;

  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;

  /* \brief Map of GlobalVar to their symbol.
   *
   * For externally-exposed functions, this is given by the
   * tvm::attr::kTarget attribute of the PrimFunc.  For internal
   * functions, this is the name of the function's GlobalVar, possibly
   * altered to prevent duplicate names.
   */
  std::unordered_map<GlobalVar, String, ObjectPtrHash, ObjectPtrEqual> internal_functions_;

  /* \brief Name supply to generate unique function names */
  NameSupply func_name_supply_{""};
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SOURCE_CODEGEN_C_H_
