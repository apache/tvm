/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_llvm.h
 * \brief Common base class for generating into LLVM IR
 */
#ifndef TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
#define TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
#ifdef TVM_LLVM_VERSION

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/codegen.h>
#include <tvm/arithmetic.h>
#include <memory>
#include <vector>
#include <string>
#include "./llvm_common.h"

namespace tvm {
namespace codegen {

using namespace ir;

/*!
 * \brief A base class to generate a LLVM.
 */
class CodeGenLLVM :
      public ExprFunctor<llvm::Value* (const Expr&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Initialize the code generator with given context
   * \param module_name The name of the module.
   * \param target_triple The target triple, can be empty.
   * \param ctx The context.
   */
  void Init(const std::string& module_name,
            const std::string& target_triple,
            llvm::LLVMContext* ctx);
  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   */
  void AddFunction(const LoweredFunc& f);
  /*!
   * \brief Add main function as the entry name
   * \param entry_func_name The name of entry function to be added.
   */
  void AddMainFunction(const std::string& entry_func_name);
  /*!
   * \brief Finish current pass of codegen, get the module.
   * \return the created module.
   */
  std::unique_ptr<llvm::Module> Finish();
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  llvm::Value* MakeValue(const Expr& e) {
    return VisitExpr(e);
  }
  // Short hande code to get a constant int 32
  llvm::Constant* ConstInt32(unsigned value) const {
    return llvm::ConstantInt::get(t_int32_, value);
  }
  // override codegen
  llvm::Value* VisitExpr_(const Variable* op) override;
  llvm::Value* VisitExpr_(const Cast* op) override;
  llvm::Value* VisitExpr_(const IntImm* op) override;
  llvm::Value* VisitExpr_(const UIntImm* op) override;
  llvm::Value* VisitExpr_(const FloatImm* op) override;
  llvm::Value* VisitExpr_(const StringImm* op) override;
  llvm::Value* VisitExpr_(const Add* op) override;
  llvm::Value* VisitExpr_(const Sub* op) override;
  llvm::Value* VisitExpr_(const Mul* op) override;
  llvm::Value* VisitExpr_(const Div* op) override;
  llvm::Value* VisitExpr_(const Mod* op) override;
  llvm::Value* VisitExpr_(const Min* op) override;
  llvm::Value* VisitExpr_(const Max* op) override;
  llvm::Value* VisitExpr_(const LT* op) override;
  llvm::Value* VisitExpr_(const LE* op) override;
  llvm::Value* VisitExpr_(const GT* op) override;
  llvm::Value* VisitExpr_(const GE* op) override;
  llvm::Value* VisitExpr_(const EQ* op) override;
  llvm::Value* VisitExpr_(const NE* op) override;
  llvm::Value* VisitExpr_(const And* op) override;
  llvm::Value* VisitExpr_(const Or* op) override;
  llvm::Value* VisitExpr_(const Not* op) override;
  llvm::Value* VisitExpr_(const Select* op) override;
  llvm::Value* VisitExpr_(const Let* op) override;
  llvm::Value* VisitExpr_(const Load* op) override;
  llvm::Value* VisitExpr_(const Call* op) override;
  llvm::Value* VisitExpr_(const Ramp* op) override;
  llvm::Value* VisitExpr_(const Broadcast* op) override;
  // stmt
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
  // create intrinstic given call
  virtual llvm::Value* CreateIntrinstic(const Call* op);
  // create extern function call
  virtual llvm::Value* CreateCallExtern(const Call* op);
  // create call into tvm packed function.
  virtual llvm::Value* CreateCallPacked(const Call* op);
  // Scalarize e by iterating elements of e.
  // f is a callback that takes index and v.
  virtual void Scalarize(const Expr& e,
                         std::function<void(int i, llvm::Value* v)> f);
 protected:
  /*!
   * \param t The original type.
   * \return LLVM type of t
   */
  llvm::Type* LLVMType(const Type& t) const;
  // initialize the function state.
  void InitFuncState();
  // Get alignment given index.
  void GetAlignment(
      Type t, const Variable* buf_var, const Expr& index,
      int* p_alignment, int* p_native_bits);
  // do a scalarize call with f
  llvm::Value* CreateScalarizedCall(
      const Call* op, llvm::Function* f, const std::vector<llvm::Value*>& args);
  // apply optimization on the module.
  virtual void Optimize();
  // Get the maximim storage align bits of buffer pointer given storage scope.
  virtual int NativeVectorBits(const std::string& storage_scope) const;
  // The IRBuilder.
  using IRBuilder = llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;
  // The current function
  llvm::Function* function_;
  // Internal builder
  std::unique_ptr<IRBuilder> builder_;
  // The module to be returned;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::DataLayout> data_layout_;
  // Internal metabuilder
  std::unique_ptr<llvm::MDBuilder> md_builder_;
  // llvm context
  llvm::LLVMContext* ctx_{nullptr};
  // helpful data types
  llvm::Type* t_void_{nullptr};
  llvm::Type* t_void_p_{nullptr};
  llvm::Type* t_int_{nullptr};
  llvm::Type* t_char_{nullptr};
  llvm::Type* t_int8_{nullptr};
  llvm::Type* t_int16_{nullptr};
  llvm::Type* t_int32_{nullptr};
  llvm::Type* t_int64_{nullptr};
  llvm::Type* t_float64_{nullptr};
  // branch
  llvm::MDNode* md_very_likely_branch_{nullptr};
  llvm::MDNode* md_tbaa_root_{nullptr};
  // TVM related data types
  llvm::Type* t_tvm_shape_index_{nullptr};
  llvm::Type* t_tvm_func_handle_{nullptr};
  llvm::StructType* t_tvm_context_{nullptr};
  llvm::StructType* t_tvm_type_{nullptr};
  llvm::StructType* t_tvm_array_{nullptr};
  llvm::StructType* t_tvm_value_{nullptr};
  llvm::FunctionType* t_f_tvm_par_for_lambda_{nullptr};
  // tvm api functions
  llvm::Function* f_tvm_func_call_{nullptr};
  llvm::Function* f_tvm_get_func_from_env_{nullptr};
  llvm::Function* f_tvm_api_set_last_error_{nullptr};
  llvm::Function* f_tvm_parallel_for_{nullptr};
  // The acting body
  llvm::BasicBlock* block_{nullptr};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;

 private:
  // comparison op
  llvm::Value* GetVarValue(const Variable* v) const;
  llvm::Value* CreateLT(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateLE(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateGT(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateGE(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateAdd(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateSub(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateMul(Type t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateBroadcast(llvm::Value* value, int lanes);
  llvm::Value* GetConstString(const std::string& str);
  llvm::Value* CreateBufferPtr(Type t, llvm::Value* buffer, llvm::Value* index);
  llvm::Value* CreateStructRefPtr(Type t, llvm::Value* buffer, llvm::Value* index, int kind);
  llvm::Value* CreateCast(Type from, Type to, llvm::Value* value);
  llvm::Value* GetPackedFuncHandle(const std::string& str);
  // Vector concatenation.
  llvm::Value* CreateVecSlice(llvm::Value* vec, int begin, int extent);
  llvm::Value* CreateVecFlip(llvm::Value* vec);
  llvm::Value* CreateVecConcat(std::vector<llvm::Value*> vecs);
  llvm::Value* CreateVecPad(llvm::Value* vec, int target_lanes);
  // Create parallel for.
  void CreateParallelFor(const For* op);
  // Create serial for
  void CreateSerialFor(llvm::Value* begin, llvm::Value* end,
                       const VarExpr& loop_var, const Stmt& body);
  // Check if the call to packed function is successful
  // if not directly finalize function and pass on return code.
  // return the end block after the check
  llvm::BasicBlock* CheckCallSuccess(llvm::Value* retcode);
  // Initialize target
  void InitTarget(const std::string& target);
  // Add a function to set global module context
  void InitGlobalContext();
  // add alias information.
  void AddAliasInfo(llvm::Instruction* load, const Variable* buffer, Expr index, Type type);
  // The definition of local variable.
  std::unordered_map<const Variable*, llvm::Value*> var_map_;
  // global strings
  std::unordered_map<std::string, llvm::Constant*> str_map_;
  // The alignment information
  std::unordered_map<const Variable*, arith::ModularEntry> align_map_;
  // The local module_context
  llvm::GlobalVariable* gv_mod_ctx_{nullptr};
  // global to packed function handle
  std::unordered_map<std::string, llvm::GlobalVariable*> func_handle_map_;
};
}  // namespace codegen
}  // namespace tvm
#endif  // LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
