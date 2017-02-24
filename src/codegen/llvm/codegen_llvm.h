/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_llvm.h
 * \brief Common base class for generating into LLVM IR
 */
#ifndef TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
#define TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
#ifdef TVM_LLVM_VERSION

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/codegen.h>
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
class CodeGenLLVM : public IRVisitor {
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
    value_ = nullptr;
    this->Visit(e);
    CHECK(value_ != nullptr);
    return value_;
  }
  // Short hande code to get a constant int 32
  llvm::Constant* ConstInt32(unsigned value) const {
    return llvm::ConstantInt::get(t_int32_, value);
  }
  // override codegen
  void Visit_(const Variable* op) final;
  void Visit_(const Cast* op) final;
  void Visit_(const IntImm* op) final;
  void Visit_(const UIntImm* op) final;
  void Visit_(const FloatImm* op) final;
  void Visit_(const StringImm* op) final;
  void Visit_(const Add* op) final;
  void Visit_(const Sub* op) final;
  void Visit_(const Mul* op) final;
  void Visit_(const Div* op) final;
  void Visit_(const Mod* op) final;
  void Visit_(const Min* op) final;
  void Visit_(const Max* op) final;
  void Visit_(const LT* op) final;
  void Visit_(const LE* op) final;
  void Visit_(const GT* op) final;
  void Visit_(const GE* op) final;
  void Visit_(const EQ* op) final;
  void Visit_(const NE* op) final;
  void Visit_(const And* op) final;
  void Visit_(const Or* op) final;
  void Visit_(const Not* op) final;
  void Visit_(const Select* op) final;
  void Visit_(const Let* op) final;
  void Visit_(const Load* op) final;
  void Visit_(const Call* op) final;
  void Visit_(const Ramp* op) final;
  void Visit_(const Broadcast* op) final;
  // stmt
  void Visit_(const Store* op) final;
  void Visit_(const For* op) final;
  void Visit_(const IfThenElse* op) final;
  void Visit_(const Allocate* op) final;
  void Visit_(const AttrStmt* op) override;
  void Visit_(const AssertStmt* op) final;
  void Visit_(const LetStmt* op) final;
  // create intrinstic given call
  virtual llvm::Value* CreateIntrinstic(const Call* op);
  // create extern function call
  virtual llvm::Value* CreateCallExtern(const Call* op);
  // create call into tvm packed function.
  virtual llvm::Value* CreateCallPacked(const Call* op);

 protected:
  /*!
   * \param t The original type.
   * \return LLVM type of t
   */
  llvm::Type* LLVMType(const Type& t) const;
  // do a scalarize call with f
  llvm::Value* CreateScalarizedCall(
      const Call* op, llvm::Function* f, const std::vector<llvm::Value*>& args);
  // apply optimization on the module.
  virtual void Optimize();
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
  llvm::Type* t_tvm_index_{nullptr};
  llvm::Type* t_tvm_func_handle_{nullptr};
  llvm::StructType* t_tvm_context_{nullptr};
  llvm::StructType* t_tvm_type_{nullptr};
  llvm::StructType* t_tvm_array_{nullptr};
  llvm::StructType* t_tvm_value_{nullptr};
  // tvm api functions
  llvm::Function* f_tvm_func_call_{nullptr};
  llvm::Function* f_tvm_get_func_from_env_{nullptr};
  llvm::Function* f_tvm_api_set_last_error_{nullptr};
  // The acting body
  llvm::BasicBlock* block_{nullptr};
  // Last value returned codegen call.
  llvm::Value* value_{nullptr};

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
  llvm::Value* CreateCast(Type from, Type to, llvm::Value* value);
  llvm::Value* GetPackedFuncHandle(const std::string& str);
  // Check if the call to packed function is successful
  // if not directly finalize function and pass on return code.
  // return the end block after the check
  llvm::BasicBlock* CheckPackedCallSuccess(llvm::Value* retcode);
  // Initialize target
  void InitTarget(const std::string& target);
  // Add a function to set global module context
  void InitGlobalContext();
  // add alias information.
  void AddAliasInfo(llvm::Instruction* load, const Variable* buffer, Expr index);
  // The definition of local variable.
  std::unordered_map<const Variable*, llvm::Value*> var_map_;
  // global strings
  std::unordered_map<std::string, llvm::Constant*> str_map_;
  // The local module_context
  llvm::GlobalVariable* gv_mod_ctx_{nullptr};
  // global to packed function handle
  std::unordered_map<std::string, llvm::GlobalVariable*> func_handle_map_;
};
}  // namespace codegen
}  // namespace tvm
#endif  // LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_CODEGEN_LLVM_H_
