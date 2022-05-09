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
 * \file codegen_llvm.h
 * \brief Common base class for generating into LLVM IR
 */
#ifndef TVM_TARGET_LLVM_CODEGEN_LLVM_H_
#define TVM_TARGET_LLVM_CODEGEN_LLVM_H_
#include <llvm/IR/GlobalValue.h>
#ifdef TVM_LLVM_VERSION

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../runtime/thread_storage_scope.h"
#include "../../tir/transforms/ir_utils.h"
#include "llvm_common.h"

namespace tvm {
namespace codegen {

using namespace tir;

/*!
 * \brief A base class to generate a LLVM.
 */
class CodeGenLLVM : public ExprFunctor<llvm::Value*(const PrimExpr&)>,
                    public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Create new code generator based on target machine.
   * \param tm The target machine
   * \return The created llvm generator.
   */
  static std::unique_ptr<CodeGenLLVM> Create(llvm::TargetMachine* tm);
  /*!
   * \brief Initialize the code generator with given context
   * \param module_name The name of the module.
   * \param tm Target machine model
   * \param ctx The context.
   * \param system_lib Whether to insert system library registration.
   * \param dynamic_lookup Whether dynamically lookup runtime function
   *                       or use the runtime function table passed by caller.
   * \param target_c_runtime If true, generate a module to be executed by the C runtime. In practice
   *                       this option influences whether global ctors are used.
   */
  virtual void Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
                    bool system_lib, bool dynamic_lookup, bool target_c_runtime);

  /*!
   * \brief Turn on fast math flags for floating point operations.
   * \param fmf FastMathFlags to use for code generation.
   */
  void SetFastMathFlag(llvm::FastMathFlags fmf);

  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   */
  virtual void AddFunction(const PrimFunc& f);
  /*!
   * \brief Add main function as the entry name
   * \param entry_func_name The name of entry function to be added.
   */
  virtual void AddMainFunction(const std::string& entry_func_name);
  /*!
   * \brief Finish current pass of codegen, get the module.
   * \return the created module.
   */
  virtual std::unique_ptr<llvm::Module> Finish();
  /*!
   * \brief Add functions from the (unordered) range to the current module in a deterministic order.
   *        The range consists of objects convertible to PrimFunc.
   * \param begin The beginning of the range.
   * \param end The end of the range.
   * \param pfunc Converter function from the range element type to PrimFunc.
   */
  template <typename IterType, typename ConvType>
  void AddFunctionsOrdered(IterType begin, IterType end, ConvType pfunc);
  /*!
   * \brief Add functions from the (unordered) range of elements of type PrimFunc to the current
   *        module in a deterministic order.
   * \param begin The beginning of the range.
   * \param end The end of the range.
   */
  template <typename IterType>
  void AddFunctionsOrdered(IterType begin, IterType end) {
    this->AddFunctionsOrdered(begin, end, [](auto f) { return f; });
  }
  /*!
   * \brief Add mod to be linked with the generated module
   * \param mod The module to be linked.
   */
  void AddLinkModule(std::unique_ptr<llvm::Module>&& mod);
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  llvm::Value* MakeValue(const PrimExpr& e) { return VisitExpr(e); }
  // Short hande code to get a constant int 32
  llvm::Constant* ConstInt32(int64_t value) const {
    return llvm::ConstantInt::getSigned(t_int32_, value);
  }
  // override codegen
  llvm::Value* VisitExpr_(const VarNode* op) override;
  llvm::Value* VisitExpr_(const CastNode* op) override;
  llvm::Value* VisitExpr_(const IntImmNode* op) override;
  llvm::Value* VisitExpr_(const FloatImmNode* op) override;
  llvm::Value* VisitExpr_(const StringImmNode* op) override;
  llvm::Value* VisitExpr_(const AddNode* op) override;
  llvm::Value* VisitExpr_(const SubNode* op) override;
  llvm::Value* VisitExpr_(const MulNode* op) override;
  llvm::Value* VisitExpr_(const DivNode* op) override;
  llvm::Value* VisitExpr_(const ModNode* op) override;
  llvm::Value* VisitExpr_(const MinNode* op) override;
  llvm::Value* VisitExpr_(const MaxNode* op) override;
  llvm::Value* VisitExpr_(const LTNode* op) override;
  llvm::Value* VisitExpr_(const LENode* op) override;
  llvm::Value* VisitExpr_(const GTNode* op) override;
  llvm::Value* VisitExpr_(const GENode* op) override;
  llvm::Value* VisitExpr_(const EQNode* op) override;
  llvm::Value* VisitExpr_(const NENode* op) override;
  llvm::Value* VisitExpr_(const AndNode* op) override;
  llvm::Value* VisitExpr_(const OrNode* op) override;
  llvm::Value* VisitExpr_(const NotNode* op) override;
  llvm::Value* VisitExpr_(const SelectNode* op) override;
  llvm::Value* VisitExpr_(const LetNode* op) override;
  llvm::Value* VisitExpr_(const LoadNode* op) override;
  llvm::Value* VisitExpr_(const BufferLoadNode* op) override;
  llvm::Value* VisitExpr_(const CallNode* op) override;
  llvm::Value* VisitExpr_(const RampNode* op) override;
  llvm::Value* VisitExpr_(const ShuffleNode* op) override;
  llvm::Value* VisitExpr_(const BroadcastNode* op) override;
  // stmt
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const WhileNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AllocateConstNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;

  // Get constant string
  llvm::Constant* GetConstString(const std::string& str);

  llvm::Constant* GetGlobalConstant(
      llvm::Constant* const_data, const std::string& name = "",
      llvm::GlobalValue::LinkageTypes linkage_type = llvm::GlobalValue::InternalLinkage);

 protected:
  /*!
   * \brief Address and type pair to assist in handling opaque pointers.
   */
  struct TypedPointer {
    TypedPointer() = default;
    TypedPointer(llvm::Type* t, llvm::Value* a) : type(t), addr(a) {}
    llvm::Type* type = nullptr;  /*!< Type of the value pointed to. */
    llvm::Value* addr = nullptr; /*!< Address of the value.         */
  };
  /*! \brief The storage information */
  struct StorageInfo {
    /*! \brief The alignment of allocation */
    int alignment{0};
  };
  /*!
   * \brief Execute falloca at the beginning of the
   *  currrent function and obtain its return value.
   *
   *  This is a helper function to make sure that
   *  alloca always happen in the beginning of the function.
   *
   * \param falloca The allocation function to be executed.
   * \tparam F The function to be executed.
   * \return The result.
   */
  template <typename F>
  llvm::AllocaInst* WithFunctionEntry(F falloca) {
    llvm::BasicBlock* current = builder_->GetInsertBlock();
    llvm::BasicBlock* entry = &(function_->getEntryBlock());
    builder_->SetInsertPoint(entry, entry->begin());
    llvm::AllocaInst* res = falloca();
    builder_->SetInsertPoint(current);
    return res;
  }
  // create intrinstic given call
  virtual llvm::Value* CreateIntrinsic(const CallNode* op);
  // create extern function call
  // skip first arg mode used for call extern intrinsic.
  virtual llvm::Value* CreateCallExtern(Type ret_type, String global_symbol,
                                        const Array<PrimExpr>& args, bool skip_first_arg);

  /*! \brief Insert a printf() call to the generated LLVM
   *
   * This is intended solely for debugging purposes.  After calling
   * printf(), immediately calls fflush() to flush the stdout buffer
   * in case of segfault.
   */
  virtual void CreatePrintf(const std::string& format, llvm::ArrayRef<llvm::Value*> format_args);

  /*! \brief Lookup return address, for debugging purposes
   *
   * This is intended solely for debugging purposes.  Calls the
   * `llvm::Intrinsic::returnaddress`, returning the return address of
   * the current function call.
   *
   * \param level Look up the return address of a frame `level` steps
   * above the current stack frame.
   */
  llvm::Value* CreateLookupReturnAddress(unsigned int level = 0);

  // Get the corresponding thread index
  virtual llvm::Value* GetThreadIndex(const IterVar& iv);
  // Get the corresponding thread index
  virtual llvm::Value* CreateStorageSync(const CallNode* op);
  // apply optimization on the module.
  virtual void InitPassManagerBuilder(llvm::PassManagerBuilder* builder);
  // Scalarize by iterating elements of e.
  // f is a callback that takes index and v.
  void Scalarize(const PrimExpr& e, std::function<void(int i, llvm::Value* v)> f);

  /* \brief Helper function for handling buffer access
   *
   * \param buffer The buffer being accessed
   *
   * \param indices The indices at which the buffer is being accessed.
   *
   * \param value_dtype The datatype to be read from (BufferLoad) or
   * written to (BufferStore) the buffer.
   *
   * \param make_instruction A callback function that generates that
   * actual call.
   *
   *       - buffer_ptr: A typed pointer to the element being accessed
   *
   *       - subelement_i: The index of a vectorized type to be
   *         stored/loaded.  If -1, indicates that the entire type,
   *         vector or scalar, should be written.
   *
   *       - alignment: The alignment to be used for the read/write.
   *
   *       - is_volatile: Whether the read/write should be volatile.
   *
   *       - Should return the generated expression.
   */
  void BufferAccessHelper(
      Buffer buffer, Array<PrimExpr> indices, DataType value_dtype,
      std::function<llvm::Instruction*(TypedPointer buffer_ptr, int subelement_i, int alignment,
                                       bool is_volatile)>
          make_instruction);
  // Initialize target
  virtual void InitTarget(llvm::TargetMachine* tm);
  // Add module startup function if needed.
  virtual void AddStartupFunction() {}
  // apply optimization on the module.
  virtual void Optimize();
  // Get the maximim storage align bits of buffer pointer given storage scope.
  virtual int NativeVectorBits(const runtime::StorageScope& storage_scope) const;
  // Get correct address space depending on the backend
  virtual unsigned GetGlobalAddressSpace() const;
  void AddFunctionInternal(const PrimFunc& f, bool ret_void);
  // Create extern call
  llvm::CallInst* CreateCallExtern(llvm::Type* ret, const std::string& name,
                                   const std::vector<llvm::Value*>& value);
  /*!
   * \brief Get the LLVM Type for a given runtime type.
   * \param dtype The runtime dtype.
   *
   * \note Only use this function for dealing with PrimTypes.
   *       For Call and Var that could have more refined types,
   *       use GetLLVMType instead.
   *
   * \return LLVM type of dtype
   */
  llvm::Type* DTypeToLLVMType(const DataType& dtype) const;
  /*!
   * \brief Get the LLVM Type for a given type.
   * \param dtype The runtime dtype.
   * \param type The corresponding TVM Type.
   */
  llvm::Type* GetLLVMType(const Type& type) const;
  /*!
   * \brief Get the LLVM Type for a given type.
   * \param dtype The runtime dtype.
   * \param type The corresponding TVM Type.
   */
  llvm::Type* GetLLVMType(const PrimExpr& expr) const;
  /*!
   * \brief Get the declaration of the LLVM intrinsic based on the intrinsic
   *        id, and the type of the actual call.
   *
   * \param id The intrinsic id.
   * \param ret_type The call return type.
   * \param arg_types The types of the call arguments.
   *
   * \return Return the llvm::Function pointer, or nullptr if the declaration
   *         could not be generated (e.g. if the argument/return types do not
   *         match).
   */
  llvm::Function* GetIntrinsicDecl(llvm::Intrinsic::ID id, llvm::Type* ret_type,
                                   llvm::ArrayRef<llvm::Type*> arg_types);
  /*!
   * \brief Set target-related attributes on the LLVM function \p func. This
   *        includes "target-cpu" and "target-features" if present.
   *
   * \param func The function to set attributes on.
   */
  void SetTargetAttributes(llvm::Function* func);
  /*!
   * \brief Get the number of elements in the given vector value.
   * \param vec The value, must be of a vector type.
   */
  inline int GetVectorNumElements(llvm::Value* vec);
  // initialize the function state.
  void InitFuncState();
  // Get alignment given index.
  void GetAlignment(DataType t, const VarNode* buf_var, const PrimExpr& index, int* p_alignment,
                    int* p_native_bits);
  // Returns whether the LLVM type has padding for alignment
  bool HasAlignmentPadding(DataType dtype);
  // do a scalarize call with f
  llvm::Value* CreateScalarizedCall(const CallNode* op, llvm::Function* f,
                                    const std::vector<llvm::Value*>& args);
  // handle module import
  void HandleImport(const std::string& code);
  // cast operatpr
  llvm::Value* CreateCast(DataType from, DataType to, llvm::Value* value);
  // comparison op
  llvm::Value* GetVarValue(const VarNode* v) const;
  llvm::Value* CreateLT(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateLE(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateGT(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateGE(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateAdd(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateSub(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateMul(DataType t, llvm::Value* a, llvm::Value* b);
  llvm::Value* CreateBroadcast(llvm::Value* value, int lanes);
  virtual TypedPointer CreateBufferPtr(llvm::Value* buffer_ptr, DataType buffer_element_dtype,
                                       llvm::ArrayRef<llvm::Value*> indices, DataType value_dtype);
  // Vector concatenation.
  llvm::Value* CreateVecSlice(llvm::Value* vec, int begin, int extent);
  llvm::Value* CreateVecFlip(llvm::Value* vec);
  llvm::Value* CreateVecConcat(std::vector<llvm::Value*> vecs);
  llvm::Value* CreateVecPad(llvm::Value* vec, int target_lanes);
  // Create serial for
  void CreateSerialFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride,
                       const Var& loop_var, const Stmt& body);
  // add alias information.
  void AddAliasInfo(llvm::Instruction* inst, const VarNode* buffer_var, PrimExpr index,
                    DataType access_dtype);

  llvm::GlobalVariable* AllocateSharedMemory(DataType dtype, size_t size,
                                             unsigned int shared_address_space, int alignment,
                                             llvm::GlobalValue::LinkageTypes linkage);

  /*!
   * \brief Get the `i`th argument to the given function, respecting LLVM API changes.
   *
   * NOTE: in LLVM < 10.0, the underlying API returns a const llvm::Argument*. To provide a uniform
   * API, const is removed here. Proper usage of LLVM APIs depends on having a non-const Argument*,
   * so we take this appraoch here rather than adding const.
   *
   * \param function The function containing the arguments.
   * \param i The index of the argument to retrieve.
   * \return The retrieved argument.
   */
  llvm::Argument* GetArg(const llvm::Function* function, int i) const {
#if TVM_LLVM_VERSION >= 100
    return function->getArg(i);
#elif TVM_LLVM_VERSION >= 50
    return const_cast<llvm::Argument*>(&function->arg_begin()[i]);
#else
    return const_cast<llvm::Argument*>(&*std::next(function->arg_begin(), i));
#endif
  }

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
  // llvm target machine
  llvm::TargetMachine* target_machine_{nullptr};
  // llvm context
  llvm::LLVMContext* ctx_{nullptr};
  // helpful data types
  llvm::Type* t_void_{nullptr};
  llvm::PointerType* t_void_p_{nullptr};
  llvm::Type* t_int_{nullptr};
  llvm::Type* t_char_{nullptr};
  llvm::Type* t_int8_{nullptr};
  llvm::Type* t_int16_{nullptr};
  llvm::Type* t_int32_{nullptr};
  llvm::Type* t_int64_{nullptr};
  llvm::Type* t_float64_{nullptr};
  // meta data
  llvm::MDNode* md_very_likely_branch_{nullptr};
  llvm::MDNode* md_tbaa_root_{nullptr};
  llvm::MDNode* md_tbaa_alias_set_{nullptr};
  // modules to be linked.
  std::vector<std::unique_ptr<llvm::Module> > link_modules_;
  /*! \brief native vector bits of current targetx*/
  int native_vector_bits_{0};
  /*! \brief the storage scope of allocation */
  std::unordered_map<const VarNode*, StorageInfo> alloc_storage_info_;
  // The definition of local variable.
  std::unordered_map<const VarNode*, llvm::Value*> var_map_;
  // global strings
  std::unordered_map<std::string, llvm::Constant*> str_map_;
  // Whether current function is restricted
  bool is_restricted_{true};
  // The analyzer information
  std::unique_ptr<arith::Analyzer> analyzer_;
  // set of var that are not restricted(can alias)
  std::unordered_set<const VarNode*> alias_var_set_;
  // set of volatile buffer.
  std::unordered_set<const VarNode*> volatile_buf_;
  // deep comparison of PrimExpr
  ExprDeepEqual deep_equal_;
  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  // Cache potential common path ops to slightly improve lookup time.
  // global symbol table.
  OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
  const Op& builtin_call_extern_ = builtin::call_extern();
  const Op& builtin_call_pure_extern_ = builtin::call_pure_extern();
  const Op& builtin_call_llvm_intrin_ = builtin::call_llvm_intrin();
  const Op& builtin_call_llvm_pure_intrin_ = builtin::call_llvm_pure_intrin();
  const Op& builtin_lookup_param_ = builtin::lookup_param();
  const Op& builtin_tvm_call_cpacked_lowered_ = builtin::tvm_call_cpacked_lowered();

  /*! \brief Helper struct for debug infos. */
  struct DebugInfo {
    std::unique_ptr<llvm::DIBuilder> di_builder_;
    llvm::DICompileUnit* compilation_unit_{nullptr};
    llvm::DIFile* file_{nullptr};
  };
  /*!
   * \brief Create a new DebugInfo struct from the given Module that
   *  initializes file and compilation_unit_ to TVM defaults.
   */
  static std::unique_ptr<DebugInfo> CreateDebugInfo(llvm::Module* module);
};

inline int CodeGenLLVM::GetVectorNumElements(llvm::Value* vec) {
#if TVM_LLVM_VERSION >= 120
  return llvm::cast<llvm::FixedVectorType>(vec->getType())->getNumElements();
#else
  return llvm::cast<llvm::VectorType>(vec->getType())->getNumElements();
#endif
}

template <typename IterType, typename ConvType>
void CodeGenLLVM::AddFunctionsOrdered(IterType begin, IterType end, ConvType pfunc) {
  std::vector<PrimFunc> funcs;
  for (auto it = begin; it != end; ++it) {
    funcs.push_back(pfunc(*it));
  }
  std::sort(funcs.begin(), funcs.end(), [](PrimFunc func_a, PrimFunc func_b) {
    std::string name_a = func_a->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
    std::string name_b = func_b->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
    return name_a < name_b;
  });
  for (auto& f : funcs) {
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    AddFunction(f);
  }
}

}  // namespace codegen
}  // namespace tvm
#endif  // LLVM_VERSION
#endif  // TVM_TARGET_LLVM_CODEGEN_LLVM_H_
