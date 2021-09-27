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
 * \file codegen_llvm_cpu.h
 * \brief Common base class for generating into LLVM IR on CPU host.
 */
#ifndef TVM_TARGET_LLVM_CODEGEN_CPU_H_
#define TVM_TARGET_LLVM_CODEGEN_CPU_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "codegen_llvm.h"

namespace tvm {
namespace codegen {

// CPU host code generation
class CodeGenCPU : public CodeGenLLVM {
 public:
  void Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
            bool system_lib, bool dynamic_lookup, bool target_c_runtime) override;
  void AddFunction(const PrimFunc& f) override;
  void AddMainFunction(const std::string& entry_func_name) override;
  std::unique_ptr<llvm::Module> Finish() override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  llvm::Value* CreateIntrinsic(const CallNode* op) override;
  llvm::Value* CreateCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                bool skip_first_arg) override;

  /*!
   * \brief A CPU-specific function to create the FuncRegistry.
   * \param func_names List of functions to be included, in order.
   */
  void DefineFunctionRegistry(Array<String> func_names);

 protected:
  void AddStartupFunction() final;
  // meta data
  llvm::MDNode* md_tbaa_ctx_ptr_{nullptr};
  // TVM related data types
  llvm::Type* t_tvm_shape_index_{nullptr};
  llvm::Type* t_tvm_func_handle_{nullptr};
  llvm::StructType* t_tvm_context_{nullptr};
  llvm::StructType* t_tvm_type_{nullptr};
  llvm::StructType* t_tvm_array_{nullptr};
  llvm::StructType* t_tvm_value_{nullptr};
  llvm::StructType* t_tvm_parallel_group_env_{nullptr};

  llvm::FunctionType* ftype_tvm_backend_packed_c_func_{nullptr};
  llvm::StructType* t_tvm_crt_func_registry_{nullptr};
  llvm::StructType* t_tvm_crt_module_{nullptr};

  llvm::FunctionType* ftype_tvm_parallel_lambda_{nullptr};
  llvm::FunctionType* ftype_tvm_func_call_{nullptr};
  llvm::FunctionType* ftype_tvm_get_func_from_env_{nullptr};
  llvm::FunctionType* ftype_tvm_api_set_last_error_{nullptr};
  llvm::FunctionType* ftype_tvm_parallel_launch_{nullptr};
  llvm::FunctionType* ftype_tvm_parallel_barrier_{nullptr};
  llvm::FunctionType* ftype_tvm_register_system_symbol_{nullptr};
  // Lazy entry for function call.
  llvm::FunctionType* ftype_tvm_static_init_callback_{nullptr};
  llvm::FunctionType* ftype_tvm_static_init_{nullptr};

 private:
  // the parallel group information
  struct ParallelEnv {
    Var task_id;
    Var num_task;
    bool stride_pattern{false};
    bool in_parallel_loop{false};
    int parallel_loop_count{0};
    llvm::Value* penv{nullptr};
  };
  // Get runtime functions
  void InitGlobalContext(bool dynamic_lookup);
  llvm::GlobalVariable* InitContextPtr(llvm::Type* type, std::string name);
  llvm::Value* GetContextPtr(llvm::GlobalVariable* gv);
  llvm::Value* RuntimeTVMFuncCall();
  llvm::Value* RuntimeTVMGetFuncFromEnv();
  llvm::Value* RuntimeTVMAPISetLastError();
  llvm::Value* RuntimeTVMParallelLaunch();
  llvm::Value* RuntimeTVMParallelBarrier();
  llvm::Value* CreateStaticHandle();
  llvm::Value* GetPackedFuncHandle(const std::string& str);
  llvm::Value* PackClosureData(const Array<Var>& fields, uint64_t* num_bytes);
  llvm::Value* CreateStructRefPtr(DataType t, llvm::Value* buffer, llvm::Value* index, int kind);
  void UnpackClosureData(llvm::Value* cdata, const Array<Var>& fields,
                         std::unordered_map<const VarNode*, llvm::Value*>* vmap);
  // Make packed call.
  struct PackedCall {
    llvm::Value* ret_value;
    llvm::Value* ret_tcode;
    llvm::BasicBlock* end_block;
  };
  PackedCall MakeCallPackedLowered(const Array<PrimExpr>& args, const DataType& r_type,
                                   const int64_t begin, const int64_t end);
  // create call into tvm packed function.
  llvm::Value* CreateCallPacked(const CallNode* op);
  // Create trace call into tvm packed function.
  llvm::Value* CreateCallTracePacked(const CallNode* op);
  // Create static initialization
  void CreateStaticInit(const std::string& init_fname, const Stmt& body);
  // Create parallel launch
  void CreateParallelLaunch(const Stmt& body, int num_task);
  // Create a new compute scope.
  void CreateComputeScope(const AttrStmtNode* op);
  // Check if the call to packed function is successful
  // if not directly finalize function and pass on return code.
  // return the end block after the check
  llvm::BasicBlock* CheckCallSuccess(llvm::Value* retcode);
  // Context for injection lookup
  llvm::GlobalVariable* gv_mod_ctx_{nullptr};
  llvm::GlobalVariable* gv_tvm_func_call_{nullptr};
  llvm::GlobalVariable* gv_tvm_get_func_from_env_{nullptr};
  llvm::GlobalVariable* gv_tvm_api_set_last_error_{nullptr};
  llvm::GlobalVariable* gv_tvm_parallel_launch_{nullptr};
  llvm::GlobalVariable* gv_tvm_parallel_barrier_{nullptr};
  std::unordered_map<String, llvm::GlobalVariable*> gv_func_map_;
  // context for direct dynamic lookup
  llvm::Function* f_tvm_func_call_{nullptr};
  llvm::Function* f_tvm_get_func_from_env_{nullptr};
  llvm::Function* f_tvm_api_set_last_error_{nullptr};
  llvm::Function* f_tvm_parallel_launch_{nullptr};
  llvm::Function* f_tvm_parallel_barrier_{nullptr};
  llvm::Function* f_tvm_register_system_symbol_{nullptr};
  // Current parallel environment scope.
  ParallelEnv parallel_env_;
  // global to packed function handle
  std::unordered_map<std::string, llvm::GlobalVariable*> func_handle_map_;
  // List of symbols to be exported to TVM system lib.
  std::vector<std::pair<std::string, llvm::Constant*>> export_system_symbols_;
  // List of functions to be registered in the FuncRegistry, if generated.
  std::vector<std::pair<std::string, llvm::Function*>> registry_functions_;
  // internal debug information, to be populated by
  std::unique_ptr<DebugInfo> dbg_info_;
  bool target_c_runtime_;
  bool is_system_lib_;

  // Get the DWARF type corresponding to the LLVM type |ty|. The current API in practice only
  // generates |int32|, and |int8*|.
  static llvm::DIType* getDebugType(IRBuilder* builder, llvm::DIBuilder* di_builder,
                                    llvm::Type* ty);
  // Adds the DWARF debug information for |function| to |dbg_info_|.
  void AddDebugInformation(llvm::Function* function);
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_LLVM_CODEGEN_CPU_H_
