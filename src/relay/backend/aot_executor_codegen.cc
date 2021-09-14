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
 * \file relay/backend/graph_codegen.cc
 * \brief Graph runtime codegen
 */

#include <tvm/ir/module.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include "te_compiler.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

using IntegerArray = Array<Integer>;
using StorageMap =
    std::unordered_map<Expr, StorageInfo, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

/**
 * This is an on demand allocator for AOT. A new temporary
 * (storage allocator identifier) is allocated for each operation.
 */
class AOTOnDemandAllocator : public MixedModeVisitor {
 public:
  // run the visitor on a function.
  void Run(const Function& func) {
    node_device_map_ = CollectDeviceInfo(func);

    for (Expr param : func->params) {
      CreateStorage(param.operator->());
    }

    GetStorage(func->body);
  }

  std::vector<int> GetReturnIds() const { return return_ids_; }

  StorageMap GetStorageMap() const { return storage_device_map_; }

  void VisitExpr_(const ConstantNode* op) final {
    CreateStorage(op);
    AssignReturnSid(GetRef<Expr>(op));
  }

  void VisitExpr_(const CallNode* op) final {
    // create token for the call node.
    CreateStorage(op);
    for (Expr arg : op->args) {
      GetStorage(arg);
    }
    AssignReturnSid(GetRef<Expr>(op));
  }

  void VisitExpr_(const VarNode* op) final { AssignReturnSid(GetRef<Expr>(op)); }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<int64_t> storage_ids;
    std::vector<DLDeviceType> device_types;
    std::vector<int64_t> storage_sizes_in_bytes;
    Expr expr = GetRef<Expr>(op);
    for (Expr field : op->fields) {
      auto sid = GetStorage(field);
      storage_ids.insert(storage_ids.end(), sid->storage_ids.begin(), sid->storage_ids.end());
      device_types.insert(device_types.end(), sid->device_types.begin(), sid->device_types.end());
      storage_sizes_in_bytes.insert(storage_sizes_in_bytes.end(),
                                    sid->storage_sizes_in_bytes.begin(),
                                    sid->storage_sizes_in_bytes.end());
    }
    storage_device_map_[expr] = StorageInfo(storage_ids, device_types, storage_sizes_in_bytes);
    AssignReturnSid(expr);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = GetRef<Expr>(op);
    auto sids = GetStorage(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), sids->storage_ids.size());
    storage_device_map_[expr] =
        StorageInfo({sids->storage_ids[op->index]}, {sids->device_types[op->index]},
                    {sids->storage_sizes_in_bytes[op->index]});
    AssignReturnSid(expr);
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void VisitExpr_(const LetNode* op) final { LOG(FATAL) << "let is not supported."; }

 private:
  void AssignReturnSid(Expr e) {
    if (storage_device_map_.find(e) != storage_device_map_.end()) {
      StorageInfo& sinfo = storage_device_map_[e];
      return_ids_.clear();
      for (auto sid : sinfo->storage_ids) {
        return_ids_.push_back(sid);
      }
    }
  }
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }
  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  size_t GetMemorySizeBytes(const TensorTypeNode* ttype) {
    ICHECK(ttype != nullptr);
    size_t size = 1;
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ICHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
      ICHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
      size *= static_cast<size_t>(pval[0]);
    }
    size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
    return size;
  }
  /*!
   * \brief Get the necessary storage for the expression.
   * \param expr The expression.
   * \return The corresponding token.
   */
  StorageInfo GetStorage(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = storage_device_map_.find(expr);
    ICHECK(it != storage_device_map_.end());
    return it->second;
  }

  /*!
   * \brief Create storage for the expression.
   * \param expr The expression.
   */
  void CreateStorage(const ExprNode* op) {
    std::vector<int64_t> storage_ids;
    std::vector<DLDeviceType> device_types;
    std::vector<int64_t> storage_sizes_in_bytes;
    Expr expr = GetRef<Expr>(op);
    int device_type_int =
        node_device_map_.count(GetRef<Expr>(op)) ? node_device_map_[expr]->value : 0;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        ICHECK(ttype);
        storage_ids.push_back(next_available_sid_++);
        storage_sizes_in_bytes.push_back(GetMemorySizeBytes(ttype));
        device_types.push_back(DLDeviceType(device_type_int));
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      ICHECK(ttype);
      storage_ids.push_back(next_available_sid_++);
      storage_sizes_in_bytes.push_back(GetMemorySizeBytes(ttype));
      device_types.push_back(DLDeviceType(device_type_int));
    }
    storage_device_map_[expr] = StorageInfo(storage_ids, device_types, storage_sizes_in_bytes);
  }
  /*! \brief mapping of expression -> storageInfo*/
  StorageMap storage_device_map_;
  /*! \brief mapping of expression -> device type*/
  Map<Expr, Integer> node_device_map_;
  /*! \brief current id of the temporary allocated*/
  int next_available_sid_{0};
  /*! \brief the set of intermediate tensors that are return variables */
  std::vector<int> return_ids_;
};

/*! \brief Code generator for AOT executor */
class AOTExecutorCodegen : public MixedModeVisitor {
 protected:
  /*!
   * \brief Utility function to allocate a DLTensor or TVMValue
   * \param  type the type of allocation
   * \param num the number of variable to allocate on the stack
   * \return PrimExpr representing the allocated object
   */
  PrimExpr StackAlloca(std::string type, size_t num) {
    Array<PrimExpr> args = {tir::StringImm(type), ConstInt32(num)};
    return tir::Call(DataType::Handle(), tir::builtin::tvm_stack_alloca(), args);
  }

  /*!
   * \brief Utility function to convert a concrete integer to a PrimExpr.
   * \param num the number to convert
   * \return PrimExpr representing num
   */
  inline PrimExpr ConstInt32(size_t num) {
    ICHECK_LE(num, std::numeric_limits<int>::max());
    return tir::make_const(DataType::Int(32), static_cast<int>(num));
  }

  /*!
   * \brief Return a vector of variables that represents the sids for the given Relay Expr
   */
  std::vector<tir::Var> PackSid(Expr expr) {
    std::vector<tir::Var> buffer_vars;
    StorageInfo& sinfo = storage_device_map_[expr];

    // Note that an expression can have multiple sids associated with it
    // e.g., returning multiple values from a function
    for (auto sid : sinfo->storage_ids) {
      // Determine if an sid is an output buffer
      auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sid);
      if (output_iter != return_sid_.end()) {
        int output_index = std::distance(return_sid_.begin(), output_iter);
        buffer_vars.push_back(main_signature_[input_vars_.size() + output_index]);
        continue;
      }

      auto sid_value = sids_table_[sid];
      buffer_vars.push_back(sid_value);
    }
    return buffer_vars;
  }

  /*!
   * brief Given an expression return the variable(s) associated with that expression
   */
  std::vector<te::Var> FindExpr(Expr arg) {
    auto input_iter = std::find(input_vars_.begin(), input_vars_.end(), arg);
    if (input_iter != input_vars_.end()) {
      // Input variable
      int main_index = std::distance(input_vars_.begin(), input_iter);
      return {main_signature_[main_index]};
    } else {
      // Storage identifier (i.e., intermediate memory)
      return PackSid(arg);
    }
  }

  /*!
   * brief Call a function with a given name
   */
  void CreateFuncCall(Call call, std::string func_name) {
    tvm::Array<PrimExpr> args{tvm::tir::StringImm(func_name)};
    std::vector<tir::Stmt> create_func_call_stmts;
    // Pack the inputs
    for (Expr arg : call->args) {
      if (params_by_expr_.find(arg) != params_by_expr_.end()) {
        auto param_handle = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::lookup_param(),
                                           {tir::StringImm(params_by_expr_[arg])});
        args.push_back(param_handle);
      } else {
        auto var_arg = FindExpr(arg);
        args.push_back(var_arg[0]);
      }
    }

    auto ret_expr = Downcast<Expr>(call);
    // Pack the return(s) value. A call node can produce multiple outputs
    for (const auto& var : PackSid(ret_expr)) {
      args.push_back(var);
    }

    // Use tvm_call_packed to execute the function unless we're calling directly
    auto calling_pattern = tvm::tir::builtin::tvm_call_cpacked();
    if (use_unpacked_api_) {
      calling_pattern = tvm::tir::builtin::call_extern();
    }

    create_func_call_stmts.push_back(
        tir::Evaluate(tvm::tir::Call(DataType::Int(32), calling_pattern, args)));

    tir::Stmt body = tir::SeqStmt(create_func_call_stmts);
    stmts_.push_back(body);
  }

  /*!
   * brief Copy a variable to the output. This function is mainly used in edge cases
   * when we want to return an input or a parameter.
   * TODO(giuseros): we should try to avoid unnecessary copy to the output, e.g., in a
   * copy-on-write fashion.
   */
  void CopyToOutput(PrimExpr out, PrimExpr in, bool pack_input, size_t size) {
    // Define intermediate DLTensor to load/store the data
    auto tmp0 = te::Var("tmp0", DataType::Handle());
    auto tmp1 = te::Var("tmp1", DataType::Handle());
    te::Var loop_idx("i", DataType::Int(32));
    auto retval_i = tir::Load(DataType::UInt(8), tmp0, loop_idx, tir::const_true());

    PrimExpr retval_get = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_struct_get(),
                                         {in, 0, tir::builtin::kArrData});
    PrimExpr tostore = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_struct_get(),
                                      {out, 0, tir::builtin::kArrData});
    if (use_unpacked_api_) {
      tostore = out;
    }

    // Do not pack the input if the flag is set or the caller
    // explicitly asked to do so (e.g., copying a param to the output)
    if (use_unpacked_api_ || !pack_input) {
      retval_get = in;
    }

    // Copy the variable from the input to the output
    tir::Stmt copy = tir::For(
        loop_idx, 0, ConstInt32(size), tir::ForKind::kSerial,
        tir::Store(tmp1, tir::Let(tmp0, retval_get, retval_i), loop_idx, tir::const_true()));
    stmts_.push_back(tir::LetStmt(tmp1, tostore, copy));
  }

  /*!
   * Utility function to string together different arguments
   */
  template <typename... Args>
  std::string MakeString(Args const&... args) {
    std::ostringstream ss;
    using List = int[];
    (void)List{0, ((void)(ss << args), 0)...};

    return ss.str();
  }

  void VisitExpr_(const CallNode* op) override {
    // Descend the call tree
    for (auto arg : op->args) {
      VisitExpr(arg);
    }

    if (op->op.as<OpNode>()) {
      LOG(FATAL) << "Operators should be transformed away; try applying"
                 << "the fuse_ops transformation to the expression.";
    } else if (op->op.as<GlobalVarNode>()) {
      GlobalVar node = GetRef<GlobalVar>(op->op.as<GlobalVarNode>());
      CreateFuncCall(GetRef<Call>(op), node->name_hint);
    } else {
      LOG(FATAL) << "TVM runtime does not support calls to " << op->op->GetTypeKey();
    }
  }

  void VisitExpr_(const VarNode* op) override {
    Expr expr = GetRef<Expr>(op);
    StorageInfo& sinfo = storage_device_map_[expr];

    // If the Var node is an output node we need to copy the content of the variable to the output
    // It's safe to check the SID here because Var StorageToken are never reallocated
    auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sinfo->storage_ids[0]);
    if (output_iter != return_sid_.end()) {
      int output_index = std::distance(return_sid_.begin(), output_iter);
      if (params_by_expr_.find(expr) != params_by_expr_.end()) {
        auto param_handle = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::lookup_param(),
                                           {tir::StringImm(params_by_expr_[expr])});
        CopyToOutput(main_signature_[input_vars_.size() + output_index], param_handle,
                     /*pack_input*/ true, sinfo->storage_sizes_in_bytes[0]);
      } else {
        auto var_expr = FindExpr(expr);
        CopyToOutput(main_signature_[input_vars_.size() + output_index], var_expr[0],
                     /*pack_input*/ true, sinfo->storage_sizes_in_bytes[0]);
      }
    }
  }

  void VisitExpr_(const ConstantNode* op) override {
    Expr expr = GetRef<Expr>(op);
    size_t index = params_.size();
    std::string name = "p" + std::to_string(index);
    StorageInfo& sinfo = storage_device_map_[expr];
    param_storage_ids_[name] = sinfo->storage_ids[0];
    params_[name] = op->data;
    params_by_expr_.Set(expr, name);

    // If the Constant node is an output node we need to copy the content of the parameter to the
    // output A Var node can only produce a single output
    auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sinfo->storage_ids[0]);
    if (output_iter != return_sid_.end()) {
      int output_index = std::distance(return_sid_.begin(), output_iter);
      auto param_handle = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::lookup_param(),
                                         {tir::StringImm(params_by_expr_[expr])});
      CopyToOutput(main_signature_[input_vars_.size() + output_index], param_handle, false,
                   sinfo->storage_sizes_in_bytes[0]);
    }
  }

  void VisitExpr_(const TupleNode* op) override {
    for (auto field : op->fields) {
      VisitExpr(field);
    }
  }

  void VisitExpr_(const LetNode* op) override {
    // TODO(giuseros): support Let nodes in AOT
    CHECK(false) << "Let not yet implemented in AOT";
  }
  void VisitExpr_(const TupleGetItemNode* op) override { VisitExpr(op->tuple); }
  void VisitExpr_(const OpNode* op) override {
    throw std::runtime_error("can not compile op in non-eta expanded form");
  }
  void VisitExpr_(const IfNode* op) override { throw std::invalid_argument("if not supported"); }
  void VisitExpr_(const FunctionNode* op) override {
    ICHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "FunctionNode only supported by custom codegen";
  }
  void VisitExpr_(const RefCreateNode* op) override {
    throw std::invalid_argument("reference not supported");
  }
  void VisitExpr_(const RefReadNode* op) override {
    throw std::invalid_argument("reference not supported");
  }
  void VisitExpr_(const RefWriteNode* op) override {
    throw std::invalid_argument("reference not supported");
  }
  void VisitExpr_(const ConstructorNode* op) override {
    throw std::invalid_argument("ADT constructor case not yet implemented");
  }
  void VisitExpr_(const MatchNode* op) override {
    throw std::invalid_argument("match case not yet implemented");
  }

  // Create the main PrimFunc to execute the graph. Please note that
  // the packed function calls don't pack their arguments. The AOT
  // runner function needs to be legalized by the LegalizePackedCalls pass.
  tir::PrimFunc CreateMainFunc(String mod_name, unsigned int relay_params) {
    tir::Stmt body = tir::SeqStmt(stmts_);

    // Allocate the sids
    std::unordered_map<int, bool> allocated;

    for (auto kv : storage_device_map_) {
      // Only allocate sids that are needed
      const bool is_input =
          (std::find(input_vars_.begin(), input_vars_.end(), kv.first) != input_vars_.end());
      const bool is_param = (params_by_expr_.find(kv.first) != params_by_expr_.end());
      if (is_input || is_param) {
        continue;
      }

      for (unsigned int i = 0; i < kv.second->storage_ids.size(); i++) {
        int size = kv.second->storage_sizes_in_bytes[i];
        int sid = kv.second->storage_ids[i];

        if (std::find(return_sid_.begin(), return_sid_.end(), sid) != return_sid_.end()) {
          continue;
        }

        // TODO(giuseros): we should allocate this once outside the PrimFunc
        // so we don't pay the price of allocation for every inference
        if (!allocated[sid]) {
          body = tir::Allocate(sids_table_[sid], DataType::Int(8), {size}, tir::const_true(), body);
        }
        allocated[sid] = true;
      }
    }

    // Define the attributes
    body = tir::AttrStmt(PrimExpr(), tvm::tir::attr::device_type, 1, body);
    body = tir::AttrStmt(PrimExpr(), tvm::tir::attr::device_id, 0, body);

    // Define the PrimFunc attributes
    Map<String, ObjectRef> dict_attrs;
    String run_func_name =
        runtime::get_name_mangled(mod_name, runtime::symbol::tvm_run_func_suffix);
    dict_attrs.Set("global_symbol", run_func_name);
    dict_attrs.Set("runner_function", Bool(true));

    // Make the PrimFunc
    return tir::PrimFunc(main_signature_, body, VoidType(), Map<tir::Var, tir::Buffer>(),
                         DictAttrs(dict_attrs));
  }

 protected:
  /*! \brief mod */
  runtime::Module* mod_;
  /*! \brief list of input expressions (i.e., variable passed by the user) */
  std::vector<Var> input_vars_;
  /*! \brief input and output variables belonging to the main function signature */
  Array<tir::Var> main_signature_;
  /*! \brief target device */
  tec::TargetMap targets_;
  /*! \brief target host */
  Target target_host_;
  /*!
   * \brief unpacked api toggle
   * When set to true the code generated will use unpacked calls to functions:
   * func(void* arg0, void* arg1)
   * Rather than packed calls:
   * func(void* args)
   * Defaults to using the packed calling convention
   */
  Bool use_unpacked_api_;

  /*!
   * \brief parameters (i.e. ConstantNodes found in the graph).
   * These are take as inputs to the GraphRuntime.
   * Maps param name to a pair of storage_id and NDArray. At runtime, the storage_id can be
   * used to lookup the parameter.
   */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief mapping between expression and parameters */
  Map<Expr, String> params_by_expr_;
  /*! \brief mapping between parameter names ("p0", "p1", etc..) and storage identifiers*/
  std::unordered_map<std::string, int64_t> param_storage_ids_;

  /*! \brief plan memory of device result */
  StorageMap storage_device_map_;
  /*! \brief mapping sid -> tir::Var */
  std::unordered_map<int, te::Var> sids_table_;
  /*! \brief lowered funcs */
  Map<String, FunctionInfo> function_metadata_;
  /*! \brief the set of statements that make the program */
  std::vector<tir::Stmt> stmts_;
  /*! \brief the list of return sids (note that the function might return more then one output */
  std::vector<int> return_sid_;

 public:
  AOTExecutorCodegen(runtime::Module* mod, const tec::TargetMap& targets, Target target_host)
      : mod_(mod),
        targets_(targets),
        target_host_(target_host),
        use_unpacked_api_(target_host->GetAttr<Bool>("unpacked-api").value_or(Bool(false))) {}

  LoweredOutput Codegen(relay::Function func, String mod_name) {
    auto aot_allocator = AOTOnDemandAllocator();
    aot_allocator.Run(func);

    // Pre-lowering storage map and memory plan
    StorageMap initial_storage_map = aot_allocator.GetStorageMap();
    StaticMemoryPlan memory_plan(initial_storage_map);

    // Build a map from each operation to device.
    tec::DeviceMap device_context_map;
    for (const auto& it : memory_plan->expr_to_storage_info) {
      auto expr = it.first;
      auto storage_info = it.second;
      auto device_types = storage_info->device_types;
      // CHECK_EQ(device_types.size(), 1);
      tvm::Device dev;
      dev.device_id = 0;
      dev.device_type = device_types[0];
      device_context_map.insert({expr, dev});
    }

    // This first phase moves from implicit use of compile engine,
    // to instead explicitly lowering the incoming IRModule, and then
    // performing the preexisting AOT executor code generation phase.
    IRModule mod = IRModule::FromExpr(func);

    IRModule lowered_mod =
        LowerTEPass(targets_, device_context_map, memory_plan, mod_name, [this](Function func) {
          // We need to maintain the constant map for external
          // functions so we pass this processing function which
          // allows us to process each function as we lower it.
          if (func->GetAttr<String>(attr::kCompiler).defined()) {
            UpdateConstants(func, &params_);
          }

          // TODO(@areusch, @jroesch): We should refactor this to
          // execute as a further pass, instead writing data to the
          // lowering process directly.
          tec::UpdateFunctionMetadata(func, this->function_metadata_);
        })(mod);

    auto lowered_main = lowered_mod->Lookup("main");
    auto lowered_main_func = GetRef<Function>(lowered_main.as<FunctionNode>());

    // Post-lowering storage map for writing main func - this should be the same map as previously
    // created, just referencing the new expressions created from lowering
    auto new_allocator = AOTOnDemandAllocator();
    new_allocator.Run(lowered_main_func);
    storage_device_map_ = new_allocator.GetStorageMap();

    for (auto input : lowered_main_func->params) {
      input_vars_.push_back(input);
      main_signature_.push_back(tir::Var("input", DataType::Handle()));
    }

    // Define the storage allocator ids
    for (auto kv : storage_device_map_) {
      for (auto sid : kv.second->storage_ids) {
        te::Var buffer_var(MakeString("sid_", sid),
                           PointerType(PrimType(DataType::Int(8)), "global"));
        sids_table_[sid] = buffer_var;
      }
    }

    // Retrieve the return sids
    return_sid_ = aot_allocator.GetReturnIds();
    for (unsigned int output_index = 0; output_index < return_sid_.size(); output_index++) {
      main_signature_.push_back(tir::Var("output", DataType::Handle()));
    }

    VisitExpr(lowered_main_func->body);

    // Create the runner function. Please note that the function is not legal yet
    // because the packed calls arguments are not wrapped in TVMValues. To make this happen we need
    // to run the LegalizePackedCalls pass.
    auto prim_func = CreateMainFunc(mod_name, lowered_main_func->params.size());
    LoweredOutput ret;

    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }

    // Build the TIR IRModule for the AOT function
    Map<GlobalVar, BaseFunc> symbol_map;
    symbol_map.Set(GlobalVar(::tvm::runtime::symbol::tvm_run_func_suffix), prim_func);
    IRModule mod_run(symbol_map);

    // Apply storage rewrite pass to the runner function to do memory planning
    auto storage_rewrite = tir::transform::StorageRewrite();
    mod_run = storage_rewrite(mod_run);

    // The workspace for main function should be calculated after performing storage_rewrite for
    // the top level TIR function.
    auto workspace_byte_alignment =
        target_host_->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
    Integer main_workspace_size = CalculateWorkspaceBytes(
        Downcast<tir::PrimFunc>(mod_run->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix)),
        workspace_byte_alignment);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");
    ICHECK(main_func_info) << "The attribute \"main_func_info\" should be set at this point.";
    main_func_info.value()->workspace_sizes.Set(target_host_, main_workspace_size);
    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());

    // Legalize AOT if needed. This means that all the packed calls
    // need to be wrapped in TVMValues (unless use_unpacked_api is set)
    if (!use_unpacked_api_) {
      auto pack_calls = tir::transform::LegalizePackedCalls();
      mod_run = pack_calls(mod_run);
    }

    ret.function_metadata = std::move(function_metadata_);

    Optional<Array<tvm::runtime::Module>> external_modules =
        lowered_mod->GetAttr<Array<tvm::runtime::Module>>("external_mods");
    ICHECK(external_modules) << "Attribute \"external_mods\" should be set at this point.";

    // This is the point where we separate the functions in the module by target
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.external_mods = external_modules.value();

    if (ret.lowered_funcs.find(target_host_) != ret.lowered_funcs.end()) {
      ret.lowered_funcs[target_host_]->Update(mod_run);
    } else {
      ret.lowered_funcs.Set(target_host_, mod_run);
    }

    std::vector<String> input_var_names(input_vars_.size());
    std::transform(input_vars_.begin(), input_vars_.end(), input_var_names.begin(),
                   [](Var input_var) -> String { return input_var->name_hint(); });
    ret.metadata =
        runtime::Metadata(input_var_names, return_sid_.size(), runtime::kTvmExecutorAot, mod_name);
    return ret;
  }
};

class AOTExecutorCodegenModule : public runtime::ModuleNode {
 public:
  AOTExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and  Map<int, Target> targets";
        void* mod = args[0];
        Map<Integer, tvm::Target> targets = args[1];
        init(mod, targets);
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Function func = args[0];
        String mod_name = args[1];
        this->output_ = codegen(func, mod_name);
      });
    } else if (name == "list_params_name") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = list_params_name(); });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        *rv = get_param_by_name(key);
      });
    } else if (name == "get_param_id") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        *rv = get_param_id(key);
      });
    } else if (name == "get_irmodule") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = get_irmodule(); });
    } else if (name == "get_external_modules") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = get_external_modules(); });
    } else if (name == "get_function_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.function_metadata;
      });
    } else if (name == "get_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = output_.metadata; });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphRuntimeCodegenModule"; }

 private:
  void init(void* mod, Map<Integer, tvm::Target> tmp) {
    tec::TargetMap targets;
    Target target_host;
    for (const auto& it : tmp) {
      auto dev_type = it.first.as<tir::IntImmNode>();
      if (!target_host.defined() && it.second->kind->device_type == kDLCPU) {
        target_host = it.second;
      }
      ICHECK(dev_type);
      targets[static_cast<DLDeviceType>(dev_type->value)] = it.second;
    }
    codegen_ = std::make_shared<AOTExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod),
                                                    targets, target_host);
  }

  LoweredOutput codegen(Function func, String mod_name) {
    return this->codegen_->Codegen(func, mod_name);
  }

  Array<runtime::String> list_params_name() {
    Array<runtime::String> ret;
    for (const auto& kv : this->output_.params) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  runtime::NDArray get_param_by_name(String key) {
    auto it = this->output_.params.find(key);
    CHECK(it != this->output_.params.end()) << "no such parameter " << key;
    return (*it).second.second;
  }

  Array<tvm::runtime::Module> get_external_modules() { return output_.external_mods; }

  int get_param_id(String key) {
    auto it = this->output_.params.find(key);
    CHECK(it != this->output_.params.end()) << "no such parameter " << key;
    return (*it).second.first;
  }

  Map<Target, IRModule> get_irmodule() { return this->output_.lowered_funcs; }

  std::shared_ptr<AOTExecutorCodegen> codegen_;
  LoweredOutput output_;
};

runtime::Module CreateAOTExecutorCodegenMod() {
  auto ptr = make_object<AOTExecutorCodegenModule>();
  return runtime::Module(ptr);
}

TVM_REGISTER_GLOBAL("relay.build_module._AOTExecutorCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateAOTExecutorCodegenMod(); });

}  // namespace backend
}  // namespace relay
}  // namespace tvm
