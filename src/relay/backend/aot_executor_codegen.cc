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
 * \file src/relay/backend/aot_executor_codegen.cc
 * \brief AOT executor codegen
 */

#include <tvm/ir/module.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include "../../target/source/codegen_source_base.h"
#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../transforms/device_aware_visitors.h"
#include "./name_transforms.h"
#include "./te_compiler.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace backend {

using StorageMap =
    std::unordered_map<Expr, StorageInfo, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

/**
 * This is an on demand allocator for AOT. A new temporary
 * (storage allocator identifier) is allocated for each operation.
 */
class AOTOnDemandAllocator : public transform::DeviceAwareExprVisitor {
 public:
  AOTOnDemandAllocator() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}

  // run the visitor on a global function.
  void Run(const Function& func) { VisitExpr(func); }

  std::vector<int> GetReturnIds() const { return return_ids_; }

  StorageMap GetStorageMap() const { return storage_device_map_; }

  using ExprVisitor::VisitExpr_;

  void VisitExpr_(const ConstantNode* op) final {
    CreateStorage(op);
    AssignReturnSid(GetRef<Expr>(op));
  }

  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    // AOTOnDemandAllocator is run both before and after lowering, so we need to handle the case
    // where the op of the call is a generic function

    Expr func;
    Array<Expr> args;

    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    if (call_lowered_props.lowered_func.defined()) {
      func = call_lowered_props.lowered_func;
      args = call_lowered_props.arguments;
    } else {  // Relay functions that have not been lowered and lowered extern functions
      func = call_node->op;
      args = call_node->args;
      if (call_node->op.as<GlobalVarNode>()) {  // Lowered extern function
        ICHECK(!(call_node->attrs.defined())) << "Extern functions should have null attributes.";
      } else {  // Relay function which has not been lowered yet
        ICHECK(call_node->op.as<FunctionNode>())
            << "Expected the call to be to a lowered primfunc, a lowered extern function or a "
               "unlowered Relay function.";
      }
    }
    VisitExpr(func);
    CreateStorage(call_node);
    for (const Expr& arg : args) {
      GetStorage(arg);
    }
    AssignReturnSid(GetRef<Expr>(call_node));
  }

  void VisitExpr_(const VarNode* op) final { AssignReturnSid(GetRef<Expr>(op)); }

  void DeviceAwareVisitExpr_(const FunctionNode* func_node) final {
    if (function_nesting() > 1) {
      // do not recurse into sub functions.
      return;
    }
    if (func_node->HasNonzeroAttr(attr::kPrimitive)) {
      // No storage needed for primitive functions.
      return;
    }
    for (const auto& param : func_node->params) {
      CreateStorage(param.get());
    }
    GetStorage(func_node->body);
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<int64_t> storage_ids;
    std::vector<VirtualDevice> virtual_devices;
    std::vector<int64_t> storage_sizes_in_bytes;
    Expr expr = GetRef<Expr>(op);
    for (Expr field : op->fields) {
      auto sid = GetStorage(field);
      storage_ids.insert(storage_ids.end(), sid->storage_ids.begin(), sid->storage_ids.end());
      virtual_devices.insert(virtual_devices.end(), sid->virtual_devices.begin(),
                             sid->virtual_devices.end());
      storage_sizes_in_bytes.insert(storage_sizes_in_bytes.end(),
                                    sid->storage_sizes_in_bytes.begin(),
                                    sid->storage_sizes_in_bytes.end());
    }
    storage_device_map_[expr] = StorageInfo(storage_ids, virtual_devices, storage_sizes_in_bytes);
    AssignReturnSid(expr);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = GetRef<Expr>(op);
    auto sids = GetStorage(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), sids->storage_ids.size());
    storage_device_map_[expr] =
        StorageInfo({sids->storage_ids[op->index]}, {sids->virtual_devices[op->index]},
                    {sids->storage_sizes_in_bytes[op->index]});
    AssignReturnSid(expr);
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void PreVisitLetBinding_(const Var& var, const Expr& value) final {
    LOG(FATAL) << "let is not supported.";
  }

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
   *
   * TODO(mbs): Cf CalculateRelayExprSizeBytes in utils.cc, GetMemorySize is graph_plan_memory.cc
   */
  size_t GetMemorySizeBytes(const TensorType& ttype) {
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
    // See through "on_device" calls.
    Expr true_expr = IgnoreOnDevice(expr);
    VisitExpr(true_expr);
    auto it = storage_device_map_.find(true_expr);
    ICHECK(it != storage_device_map_.end());
    return it->second;
  }

  /*!
   * \brief Create storage for the expression.
   */
  void CreateStorage(const ExprNode* op) {
    Expr expr = GetRef<Expr>(op);
    return CreateStorage(expr, GetVirtualDevice(expr));
  }

  /*!
   * \brief Create storage to hold the result of evaluating \p expr in \p virtual_device.
   */
  void CreateStorage(const Expr& expr, const VirtualDevice& virtual_device) {
    ICHECK(!virtual_device->IsFullyUnconstrained())
        << "invalid virtual device for expr:" << std::endl
        << PrettyPrint(expr);
    std::vector<int64_t> storage_ids;
    std::vector<VirtualDevice> virtual_devices;
    std::vector<int64_t> storage_sizes_in_bytes;
    for (const auto& ttype : FlattenTupleType(expr->checked_type())) {
      storage_ids.push_back(next_available_sid_++);
      virtual_devices.push_back(virtual_device);
      storage_sizes_in_bytes.push_back(GetMemorySizeBytes(ttype));
    }
    storage_device_map_[expr] = StorageInfo(std::move(storage_ids), std::move(virtual_devices),
                                            std::move(storage_sizes_in_bytes));
  }

  /*! \brief mapping of expression -> storageInfo */
  StorageMap storage_device_map_;
  /*! \brief current id of the temporary allocated */
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
        buffer_vars.push_back(GetBufferVarForIO(input_vars_.size() + output_index));
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
      return {GetBufferVarForIO(main_index)};
    } else {
      // Storage identifier (i.e., intermediate memory)
      return PackSid(arg);
    }
  }

  /*!
   * brief Create a function call
   * \param call_lowered_props The lowered function and the arguments to call it with
   * \param result_expr The call we got func and args from (so as to recover the storage
   * ids to hold the result).
   */
  void CreateFuncCall(CallLoweredProps call_lowered_props, const Expr& result_expr) {
    std::string func_name = call_lowered_props.lowered_func->name_hint;
    tvm::Array<PrimExpr> args{tvm::tir::StringImm(func_name)};
    std::vector<tir::Stmt> create_func_call_stmts;

    // Pack the inputs
    for (const Expr& arg : call_lowered_props.arguments) {
      if (params_by_expr_.find(arg) != params_by_expr_.end()) {
        auto param_handle = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::lookup_param(),
                                           {tir::StringImm(params_by_expr_[arg])});
        args.push_back(tvm::tir::Cast(DataType::Handle(), param_handle));
      } else {
        auto var_arg = FindExpr(arg);
        for (const auto& var : var_arg) {
          args.push_back(var);
        }
      }
    }

    // Pack the return(s) value. A call node can produce multiple outputs
    for (const auto& var : PackSid(result_expr)) {
      args.push_back(var);
    }

    // Use tvm_call_packed to execute the function unless we're calling directly
    auto calling_pattern = tvm::tir::builtin::tvm_call_cpacked();
    if (use_unpacked_api_) {
      calling_pattern = tvm::tir::builtin::call_extern();
    }

    GlobalVar global_var = call_lowered_props.lowered_func;
    tir::Var empty_var("no_device_context", DataType::Handle());
    bool has_c_device_api_context = device_contexts_.count(global_var) != 0;
    bool use_cpacked_api = !use_unpacked_api_;

    // The device context is passed to the operator in one of the following calling patterns:
    //  * Unpacked / direct function call with context:
    //      operator(arg0, arg1, device_context);
    //  * Unpacked / direct function call without context:
    //      operator(arg0, arg1);
    //  * Type-erased packed function call with context:
    //      operator(args, type_codes, int num_args, out_ret_value, out_ret_tcode,
    //      device_context_my_device)
    //  * Type-erased packed function call without context (we create an empty var for codegen):
    //      operator(args, type_codes, int num_args, out_ret_value, out_ret_tcode,
    //      no_device_context)
    if (has_c_device_api_context) {
      // call_extern calling convention with context
      tir::Var context = device_contexts_.Get(global_var).value();
      args.push_back(context);

      tir::Evaluate func_call(tvm::tir::Call(DataType::Int(32), calling_pattern, args));
      create_func_call_stmts.push_back(tir::SeqStmt({
          GenerateDeviceHook(context, "Open"),
          func_call,
          GenerateDeviceHook(context, "Close"),
      }));
    } else if (use_cpacked_api) {
      // call_cpacked calling convention needs a blank context
      args.push_back(tir::make_zero(DataType::Handle()));
      tir::Evaluate func_call(tvm::tir::Call(DataType::Int(32), calling_pattern, args));
      create_func_call_stmts.push_back(func_call);
    } else {
      // call_extern calling convention without context
      tir::Evaluate func_call(tvm::tir::Call(DataType::Int(32), calling_pattern, args));
      create_func_call_stmts.push_back(func_call);
    }

    tir::Stmt body = tir::SeqStmt(create_func_call_stmts);
    stmts_.push_back(body);
  }

  /*!
   * \brief Copy a variable to the output. This function is mainly used in edge cases
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
    // Copy the variable from the input to the output
    tir::Stmt copy =
        tir::For(loop_idx, 0, ConstInt32(size), tir::ForKind::kSerial,
                 tir::Store(tmp1, tir::Let(tmp0, in, retval_i), loop_idx, tir::const_true()));
    stmts_.push_back(tir::LetStmt(tmp1, out, copy));
  }

  /*
   * \brief Collects device context variables for passing to operators
   */
  void CollectDeviceVariables(const Map<GlobalVar, String>& device_contexts) {
    Map<TargetKind, tir::Var> target_contexts;
    TargetKindAttrMap<Bool> target_attr_map = tvm::TargetKind::GetAttrMap<Bool>("use_device_api");

    for (const auto& it : device_contexts) {
      const GlobalVar& global_var = it.first;
      const std::string device_context_name = it.second;

      Optional<TargetKind> target_kind = tvm::TargetKind::Get(device_context_name);
      if (!target_kind || !target_attr_map.count(target_kind.value())) {
        return;
      }
      if (target_attr_map[target_kind.value()]) {
        std::string context_name = SanitizeName(device_context_name);
        tir::Var device_context_var("device_context_" + context_name, DataType::Handle());

        auto pair = target_contexts.find(target_kind.value());
        if (pair != target_contexts.end()) {
          device_context_var = (*pair).second;
        } else {
          main_signature_.push_back(device_context_var);
          devices_.Set(context_name, device_context_var);
          target_contexts.Set(target_kind.value(), device_context_var);
        }

        device_contexts_.Set(global_var, device_context_var);
      }
    }
  }

  /**
   * \brief Generates a call to a given hook for all Devices found for C Device API
   * \param Name of hook to generate statements for
   * \return Statement with function calls for each device
   */
  tir::Stmt GenerateAllDeviceHook(const String& hook) {
    std::vector<tir::Stmt> device_hooks;
    for (const auto& it : devices_) {
      const String& device_name = it.first;
      const tir::Var& context = it.second;
      Array<String> sections = {"Device", device_name, hook};
      String device_hook_name = ToCFunctionStyle(PrefixName(sections));

      tir::Evaluate device_hook(tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                               {tvm::tir::StringImm(device_hook_name), context}));
      device_hooks.push_back(device_hook);
    }
    return tir::SeqStmt(device_hooks);
  }

  /**
   * \brief Generates a call to a given hook for a single Device function
   * \param Var Device context to call hook on
   * \param Name of hook to generate statements for
   * \return Statement with function call to Device API
   */
  tir::Stmt GenerateDeviceHook(const tir::Var& context, const String& hook) {
    const auto& it = std::find_if(std::begin(devices_), std::end(devices_), [&](const auto& it) {
      return it.second->name_hint == context->name_hint;
    });
    const String& device_name = (*it).first;
    Array<String> sections = {"Device", device_name, hook};
    String device_hook = ToCFunctionStyle(PrefixName(sections));

    return tir::Evaluate(tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                   {tvm::tir::StringImm(device_hook), context}));
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

  void VisitExpr_(const CallNode* call_node) override {
    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);

    if (device_copy_props.body.defined()) {
      // TODO(mbs): device_copy cleaunp
      // Suspect treating as no-op is better since already built into the StorageInfo?
      LOG(FATAL) << "The AOT executor does not currently support device_copy";
      return;
    }

    // At this point we should only see calls of the form call_lowered(@callee, (args...)),
    // where @callee can be a PrimFunc we've compiled or an external function supplied via
    // some other mechanism.
    ICHECK(call_lowered_props.lowered_func.defined())
        << "AOT does not support calling Relay functions. Attempting to call:" << std::endl
        << PrettyPrint(GetRef<Call>(call_node));
    for (const auto& arg : call_lowered_props.arguments) {
      // Evaluate the args
      VisitExpr(arg);
    }
    CreateFuncCall(call_lowered_props, GetRef<Call>(call_node));
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
        CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), param_handle,
                     /*pack_input*/ false, sinfo->storage_sizes_in_bytes[0]);
      } else {
        auto var_expr = FindExpr(expr);
        CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), var_expr[0],
                     /*pack_input*/ false, sinfo->storage_sizes_in_bytes[0]);
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
      CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), param_handle, false,
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
    LOG(FATAL) << "Let not yet implemented in AOT";
  }
  void VisitExpr_(const TupleGetItemNode* op) override { VisitExpr(op->tuple); }
  void VisitExpr_(const OpNode* op) override {
    if (GetRef<Op>(op) != CallLoweredOp()) {
      LOG(FATAL) << "All OpNodes except for call_lowered should have been expanded";
    }
  }
  void VisitExpr_(const IfNode* op) override {
    LOG(FATAL) << "All GlobalVarNodes should be removed before AOT executor's Codegen is called";
  }
  void VisitExpr_(const FunctionNode* op) override {
    ICHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "FunctionNode only supported by custom codegen";
  }
  void VisitExpr_(const RefCreateNode* op) override {
    LOG(FATAL) << "AOT executor does not support references (found RefCreateNode)";
  }
  void VisitExpr_(const RefReadNode* op) override {
    LOG(FATAL) << "AOT executor does not support references (found RefReadNode)";
  }
  void VisitExpr_(const RefWriteNode* op) override {
    LOG(FATAL) << "AOT executor does not support references (found RefWriteNode)";
  }
  void VisitExpr_(const ConstructorNode* op) override {
    LOG(FATAL) << "AOT executor does not support ADTs (found ConstructorNode)";
  }
  void VisitExpr_(const MatchNode* op) override {
    LOG(FATAL) << "AOT executor does not support matching (found MatchNode)";
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
          PointerType ptype = Downcast<PointerType>(sids_table_[sid]->type_annotation);
          DataType element_type = Downcast<PrimType>(ptype->element_type)->dtype;
          body = tir::Allocate(sids_table_[sid], element_type, {size}, tir::const_true(), body);
        }
        allocated[sid] = true;
      }
    }

    // Define the PrimFunc attributes
    Map<String, ObjectRef> dict_attrs;
    String run_func_name =
        runtime::get_name_mangled(mod_name, runtime::symbol::tvm_run_func_suffix);
    dict_attrs.Set("global_symbol", run_func_name);
    dict_attrs.Set("runner_function", Bool(true));
    dict_attrs.Set(tvm::attr::kTarget, target_host_);

    tir::Stmt device_activations = GenerateAllDeviceHook("Activate");
    tir::Stmt device_deactivations = GenerateAllDeviceHook("Deactivate");
    tir::Stmt final_body = tir::SeqStmt({device_activations, body, device_deactivations});

    // Make the PrimFunc
    return tir::PrimFunc(main_signature_, final_body, VoidType(), main_buffer_map_,
                         DictAttrs(dict_attrs));
  }

  /*!
   * brief Access IO vars using the buffer vars and
   * not the actual var.
   */
  tir::Var GetBufferVarForIO(int index) { return main_buffer_map_[main_signature_[index]]->data; }

  /*!
   * brief Create tir::Var for input/output while updating
   * the buffer_maps.
   */
  void CreateIOVar(const Expr& expr, std::string name) {
    if (expr->IsInstance<TupleNode>()) {
      Tuple tuple = Downcast<Tuple>(expr);
      for (unsigned i = 0; i < tuple->fields.size(); i++) {
        CreateIOVar(tuple->fields[i], name + std::to_string(i) + "_");
      }
    } else {
      tir::Var var = tir::Var(name, DataType::Handle());
      main_signature_.push_back(var);
      auto tensor_type = expr->checked_type().as<TensorTypeNode>();
      DataType elem_type = tensor_type->dtype;
      tir::Var buffer_var =
          tir::Var(name + "_buffer_var", PointerType(PrimType(elem_type), "global"));
      tir::Buffer buffer = tir::Buffer(buffer_var, elem_type, tensor_type->shape, {}, 0,
                                       name + "_buffer", 16, 1, tir::BufferType::kDefault);
      main_buffer_map_.Set(var, buffer);
    }
  }

  /*!
   * brief Run USMP to plan memory for lowered IRModule
   */
  IRModule PlanMemoryWithUSMP(const IRModule& mod) {
    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    Integer workspace_byte_alignment =
        executor_config->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
    IRModule lowered_mod = mod->ShallowCopy();
    lowered_mod = tir::transform::UnifiedStaticMemoryPlanner()(lowered_mod);
    // Update workspace size based on the pool allocations.
    for (const auto& kv : function_metadata_) {
      if (lowered_mod->ContainGlobalVar(kv.first) &&
          lowered_mod->Lookup(kv.first)->IsInstance<tir::PrimFuncNode>()) {
        tir::PrimFunc pfunc = Downcast<tir::PrimFunc>(lowered_mod->Lookup(kv.first));
        Target tgt = pfunc->GetAttr<Target>(tvm::attr::kTarget).value();
        const auto& ws = CalculateWorkspaceBytes(pfunc, workspace_byte_alignment);
        kv.second->workspace_sizes.Set(tgt, ws);
      }
    }
    Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
        lowered_mod->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
    backend::FunctionInfo main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info").value();
    main_func_info->workspace_sizes.clear();
    if (allocated_pool_infos) {
      for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
        for (const auto& kv : allocated_pool_info->pool_info->target_access) {
          Target tgt = kv.first;
          if (main_func_info->workspace_sizes.find(tgt) == main_func_info->workspace_sizes.end()) {
            main_func_info->workspace_sizes.Set(tgt, allocated_pool_info->allocated_size);
          } else {
            main_func_info->workspace_sizes.Set(tgt,
                                                main_func_info->workspace_sizes[tgt]->value +
                                                    allocated_pool_info->allocated_size->value);
          }
        }
      }
    }
    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info);
    return lowered_mod;
  }

  /*!
   * brief Run StorageRewrite to plan memory for lowered IRModule
   */
  IRModule PlanMemoryWithStorageRewrite(const IRModule& mod) {
    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    Integer workspace_byte_alignment =
        executor_config->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
    IRModule lowered_mod = mod->ShallowCopy();
    // Running StorageRewrite just on the main function
    tir::PrimFunc tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix));
    IRModule main_func_mod;
    main_func_mod->Update(lowered_mod->GetGlobalVar(::tvm::runtime::symbol::tvm_run_func_suffix),
                          tir_main_func);
    main_func_mod = tir::transform::StorageRewrite()(main_func_mod);
    lowered_mod->Update(lowered_mod->GetGlobalVar(::tvm::runtime::symbol::tvm_run_func_suffix),
                        main_func_mod->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix));
    tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix));
    // Use the PrimFunc to calculate the workspace required to service the allocates
    Integer main_workspace_size_bytes =
        CalculateWorkspaceBytes(tir_main_func, workspace_byte_alignment);
    backend::FunctionInfo main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info").value();
    main_func_info->workspace_sizes.Set(target_host_, main_workspace_size_bytes);
    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info);
    return lowered_mod;
  }

 protected:
  /*! \brief mod */
  runtime::Module* mod_;
  /*! \brief list of input expressions (i.e., variable passed by the user) */
  std::vector<Var> input_vars_;
  /*! \brief map of device contexts variables */
  Map<String, tir::Var> devices_;
  /*! \brief map of GlobalVars to C Device API contexts */
  Map<GlobalVar, tir::Var> device_contexts_;
  /*! \brief input and output variables belonging to the main function signature */
  Array<tir::Var> main_signature_;
  /*! \brief input and output variables belonging to the main function signature */
  Map<tir::Var, tir::Buffer> main_buffer_map_;
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
      : mod_(mod), targets_(targets), target_host_(target_host), use_unpacked_api_(Bool(false)) {}

  LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    VLOG_CONTEXT << "AOT";
    for (const auto& kv : targets_) {
      VLOG(1) << "target: " << kv.second->ToDebugString();
    }
    ICHECK(target_host_.defined()) << "require a target_host to be given for AOT codegen";
    VLOG(1) << "target host: " << target_host_->ToDebugString();

    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    String interface_api = executor_config->GetAttr<String>("interface-api").value_or("packed");
    Integer workspace_byte_alignment =
        executor_config->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
    use_unpacked_api_ = executor_config->GetAttr<Bool>("unpacked-api").value_or(Bool(false));

    // TODO(mbs): Plumb from compiler config
    VirtualDevice host_virtual_device = VirtualDevice::ForTarget(target_host_);

    IRModule lowered_mod = tec::LowerTEPass(
        mod_name,
        [this, workspace_byte_alignment](BaseFunc func) {
          // We need to maintain the constant map for external
          // functions so we pass this processing function which
          // allows us to process each function as we lower it.
          if (func->GetAttr<String>(attr::kCompiler).defined()) {
            UpdateConstants(func, &params_);
          }

          // TODO(@areusch, @jroesch): We should refactor this to
          // execute as a further pass, instead writing data to the
          // lowering process directly.
          tec::UpdateFunctionMetadata(func, this->function_metadata_, workspace_byte_alignment);
        },
        host_virtual_device)(mod);

    auto lowered_main = lowered_mod->Lookup("main");
    auto lowered_main_func = GetRef<Function>(lowered_main.as<FunctionNode>());

    // Post-lowering storage map for writing main func
    AOTOnDemandAllocator final_aot_allocator;
    final_aot_allocator.Run(lowered_main_func);
    storage_device_map_ = final_aot_allocator.GetStorageMap();

    // TODO(@electriclilies, @jroesch, @Mousius): remove UpdateMainWorkspaceSize
    StaticMemoryPlan memory_plan(storage_device_map_);
    backend::FunctionInfo func_info =
        tec::UpdateMainWorkspaceSize(lowered_mod, targets_, memory_plan->expr_to_storage_info);
    lowered_mod = WithAttr(lowered_mod, "main_func_info", func_info);

    for (auto input : lowered_main_func->params) {
      input_vars_.push_back(input);
      std::string input_name = SanitizeName(input->name_hint());
      CreateIOVar(input, input_name);
    }

    // Define the storage allocator ids
    for (auto kv : storage_device_map_) {
      for (auto sid : kv.second->storage_ids) {
        // The buffer_var is created with storage_scope to be global.workspace to be serviced by
        // TVMBackendAllocWorkspace(TVMBAW) calls, explicitly. The reasoning being the executor
        // allocates should be serviced by TVMBAWs as the data could be accessed by many devices and
        // should not be lowered to the stack. For more details please refer to the discussion here:
        // https://github.com/apache/tvm/issues/9022
        te::Var buffer_var(MakeString("sid_", sid),
                           PointerType(PrimType(DataType::Int(8)), "global.workspace"));
        sids_table_[sid] = buffer_var;
      }
    }

    // Retrieve the return sids
    return_sid_ = final_aot_allocator.GetReturnIds();
    // Insert outputs to main func signature
    CreateIOVar(lowered_main_func->body, "output");

    CollectDeviceVariables(lowered_mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value());
    VisitExpr(lowered_main_func->body);

    // Create the runner function. Please note that the function is not legal yet
    // because the packed calls arguments are not wrapped in TVMValues. To make this happen we need
    // to run the LegalizePackedCalls pass.
    LoweredOutput ret;

    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }

    // AoT Executor codegen works completely on TIR beyond this point, hence removing relay main
    // function and replacing it with its TIR version. We should try to make this a Pass.
    lowered_mod->Remove(lowered_mod->GetGlobalVar("main"));
    auto prim_func = CreateMainFunc(mod_name, lowered_main_func->params.size());
    lowered_mod->Update(GlobalVar(::tvm::runtime::symbol::tvm_run_func_suffix), prim_func);
    // Parallel for loops are not supported in AoT codegen.
    lowered_mod = tir::transform::ConvertForLoopsToSerial()(lowered_mod);

    transform::PassContext pass_ctx = transform::PassContext::Current();
    bool enable_usmp = pass_ctx->GetConfig<Bool>(kUSMPEnableOption, Bool(false)).value();
    if (enable_usmp) {
      lowered_mod = PlanMemoryWithUSMP(lowered_mod);
    } else {
      lowered_mod = PlanMemoryWithStorageRewrite(lowered_mod);
    }
    ret.function_metadata = std::move(function_metadata_);

    // Legalize AOT if needed. This means that all the packed calls
    // need to be wrapped in TVMValues (unless use_unpacked_api is set)
    if (!use_unpacked_api_) {
      auto pack_calls = tir::transform::LegalizePackedCalls();
      lowered_mod = pack_calls(lowered_mod);
    }

    Optional<Array<tvm::runtime::Module>> external_modules =
        lowered_mod->GetAttr<Array<tvm::runtime::Module>>("external_mods");
    ICHECK(external_modules) << "Attribute \"external_mods\" should be set at this point.";

    // This is the point where we separate the functions in the module by target
    VLOG(1) << "lowered module:" << std::endl << PrettyPrint(lowered_mod);
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    VLOG(1) << "per-target modules:";
    for (const auto& kv : ret.lowered_funcs) {
      VLOG(1) << "target:" << std::endl
              << kv.first->ToDebugString() << std::endl
              << "maps to:" << std::endl
              << PrettyPrint(kv.second);
    }

    ret.external_mods = external_modules.value();

    Map<tir::Var, tir::usmp::AllocatedPoolInfo> pool_var_info;
    std::vector<tir::Var> pool_vars;
    tir::PrimFunc tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix));
    Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
        tir_main_func->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
    if (allocated_pool_infos) {
      for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
        int pool_var_index = allocated_pool_info->pool_var_idx.value()->value;
        pool_vars.push_back(tir_main_func->params[pool_var_index]);
        pool_var_info.Set(tir_main_func->params[pool_var_index], allocated_pool_info);
      }
    }
    Array<String> devices = ListDevices();
    Array<tir::Var> inputs =
        Array<tir::Var>(tir_main_func->params.begin(),
                        tir_main_func->params.begin() + tir_main_func->params.size() -
                            return_sid_.size() - pool_vars.size() - devices.size());

    std::vector<String> output_var_names;
    if (auto opt = func->GetAttr<Array<String>>("output_tensor_names")) {
      Array<String> output_tensor_names = opt.value();
      for (size_t i = 0; i < output_tensor_names.size(); ++i) {
        output_var_names.push_back(output_tensor_names[i]);
      }
    }

    // If output names have not been specified then generate default output names
    if (output_var_names.size() == 0) {
      if (return_sid_.size() == 1) {
        output_var_names.push_back(String("output"));
      } else {
        for (size_t i = 0; i < return_sid_.size(); ++i) {
          output_var_names.push_back(String("output" + std::to_string(i)));
        }
      }
    }

    ret.metadata = ExecutorCodegenMetadata(inputs, pool_vars, devices, output_var_names,
                                           runtime::kTvmExecutorAot, mod_name, interface_api,
                                           use_unpacked_api_, pool_var_info);
    return ret;
  }

  /*!
   * \brief Get list of devices found
   * \return List of devices
   */
  Array<String> ListDevices() {
    std::vector<String> device_names(devices_.size());
    std::transform(devices_.begin(), devices_.end(), device_names.begin(),
                   [](const auto& it) -> String { return it.first; });
    return device_names;
  }
};  // namespace backend

class AOTExecutorCodegenModule : public runtime::ModuleNode {
 public:
  AOTExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and  Map<int, Target> targets";
        void* mod = args[0];
        TargetMap targets = args[1];
        init(mod, targets);
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        IRModule mod = args[0];
        Function func = args[1];
        String mod_name = args[2];
        this->output_ = this->codegen_->Codegen(mod, func, mod_name);
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
    } else if (name == "get_devices") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->codegen_->ListDevices();
      });
    } else if (name == "get_executor_codegen_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = output_.metadata; });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphRuntimeCodegenModule"; }

 private:
  void init(void* mod, TargetMap tmp) {
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
