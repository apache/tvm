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
 * \file src/relay/backend/aot/aot_lower_main.cc
 * \brief Lower the Relay main func into an AOT TIR main func.
 */
#include "./aot_lower_main.h"

#include <tvm/runtime/name_transforms.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include "../../op/call/call.h"
#include "../../op/memory/device_copy.h"
#include "../../op/memory/memory.h"
#include "../../transforms/device_aware_visitors.h"
#include "../name_transforms.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

/*!
 * \brief Looks at the expressions in a given function and produces an Expr to
 * StorageInfo map by assigning one or more StorageInfos to the expressions that
 * require storage.
 *
 * This pass is leveraged by AOTMainLowerer to perform an initial naive allocation
 * for tensors in the Relay main function. The resulting storage map is then lowered
 * into TIR allocations by AOTMainLowerer where the allocation can be subsequently
 * optimized by later passes (e.g. USMP).
 */
class ExprAllocator : public transform::DeviceAwareExprVisitor {
 public:
  ExprAllocator() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}

  // run the visitor on a global function.
  void Run(const Function& func) { VisitExpr(func); }

  std::vector<int> GetReturnSIDs() const { return return_sids_; }

  StorageMap GetStorageMap() const { return expr_storage_map_; }

  using ExprVisitor::VisitExpr_;

  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    Array<Expr> args;

    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    if (call_lowered_props.lowered_func.defined()) {
      args = call_lowered_props.arguments;
    } else {  // Relay functions that have not been lowered and lowered extern functions
      args = call_node->args;
      if (call_node->op.as<GlobalVarNode>()) {  // Lowered extern function
        ICHECK(!(call_node->attrs.defined())) << "Extern functions should have null attributes.";
      } else {  // Relay function which has not been lowered yet
        ICHECK(call_node->op.as<FunctionNode>())
            << "Expected the call to be to a lowered primfunc, a lowered extern function or a "
               "unlowered Relay function.";
      }
    }
    CreateStorage(call_node);
    for (const Expr& arg : args) {
      VisitExpr(arg);
    }
    AssignReturnSID(GetRef<Expr>(call_node));
  }

  void DeviceAwareVisitExpr_(const FunctionNode* func_node) final {
    if (function_nesting() > 1) {
      // Do not recurse into sub functions.
      return;
    }
    for (const auto& param : func_node->params) {
      CreateStorage(param.get());
    }
    VisitExpr(func_node->body);
  }

  void PreVisitLetBinding_(const Var& var, const Expr& value) final {
    VisitExpr(value);
    StorageInfo si = GetStorage(value);
    expr_storage_map_[var] = si;
  }

  void VisitExpr_(const ConstantNode* op) final {
    CreateStorage(op);
    AssignReturnSID(GetRef<Expr>(op));
  }

  void VisitExpr_(const VarNode* op) final { AssignReturnSID(GetRef<Expr>(op)); }

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
    expr_storage_map_[expr] = StorageInfo(storage_ids, virtual_devices, storage_sizes_in_bytes);
    AssignReturnSID(expr);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = GetRef<Expr>(op);
    auto sids = GetStorage(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), sids->storage_ids.size());
    expr_storage_map_[expr] =
        StorageInfo({sids->storage_ids[op->index]}, {sids->virtual_devices[op->index]},
                    {sids->storage_sizes_in_bytes[op->index]});
    AssignReturnSID(expr);
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "'If' is not supported."; }

 private:
  /*!
   * \brief Assign the expression's storage IDs as the return storage IDs.
   * \note This is called when visiting every expression on the understanding
   * that the returned expression will be visited last.
   */
  void AssignReturnSID(const Expr& e) {
    if (expr_storage_map_.find(e) != expr_storage_map_.end()) {
      StorageInfo& sinfo = expr_storage_map_[e];
      return_sids_.clear();
      for (auto sid : sinfo->storage_ids) {
        return_sids_.push_back(sid);
      }
    }
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
    auto it = expr_storage_map_.find(true_expr);
    ICHECK(it != expr_storage_map_.end()) << "Could not find " << true_expr->GetTypeKey() << " "
                                          << PrettyPrint(true_expr) << " in storage device map";
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
      storage_sizes_in_bytes.push_back(GetMemorySizeBytes(ttype->shape, ttype->dtype));
    }
    expr_storage_map_[expr] = StorageInfo(std::move(storage_ids), std::move(virtual_devices),
                                          std::move(storage_sizes_in_bytes));
  }

  /*! \brief Map between Exprs and StorageInfos */
  StorageMap expr_storage_map_;
  /*! \brief The next available storage ID to be used */
  int next_available_sid_{0};
  /*! \brief The storage IDs that correspond to return values */
  std::vector<int> return_sids_;
};

std::tuple<StorageMap, std::vector<int>> CreateStorage(const Function& func) {
  ExprAllocator expr_allocator;
  expr_allocator.Run(func);
  return std::make_tuple(expr_allocator.GetStorageMap(), expr_allocator.GetReturnSIDs());
}

class AOTMainLowerer : public MixedModeVisitor {
 public:
  AOTMainLowerer(tvm::CompilationConfig config, CallType call_type)
      : config_(config), call_type_(call_type) {}

  IRModule Lower(IRModule mod, String mod_name) {
    VLOG_CONTEXT << "AOT";
    IRModule lowered_mod = GetRef<IRModule>(mod.CopyOnWrite());

    auto lowered_main = lowered_mod->Lookup("main");
    auto lowered_main_func = Downcast<Function>(lowered_main);

    // Assign StorageInfo to all the Relay exprs and get the return SIDs
    std::tie(expr_storage_map_, return_sid_) = CreateStorage(lowered_main_func);

    for (auto input : lowered_main_func->params) {
      input_vars_.push_back(input);
      std::string input_name = tvm::runtime::SanitizeName(input->name_hint());
      // We don't want the compiler changing input names in the
      // event of a sanitization collision. Therefore, enforcing
      // the var created to use the input_name strictly.
      CreateIOVar(input, input_name, /*use_unique_name = */ false);
    }

    // Define the storage allocator ids
    for (auto kv : expr_storage_map_) {
      for (auto sid : kv.second->storage_ids) {
        // The buffer_var is created with storage_scope to be global.workspace to be serviced by
        // TVMBackendAllocWorkspace(TVMBAW) calls, explicitly. The reasoning being the executor
        // allocates should be serviced by TVMBAWs as the data could be accessed by many devices and
        // should not be lowered to the stack. For more details please refer to the discussion here:
        // https://github.com/apache/tvm/issues/9022
        tir::Var buffer_var(MakeString("sid_", sid),
                            PointerType(PrimType(DataType::Int(8)), "global.workspace"));
        sids_table_[sid] = buffer_var;
      }
    }

    // Create output vars for the TIR main func
    // If output tensor names were provided use them
    if (auto opt = lowered_main->GetAttr<Array<String>>("output_tensor_names")) {
      Array<String> output_tensor_names = opt.value();
      Expr output_expr = lowered_main_func->body;
      if (output_expr->checked_type()->IsInstance<TupleTypeNode>()) {
        TupleType output_tuple_type = Downcast<TupleType>(output_expr->checked_type());
        for (unsigned i = 0; i < output_tuple_type->fields.size(); i++) {
          // AoT Executor Codegen does not create these names,
          // thus should be used as they are provided.
          CreateIOVar(output_tuple_type->fields[i], output_tensor_names[i],
                      /*use_unique_name = */ false);
        }
      } else {
        // AoT Executor Codegen does not create these names,
        // thus should be used as they are provided.
        CreateIOVar(lowered_main_func->body, output_tensor_names[0], /*use_unique_name = */ false);
      }
    } else {
      // If output tensor names are not provided we will generate output(x)
      // where x is a counter to create unique names.
      if (lowered_main_func->body->checked_type()->IsInstance<TupleTypeNode>()) {
        CreateIOVar(lowered_main_func->body, "output");
      } else {
        CreateIOVar(lowered_main_func->body, "output", /*use_unique_name = */ false);
      }
    }

    CollectDeviceVariables(lowered_mod->GetAttr<Map<GlobalVar, String>>("device_contexts")
                               .value_or(Map<GlobalVar, String>()));
    VisitExpr(lowered_main_func->body);

    // Remove the Relay main and replace it with the lowered TIR version
    lowered_mod->Remove(lowered_mod->GetGlobalVar("main"));
    auto tir_main_func = CreateMainFunc(mod_name);
    lowered_mod->Update(GlobalVar(runtime::symbol::tvm_module_main), tir_main_func);
    lowered_mod = tir::transform::RemoveNoOp()(lowered_mod);
    return lowered_mod;
  }

  void VisitExpr_(const CallNode* call_node) override {
    OnDeviceProps on_device_props = GetOnDeviceProps(call_node);
    if (on_device_props.body.defined()) {
      VisitExpr(on_device_props.body);
      return;
    }

    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);

    if (device_copy_props.body.defined()) {
      // TODO(mbs): device_copy cleaunp
      // Suspect treating as no-op is better since already built into the StorageInfo?
      LOG(FATAL) << "The AOT executor does not currently support device_copy";
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
    StorageInfo& sinfo = expr_storage_map_[expr];

    // Let bound vars refer to a value, so these should not be considered "output" vars.
    if (let_bound_vars_.find(GetRef<Var>(op)) != let_bound_vars_.end()) {
      return;
    }

    // If the Var node is an output node we need to copy the content of the variable to the output
    // It's safe to check the SID here because Var StorageToken are never reallocated
    auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sinfo->storage_ids[0]);
    if (output_iter != return_sid_.end()) {
      int output_index = std::distance(return_sid_.begin(), output_iter);
      auto var_expr = FindExpr(expr);
      CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), var_expr[0],
                   /*pack_input*/ false, sinfo->storage_sizes_in_bytes[0]);
    }
  }

  void VisitExpr_(const ConstantNode* op) override {
    Expr expr = GetRef<Expr>(op);
    ICHECK(expr_storage_map_.find(expr) != expr_storage_map_.end())
        << "Storage map did not contain constant expr " << PrettyPrint(expr);
    StorageInfo& sinfo = expr_storage_map_[expr];
    std::stringstream ss;
    ss << "constant_" << constant_map_.size();

    tir::Var constant(ss.str(), PointerType(PrimType(DataType(op->data->dtype))));
    constant_map_[constant] = op;
    auto sid = sinfo->storage_ids[0];
    sids_table_[sid] = constant;

    // If the Constant node is an output node we need to copy the content of the parameter to the
    // output. A node can only produce a single output
    auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sid);
    if (output_iter != return_sid_.end()) {
      int output_index = std::distance(return_sid_.begin(), output_iter);
      auto param_handle = tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::lookup_param(),
                                         {tir::StringImm(ss.str())});
      CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), constant,
                   /* pack_input */ false, sinfo->storage_sizes_in_bytes[0]);
    }
  }

  void VisitExpr_(const TupleNode* op) override {
    for (auto field : op->fields) {
      VisitExpr(field);
    }
  }

  void VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      let_bound_vars_.insert(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const TupleGetItemNode* op) override { VisitExpr(op->tuple); }
  void VisitExpr_(const OpNode* op) override {
    if (GetRef<Op>(op) != CallLoweredOp() && GetRef<Op>(op) != OnDeviceOp()) {
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

 private:
  /*!
   * \brief Create the main PrimFunc to execute the graph.
   * \note The packed function calls don't pack their arguments. The AOT
   * runner function needs to be legalized by the LegalizePackedCalls pass.
   */
  tir::PrimFunc CreateMainFunc(String mod_name) {
    tir::Stmt body = tir::SeqStmt::Flatten(stmts_);
    // Allocate the sids
    std::unordered_map<int, bool> allocated;
    std::vector<std::pair<int64_t, int64_t>> sids_to_allocate;

    for (auto kv : expr_storage_map_) {
      // Only allocate sids that are needed
      const bool is_input =
          (std::find(input_vars_.begin(), input_vars_.end(), kv.first) != input_vars_.end());
      if (is_input) {
        continue;
      }

      for (unsigned int i = 0; i < kv.second->storage_ids.size(); i++) {
        sids_to_allocate.push_back(
            std::make_pair(kv.second->storage_ids[i], kv.second->storage_sizes_in_bytes[i]));
      }
    }

    // Sort the SID allocation to make output deterministic
    std::sort(sids_to_allocate.begin(), sids_to_allocate.end());

    for (auto p : sids_to_allocate) {
      int sid = p.first;
      int size = p.second;

      if (std::find(return_sid_.begin(), return_sid_.end(), sid) != return_sid_.end()) {
        continue;
      }

      // Make sure it hasn't already been allocated, this can happen
      // with let-bound var/value pairs.
      if (allocated.find(sid) != allocated.end()) {
        continue;
      }

      allocated[sid] = constant_map_.count(sids_table_[sid]);

      // TODO(giuseros): we should allocate this once outside the PrimFunc
      // so we don't pay the price of allocation for every inference
      if (!allocated[sid]) {
        PointerType ptype = Downcast<PointerType>(sids_table_[sid]->type_annotation);
        DataType element_type = Downcast<PrimType>(ptype->element_type)->dtype;
        body = tir::Allocate(sids_table_[sid], element_type, {size}, tir::const_true(), body);
      }
      allocated[sid] = true;
    }

    for (auto kv : constant_map_) {
      auto buffer_var = kv.first;
      auto dtype = DataType(kv.second->data->dtype);

      int ndim = kv.second->data->ndim;
      Array<PrimExpr> extents;

      for (int i = 0; i < ndim; i++) {
        int shape = kv.second->data->shape[i];
        extents.push_back(tir::make_const(DataType::Int(32), shape, Span()));
      }
      body = tir::AllocateConst(buffer_var, dtype, extents, kv.second->data, body);
    }

    // Define the PrimFunc attributes
    Map<String, ObjectRef> dict_attrs;
    String run_func_name = runtime::get_name_mangled(mod_name, runtime::symbol::tvm_module_main);
    dict_attrs.Set("global_symbol", run_func_name);
    dict_attrs.Set("runner_function", Bool(true));
    dict_attrs.Set(tvm::attr::kTarget, config_->host_target);
    Array<tir::Var> input_vars =
        Array<tir::Var>(main_signature_.begin(), main_signature_.begin() + input_vars_.size());
    dict_attrs.Set("input_vars", input_vars);
    Array<tir::Var> output_vars =
        Array<tir::Var>(main_signature_.begin() + input_vars_.size(),
                        main_signature_.begin() + input_vars_.size() + return_sid_.size());
    dict_attrs.Set("output_vars", output_vars);
    Array<String> device_names;
    for (const auto& it : devices_) {
      device_names.push_back(it.first);
    }
    dict_attrs.Set("devices", device_names);

    tir::Stmt device_activations = GenerateAllDeviceHook("Activate");
    tir::Stmt device_deactivations = GenerateAllDeviceHook("Deactivate");
    tir::Stmt final_body = tir::SeqStmt({device_activations, body, device_deactivations});

    // Make the PrimFunc
    return tir::PrimFunc(main_signature_, final_body, VoidType(), main_buffer_map_,
                         DictAttrs(dict_attrs));
  }

  /*!
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
        std::string context_name = tvm::runtime::SanitizeName(device_context_name);
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

  /*!
   * \brief Return a vector of variables that represents the sids for the given Relay Expr
   */
  std::vector<tir::Var> PackSid(Expr expr) {
    std::vector<tir::Var> buffer_vars;

    ICHECK(expr_storage_map_.find(expr) != expr_storage_map_.end())
        << "Storage map did not contain constant expr " << PrettyPrint(expr);
    StorageInfo& sinfo = expr_storage_map_[expr];

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
   * \brief Given an expression return the variable(s) associated with that expression
   */
  std::vector<tir::Var> FindExpr(Expr arg) {
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

  void PushArgs(const Expr& expr, const std::vector<tir::Var>& sids, Array<PrimExpr>* args) {
    const TupleNode* t = expr.as<TupleNode>();
    if (t != nullptr) {
      CHECK_EQ(sids.size(), t->fields.size()) << "Relay tuple does not map 1:1 into TIR; AOT can't "
                                                 "handle this type of Relay Expr in a CallNode.";
    }

    args->insert(args->end(), sids.begin(), sids.end());
  }

  /*!
   * \brief Wraps a call_extern with a tvm_check_return annotation if required otherwise
   * returns the passed Call
   */
  tir::Call AddCheckReturn(tir::Call existing_call) {
    Array<PrimExpr> args = {tir::make_const(DataType::Int(32, 1), 0, Span()),
                            tir::make_const(DataType::Int(32, 1), -1, Span()), existing_call};
    return tir::Call(DataType::Int(32), tir::builtin::tvm_check_return(), args);
  }

  /*!
   * \brief Create a function call
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
      auto sids = FindExpr(arg);
      PushArgs(arg, sids, &args);
    }

    // Pack the return(s) value. A call node can produce multiple outputs
    auto result_expr_sid = PackSid(result_expr);
    PushArgs(result_expr, result_expr_sid, &args);

    GlobalVar global_var = call_lowered_props.lowered_func;
    bool has_c_device_api_context = device_contexts_.count(global_var) != 0;
    tir::Var device_context;
    tir::Stmt func_call;

    switch (call_type_) {
      case CallType::kUnpacked: {
        // call_extern calling convention with optional context
        if (has_c_device_api_context) {
          device_context = device_contexts_.Get(global_var).value();
          args.push_back(device_context);
        }
        func_call = tir::Evaluate(AddCheckReturn(
            tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(), args)));
        break;
      }
      case CallType::kCPacked: {
        if (has_c_device_api_context) {
          device_context = device_contexts_.Get(global_var).value();
          args.push_back(device_context);
        } else {
          // NOTE: LowerTVMBuiltin expects some device_context placeholder.
          args.push_back(tir::make_zero(DataType::Handle()));
        }
        func_call = tir::Evaluate(
            tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::tvm_call_cpacked(), args));
        create_func_call_stmts.push_back(func_call);
        break;
      }
      case CallType::kPacked: {
        // call_packed does not accept a device context.
        CHECK(!has_c_device_api_context) << "CallType::kPacked does not accept a device context";
        func_call = tir::Evaluate(AddCheckReturn(
            tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::tvm_call_packed(), args)));
        create_func_call_stmts.push_back(func_call);
        break;
      }
      default:
        ICHECK(false) << "Unknown CallType: " << call_type_;
    }

    ICHECK(func_call.defined()) << "Must define func_call";

    if (has_c_device_api_context) {
      func_call = tir::SeqStmt(Array<tir::Stmt>({
          GenerateDeviceHook(device_context, "Open"),
          func_call,
          GenerateDeviceHook(device_context, "Close"),
      }));
    }

    tir::Stmt body = tir::SeqStmt::Flatten(func_call);
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
    tir::Buffer tmp_read =
        tir::decl_buffer({IntImm(DataType::UInt(64), size)}, DataType::UInt(8), "tmp_read");
    tir::Buffer tmp_write =
        tir::decl_buffer({IntImm(DataType::UInt(64), size)}, DataType::UInt(8), "tmp_write");
    te::Var loop_idx("i", DataType::Int(32));
    auto retval_i = tir::BufferLoad(tmp_read, {loop_idx});
    // Copy the variable from the input to the output
    tir::Stmt copy = tir::For(
        loop_idx, 0, tir::make_const(DataType::Int(32, 1), size, Span()), tir::ForKind::kSerial,
        tir::BufferStore(tmp_write, tir::Let(tmp_read->data, in, retval_i), {loop_idx}));
    stmts_.push_back(tir::LetStmt(tmp_write->data, out, copy));
  }

  /*!
   * \brief Generates a call to a given hook for all Devices found for C Device API
   * \param hook Name of hook to generate statements for
   * \return Statement with function calls for each device
   */
  tir::Stmt GenerateAllDeviceHook(const String& hook) {
    std::vector<tir::Stmt> device_hooks;
    for (const auto& it : devices_) {
      const String& device_name = it.first;
      const tir::Var& context = it.second;
      Array<String> sections = {"Device", device_name, hook};
      String device_hook_name = ToCFunctionStyle(PrefixName(sections));

      tir::Evaluate device_hook(
          AddCheckReturn(tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                        {tvm::tir::StringImm(device_hook_name), context})));
      device_hooks.push_back(device_hook);
    }
    return tir::SeqStmt::Flatten(device_hooks);
  }

  /*!
   * \brief Generates a call to a given hook for a single Device function
   * \param context Device context to call hook on
   * \param hook Name of hook to generate statements for
   * \return Statement with function call to Device API
   */
  tir::Stmt GenerateDeviceHook(const tir::Var& context, const String& hook) {
    const auto& it = std::find_if(std::begin(devices_), std::end(devices_), [&](const auto& it) {
      return it.second->name_hint == context->name_hint;
    });
    const String& device_name = (*it).first;
    Array<String> sections = {"Device", device_name, hook};
    String device_hook = ToCFunctionStyle(PrefixName(sections));

    return tir::Evaluate(
        AddCheckReturn(tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                 {tvm::tir::StringImm(device_hook), context})));
  }

  /*!
   * \brief Utility function to string together different arguments
   */
  template <typename... Args>
  std::string MakeString(Args const&... args) {
    std::ostringstream ss;
    using List = int[];
    (void)List{0, ((void)(ss << args), 0)...};

    return ss.str();
  }

  /*!
   * \brief Access IO vars using the buffer vars and
   * not the actual var.
   */
  tir::Var GetBufferVarForIO(int index) { return main_buffer_map_[main_signature_[index]]->data; }

  /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   * \param expr The expression to evaluate.
   * \param original_name The name of the tir::Var.
   * \param use_unique_name Whether to generate a new unique name where a name conflicts.
   */
  void CreateIOVar(const Expr& expr, const std::string& original_name,
                   bool use_unique_name = true) {
    CreateIOVar(expr->checked_type(), original_name, use_unique_name);
  }

  /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   * \param expr The expression to evaluate.
   * \param original_name The name of the tir::Var.
   * \param use_unique_name Whether to generate a new unique name where a name conflicts.
   */
  void CreateIOVar(const Type& type, const std::string& original_name,
                   bool use_unique_name = true) {
    if (type->IsInstance<TupleTypeNode>()) {
      TupleType tuple_type = Downcast<TupleType>(type);
      for (unsigned i = 0; i < tuple_type->fields.size(); i++) {
        CreateIOVar(tuple_type->fields[i], original_name);
      }
    } else {
      std::string name = original_name;
      if (use_unique_name) {
        name = GetUniqueIOVarName(original_name);
      }
      tir::Var var = tir::Var(name, DataType::Handle());
      main_signature_.push_back(var);
      auto tensor_type = type.as<TensorTypeNode>();
      ICHECK(tensor_type) << "Expected TensorType node but was " << type->GetTypeKey();
      DataType elem_type = tensor_type->dtype;
      tir::Var buffer_var =
          tir::Var(name + "_buffer_var", PointerType(PrimType(elem_type), "global"));
      tir::Buffer buffer = tir::Buffer(buffer_var, elem_type, tensor_type->shape, {}, 0,
                                       name + "_buffer", 16, 1, tir::BufferType::kDefault);
      main_buffer_map_.Set(var, buffer);
    }
  }

  /*!
   * \brief Create a unique name for I/O Var
   */
  std::string GetUniqueIOVarName(std::string name) {
    if (io_var_names_.find(name) == io_var_names_.end()) {
      io_var_names_[name] = 1;
      return name + std::to_string(io_var_names_[name] - 1);
    } else {
      io_var_names_[name] = io_var_names_[name] + 1;
      return name + std::to_string(io_var_names_[name] - 1);
    }
  }

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
  /*! \brief All available targets. */
  CompilationConfig config_;
  /*!
   * \brief The type of kernel call to be emitted.
   * See CallType for more documentation.
   */
  CallType call_type_;
  std::unordered_map<const tir::Var, const ConstantNode*, ObjectPtrHash, ObjectPtrEqual>
      constant_map_;
  /*! \brief plan memory of device result */
  StorageMap expr_storage_map_;
  /*! \brief mapping sid -> tir::Var */
  std::unordered_map<int, tir::Var> sids_table_;
  /*! \brief the set of statements that make the program */
  std::vector<tir::Stmt> stmts_;
  /*! \brief the list of return sids (note that the function might return more then one output */
  std::vector<int> return_sid_;
  /*! \brief This is per IO var name counter to aid the generating unique names */
  std::unordered_map<std::string, int> io_var_names_;
  /*! \brief A set of variables that are let bound. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> let_bound_vars_;
};

Pass AOTLowerMain(String mod_name, tvm::CompilationConfig config, CallType call_type) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule module, transform::PassContext ctx) {
        return AOTMainLowerer(config, call_type).Lower(module, mod_name);
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "AOTLowerMain", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay.backend.aot.AOTLowerMain")
    .set_body_typed([](const String& mod_name, const tvm::CompilationConfig& config,
                       int call_type) {
      return AOTLowerMain(mod_name, config, static_cast<CallType>(call_type));
    });

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm
