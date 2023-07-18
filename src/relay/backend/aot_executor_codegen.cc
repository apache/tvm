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
#include <tvm/relay/runtime.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/name_transforms.h>
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
#include "../../tir/transforms/ir_utils.h"
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
  std::vector<TensorType> GetReturnTtypes() const { return return_ttypes_; }

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
      VisitExpr(arg);
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
    VisitExpr(func_node->body);
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
    VisitExpr(value);
    StorageInfo si = GetStorage(value);
    storage_device_map_[var] = si;
  }

 private:
  void AssignReturnSid(Expr e) {
    if (storage_device_map_.find(e) != storage_device_map_.end()) {
      StorageInfo& sinfo = storage_device_map_[e];
      return_ids_.clear();
      for (auto sid : sinfo->storage_ids) {
        return_ids_.push_back(sid);
      }
      return_ttypes_.clear();
      return_ttypes_ = FlattenTupleType(e->checked_type());
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
    ICHECK(it != storage_device_map_.end()) << "Could not find " << true_expr->GetTypeKey() << " "
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
  /*! \brief the data types of the return values */
  std::vector<TensorType> return_ttypes_;
};

/*! \brief Code generator for AOT executor */
class AOTExecutorCodegen : public MixedModeVisitor {
 protected:
  /*! \brief Describes the type of kernel call emitted. */
  enum CallType {
    /*!
     * \brief Emit PackedFunc calls bound just-in-time using TVMBackend* functions.
     *
     * When this type is selected, assumes all operators must be called via TVMFuncCall. Given the
     * implementation of TVMFuncCall in the C++ runtime, this in practice implies that those
     * functions are of type TVMBackendPackedCFunc.
     *
     * The following code is emitted at call sites to call a function named `func`:
     * void* func_ptr = TVMBackendGetFuncFromEnv("func");
     * TVMFuncCall(func_ptr, values, tcodes, num_args, ret_values, ret_tcodes)
     *
     * The arguments given to the tir::Call node are encoded into `values`, `tcodes`, and `num_args`
     * by LowerTVMBuiltin TIR transform.
     *
     * If `resource_handle` is passed to `func`, it is determined by TVMFuncCall (often,
     * `resource_handle` is registered with the C++ runtime to provide a `this` equivalent when
     * `func` is implemented in C).
     *
     * Compatible with both C++ and C runtimes, implemented with the C runtime only.
     */
    kPacked,  // Emit tir.call_packed and wrap all arguments in DLTensor.

    /*!
     * \brief Directly call a TVMBackendPackedCFunc named according to the tir::Call.
     *
     * When this type is selected, assumes all operators are implemented in functions of type
     * `TVMBackendPackedCFunc` and should be called directly. That is, presumes at the time of
     * downstream compilation that there is a symbol named after the 0th arg to tir::Call of
     * type `TVMBackendPackedCFunc`. This situation should occur when target_host == target.
     *
     * The following code is emitted at call sites to call a function named `func`:
     * func(values, tcodes, num_args, ret_values, ret_tcodes, resource_handle)
     *
     * The arguments given to the tir::Call node are encoded into `values`, `tcodes`, and `num_args`
     * by LowerTVMBuiltin TIR transform.
     *
     * `resource_handle` is encoded as the final argument to the tir::Call node. In practice, it is
     * always the device context parameter when not null. At present, the implementation does not
     * support forwarding device context parameters to CPacked.
     *
     * Compatible with the C runtime and C++ runtime (so long as target_host == target). Implemented
     * in the same scenarios.
     */
    kCPacked,  // Emit tir.call_cpacked and wrap all arguments in DLTensor.

    /*! \brief Directly call a function accepting the `data` arrays as args.
     *
     * When this type is selected, assumes all operaotrs are implemented in C functions whose
     * arguments are 1-to-1 with those in the tir::Call. DLTensor arguments are encoded as just the
     * `data` parameters (i.e. no DLTensor object is passed along).
     *
     * The following code is emitted at call sites to a function named `func`:
     * func(void* arg0, void* arg1, ..., void* argN) // no resource_handle
     * -or-
     * func(void* arg0, void* arg1, ..., void* argN, void* resource_handle) // with resource_handle
     *
     * `resource_handle` is encoded as the final argument to the tir::Call node. In practice, it is
     * always the device context parameter when not null.
     *
     * Compatible with the C runtime and C++ runtime (so long as target_host == target). Implemented
     * with the C runtime only.
     */
    kUnpacked,  // Emit tir.call_extern passing only the `data` part of DLTensors.
  };

  /*!
   * \brief Return a vector of variables that represents the sids for the given Relay Expr
   */
  std::vector<tir::Var> PackSid(Expr expr) {
    std::vector<tir::Var> buffer_vars;

    ICHECK(storage_device_map_.find(expr) != storage_device_map_.end())
        << "Storage map did not contain constant expr " << PrettyPrint(expr);
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
   * \brief Reverse lookup the device name in devices_ map.
   * \param device_context Value in devices_ to find.
   * \return Key matching device_context in devices_.
   */
  std::string FindDeviceName(tir::Var device_context) {
    for (std::pair<String, tir::Var> kv : devices_) {
      if (kv.second->name_hint == device_context->name_hint) {
        return kv.first;
      }
    }
    ICHECK(false) << "Did not find a device name associated with " << device_context;
    return "";
  }

  void PushArgs(const Expr& expr, const std::vector<tir::Var>& sids, Array<PrimExpr>* args) {
    const TupleNode* t = expr.as<TupleNode>();
    if (t != nullptr) {
      CHECK_EQ(sids.size(), t->fields.size()) << "Relay tuple does not map 1:1 into TIR; AOT can't "
                                                 "handle this type of Relay Expr in a CallNode.";
    }

    args->insert(args->end(), sids.begin(), sids.end());
  }

  /*
   * Wraps a call_extern with a tvm_check_return annotation if required otherwise
   * returns the passed Call
   */
  tir::Call AddCheckReturn(tir::Call existing_call) {
    Array<PrimExpr> args = {tir::make_const(DataType::Int(32, 1), 0, Span()),
                            tir::make_const(DataType::Int(32, 1), -1, Span()), existing_call};
    return tir::Call(DataType::Int(32), tir::builtin::tvm_check_return(), args);
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
        // NOTE: this cast looks like a no-op, but is required for compilation downstream.
        // Because DataType::Handle has default bits=64, but CodeGenC does not observe this field,
        // adding this cast forces the codegen to insert the cast. In this case, a cast is required
        // because param_handle is actually code-generated as `const void*`, and the `const` piece
        // needs to be removed.
        args.push_back(tvm::tir::Cast(DataType::Handle(32, 1), param_handle));
      } else {
        auto sids = FindExpr(arg);
        PushArgs(arg, sids, &args);
      }
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

          // call_extern has no further legalization steps, and
          // requires the number of arguments to match exactly.    For
          // internal calls, conditionally append the device context.
          bool requires_device_context = [&]() -> bool {
            Optional<Integer> opt = num_arguments_.Get(global_var);
            if (!opt.defined()) {
              // For external calls, we must trust that the user has
              // supplied a kernel that accepts a device_context
              // argument.
              return true;
            }
            int num_callee_params = opt.value()->value;
            int num_args = call_lowered_props.arguments.size();
            if (num_callee_params == num_args) {
              return false;
            } else if (num_callee_params == num_args + 1) {
              return true;
            } else {
              LOG(FATAL) << "Callee " << global_var << " requires " << num_callee_params
                         << ", but is called with " << num_args << " arguments.";
            }
          }();
          if (requires_device_context) {
            args.push_back(device_context);
          }
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
    std::vector<tir::Stmt> let_nest;

    // Define intermediate DLTensor to load/store the data
    tir::Buffer tmp_read =
        tir::decl_buffer({IntImm(DataType::UInt(64), size)}, DataType::UInt(8), "tmp_read");
    tir::Buffer tmp_write =
        tir::decl_buffer({IntImm(DataType::UInt(64), size)}, DataType::UInt(8), "tmp_write");

    // Re-use in/out as the buffer var, if possible
    if (auto opt = out.as<tir::Var>()) {
      tmp_write.CopyOnWrite()->data = opt.value();
    } else {
      let_nest.push_back(tir::LetStmt(tmp_write->data, out, tir::Evaluate(0)));
    }
    if (auto opt = in.as<tir::Var>()) {
      tmp_read.CopyOnWrite()->data = opt.value();
    } else {
      let_nest.push_back(tir::LetStmt(tmp_read->data, in, tir::Evaluate(0)));
    }

    // Copy the variable from the input to the output
    te::Var loop_idx("i", DataType::Int(32));
    tir::Stmt copy = tir::BufferStore(tmp_write, tir::BufferLoad(tmp_read, {loop_idx}), {loop_idx});
    copy = tir::For(loop_idx, 0, tir::make_const(DataType::Int(32, 1), size, Span()),
                    tir::ForKind::kSerial, copy);
    copy = tir::MergeNest(let_nest, copy);

    stmts_.push_back(copy);
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

      tir::Evaluate device_hook(
          AddCheckReturn(tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                        {tvm::tir::StringImm(device_hook_name), context})));
      device_hooks.push_back(device_hook);
    }
    return tir::SeqStmt::Flatten(device_hooks);
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

    return tir::Evaluate(
        AddCheckReturn(tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                 {tvm::tir::StringImm(device_hook), context})));
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
    StorageInfo& sinfo = storage_device_map_[expr];

    // Let bound vars refer to a value, so these should not be considered "output" vars.
    if (let_bound_vars_.find(GetRef<Var>(op)) != let_bound_vars_.end()) {
      return;
    }

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
    ICHECK(storage_device_map_.find(expr) != storage_device_map_.end())
        << "Storage map did not contain constant expr " << PrettyPrint(expr);
    StorageInfo& sinfo = storage_device_map_[expr];
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

  // Create the main PrimFunc to execute the graph. Please note that
  // the packed function calls don't pack their arguments. The AOT
  // runner function needs to be legalized by the LegalizePackedCalls pass.
  tir::PrimFunc CreateMainFunc(String mod_name, unsigned int relay_params) {
    tir::Stmt body = tir::SeqStmt::Flatten(stmts_);
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

    tir::Stmt device_activations = GenerateAllDeviceHook("Activate");
    tir::Stmt device_deactivations = GenerateAllDeviceHook("Deactivate");
    tir::Stmt final_body = tir::SeqStmt({device_activations, body, device_deactivations});

    // Make the PrimFunc
    return tir::PrimFunc(main_signature_, final_body, VoidType(), main_buffer_map_,
                         DictAttrs(dict_attrs));
  }

  /*!
   * \brief Access IO vars using the buffer vars and
   * not the actual var.
   */
  tir::Var GetBufferVarForIO(int index) { return main_buffer_map_[main_signature_[index]]->data; }

  /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   *
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
   *
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
      io_tensor_types_.Set(var, Downcast<TensorType>(type));
    }
  }

  /*!
   * \brief Create a unique name for I/O Var
   */
  std::string GetUniqueIOVarName(std::string name) {
    if (io_var_names_.find(name) == io_var_names_.end()) {
      io_var_names_[name] = 1;
      return name;
    } else {
      io_var_names_[name] = io_var_names_[name] + 1;
      return name + std::to_string(io_var_names_[name]);
    }
  }

  /*!
   * \brief Calculate workspace sizes for PrimFuncs in the IRModule
   */
  Map<String, FunctionInfo> CalculateWorkspaceSizes(
      const IRModule& lowered_mod, const Map<String, FunctionInfo>& function_metadata) {
    Integer workspace_byte_alignment = GetModuleWorkspaceByteAlignment(lowered_mod);
    Map<String, FunctionInfo> updated_function_metadata;
    for (const auto& kv : lowered_mod->functions) {
      GlobalVar global_var = kv.first;
      BaseFunc base_func = kv.second;
      if (base_func->IsInstance<tir::PrimFuncNode>()) {
        tir::PrimFunc pfunc = Downcast<tir::PrimFunc>(base_func);
        Target tgt = pfunc->GetAttr<Target>(tvm::attr::kTarget).value();
        const auto& ws = CalculateWorkspaceBytes(pfunc, workspace_byte_alignment);
        if (function_metadata.count(global_var->name_hint)) {
          updated_function_metadata.Set(global_var->name_hint,
                                        function_metadata[global_var->name_hint]);
          updated_function_metadata[global_var->name_hint]->workspace_sizes.Set(tgt, ws);
        } else {
          FunctionInfo finfo{{{tgt, ws}}, {}, {}, {{tgt, pfunc}}, {}};
          updated_function_metadata.Set(global_var->name_hint, finfo);
        }
      }
    }
    return updated_function_metadata;
  }

  /*!
   * \brief Run USMP to plan memory for lowered IRModule.
   */
  IRModule PlanMemoryWithUSMP(const IRModule& mod) {
    VLOG(1) << "Planning memory with USMP for module:" << std::endl << PrettyPrint(mod);
    Integer workspace_byte_alignment = GetModuleWorkspaceByteAlignment(mod);
    IRModule lowered_mod = mod->ShallowCopy();
    lowered_mod = tir::transform::UnifiedStaticMemoryPlanner()(lowered_mod);
    function_metadata_ = CalculateWorkspaceSizes(lowered_mod, function_metadata_);
    Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
        lowered_mod->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
    backend::FunctionInfo main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info").value();
    main_func_info->workspace_sizes.clear();
    if (allocated_pool_infos) {
      for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
        for (const auto& tgt : allocated_pool_info->pool_info->targets) {
          VLOG(1) << "USMP requires target " << tgt->ToDebugString() << " to have pool size "
                  << allocated_pool_info->allocated_size->value;
          size_t size = allocated_pool_info->allocated_size->value;
          if (allocated_pool_info->pool_info->IsInstance<ConstantPoolInfoNode>()) {
            size += main_func_info->constant_sizes.count(tgt)
                        ? main_func_info->constant_sizes[tgt]->value
                        : 0;
            main_func_info->constant_sizes.Set(tgt, size);
          } else if (allocated_pool_info->pool_info->IsInstance<WorkspacePoolInfoNode>()) {
            size += main_func_info->workspace_sizes.count(tgt)
                        ? main_func_info->workspace_sizes[tgt]->value
                        : 0;
            main_func_info->workspace_sizes.Set(tgt, size);
          } else {
            LOG(FATAL) << "Unknown pool type: " << allocated_pool_info->pool_info->GetTypeKey();
          }
        }
      }
    }
    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info);
    return lowered_mod;
  }

  /*!
   * \brief Run StorageRewrite to plan memory for lowered IRModule.
   */
  IRModule PlanMemoryWithStorageRewrite(const IRModule& mod) {
    Integer workspace_byte_alignment = GetModuleWorkspaceByteAlignment(mod);
    IRModule lowered_mod = mod->ShallowCopy();
    function_metadata_ = CalculateWorkspaceSizes(lowered_mod, function_metadata_);
    // Running StorageRewrite just on the main function
    tir::PrimFunc tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
    IRModule main_func_mod;
    main_func_mod->Update(lowered_mod->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main),
                          tir_main_func);
    main_func_mod = tir::transform::StorageRewrite()(main_func_mod);
    lowered_mod->Update(lowered_mod->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main),
                        main_func_mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
    tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
    // Use the PrimFunc to calculate the workspace required to service the allocates
    Integer main_workspace_size_bytes =
        CalculateWorkspaceBytes(tir_main_func, workspace_byte_alignment);
    backend::FunctionInfo main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info").value();
    main_func_info->workspace_sizes.Set(config_->host_target, main_workspace_size_bytes);
    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info);
    return lowered_mod;
  }

  /*!
   * \brief Gets module workspace alignment from supplied executor or defaults to 16
   */
  Integer GetModuleWorkspaceByteAlignment(const IRModule& mod) {
    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    return executor_config->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
  }

  /*!
   * \brief Gets module constant alignment from supplied executor or defaults to 16
   */
  Integer GetModuleConstantByteAlignment(const IRModule& mod) {
    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    return executor_config->GetAttr<Integer>("constant-byte-alignment").value_or(16);
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
  /*! \brief map of GlobalVars to the number of arguments they require */
  Map<GlobalVar, Integer> num_arguments_;
  /*! \brief input and output variables belonging to the main function signature */
  Array<tir::Var> main_signature_;
  /*! \brief input and output variables belonging to the main function signature */
  Map<tir::Var, tir::Buffer> main_buffer_map_;
  /*! \brief maps input and output variables to TensorType which describe them */
  Map<tir::Var, TensorType> io_tensor_types_;
  /*! \brief All available targets. */
  CompilationConfig config_;
  /*!
   * \brief The type of kernel call to be emitted.
   * See CallType for more documentation.
   */
  CallType call_type_;

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
  std::unordered_map<const tir::Var, const ConstantNode*, ObjectPtrHash, ObjectPtrEqual>
      constant_map_;

  /*! \brief plan memory of device result */
  StorageMap storage_device_map_;
  /*! \brief mapping sid -> tir::Var */
  std::unordered_map<int, tir::Var> sids_table_;
  /*! \brief lowered funcs */
  Map<String, FunctionInfo> function_metadata_;
  /*! \brief the set of statements that make the program */
  std::vector<tir::Stmt> stmts_;
  /*! \brief the list of return sids (note that the function might return more then one output */
  std::vector<int> return_sid_;
  /*! \brief This is per IO var name counter to aid the generating unique names */
  std::unordered_map<std::string, int> io_var_names_;
  /*! \brief A set of variables that are let bound. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> let_bound_vars_;

 public:
  AOTExecutorCodegen(runtime::Module* mod, const Array<Target>& targets)
      : mod_(mod), config_(transform::PassContext::Current(), targets) {}

  LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    VLOG_CONTEXT << "AOT";

    Runtime runtime_config = mod->GetAttr<Runtime>(tvm::attr::kRuntime).value();
    Integer workspace_byte_alignment = GetModuleWorkspaceByteAlignment(mod);

    Executor executor_config = mod->GetAttr<Executor>(tvm::attr::kExecutor).value();
    std::string interface_api =
        executor_config->GetAttr<String>("interface-api").value_or("packed");
    bool unpacked_api = executor_config->GetAttr<Bool>("unpacked-api").value_or(Bool(false));

    // Validate choice of unpacked_api and use_call_cpacked_
    if (runtime_config->name == kTvmRuntimeCrt) {
      if (unpacked_api == true) {
        call_type_ = CallType::kUnpacked;
      } else if (unpacked_api == false && interface_api == "packed") {
        call_type_ = CallType::kCPacked;
      } else {
        CHECK(interface_api == "packed" || unpacked_api == true)
            << "Either need interface_api == \"packed\" (got: " << interface_api
            << ") or unpacked-api == true (got: " << unpacked_api << ") when targeting c runtime";
        ICHECK(false) << "Unhandled executor option config: interface-api=" << interface_api
                      << ", unpacked-api=" << unpacked_api;
      }
    } else if (runtime_config->name == kTvmRuntimeCpp) {
      if (unpacked_api == false && interface_api == "packed") {
        call_type_ = CallType::kCPacked;
      } else {
        CHECK(static_cast<bool>(unpacked_api) == false && interface_api == "packed")
            << "Need unpacked-api == false (got: " << unpacked_api
            << ") and interface-api == \"packed\" (got: " << interface_api
            << ") when targeting c++ runtime";
        ICHECK(false) << "Unhandled executor option config: interface-api=" << interface_api
                      << ", unpacked-api=" << unpacked_api;
      }
    } else {
      ICHECK(false) << "runtime_config (" << runtime_config->name
                    << ") is not one of the expected values";
    }

    mod = transform::ToANormalForm()(mod);
    mod = transform::InferType()(mod);
    mod = transform::AnnotateUsedMemory()(mod);

    IRModule lowered_mod =
        tec::LowerTE(mod_name, config_, [this, workspace_byte_alignment](BaseFunc func) {
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
        })(mod);

    transform::PassContext pass_ctx = transform::PassContext::Current();
    bool enable_remove_reshapes =
        pass_ctx->GetConfig<Bool>("relay.remove_standalone_reshapes.enable", Bool(true)).value();
    if (enable_remove_reshapes) {
      lowered_mod = transform::RemoveStandaloneReshapes()(lowered_mod);
    }
    auto lowered_main = lowered_mod->Lookup("main");
    auto lowered_main_func = Downcast<Function>(lowered_main);

    // Post-lowering storage map for writing main func
    AOTOnDemandAllocator final_aot_allocator;
    final_aot_allocator.Run(lowered_main_func);
    storage_device_map_ = final_aot_allocator.GetStorageMap();

    // TODO(@electriclilies, @jroesch, @Mousius): remove UpdateMainWorkspaceSize
    StaticMemoryPlan memory_plan(storage_device_map_);
    backend::FunctionInfo func_info =
        tec::UpdateMainWorkspaceSize(lowered_mod, config_, memory_plan->expr_to_storage_info);
    lowered_mod = WithAttr(lowered_mod, "main_func_info", func_info);

    for (auto input : lowered_main_func->params) {
      input_vars_.push_back(input);
      std::string input_name = SanitizeName(input->name_hint());
      // We dont want the compiler changing input names in the
      // event of a sanitization collision. Therefore, enforcing
      // the var created to use the input_name strictly.
      CreateIOVar(input, input_name, /*use_unique_name = */ false);
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
    // If output tensor names were provided use them
    if (auto opt = func->GetAttr<Array<String>>("output_tensor_names")) {
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
      CreateIOVar(lowered_main_func->body, "output");
    }

    CollectDeviceVariables(lowered_mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value());
    num_arguments_ = [&]() -> Map<GlobalVar, Integer> {
      Map<GlobalVar, Integer> arg_count;
      for (const auto& [gvar, func] : lowered_mod->functions) {
        if (const auto* prim_func = func.as<tir::PrimFuncNode>()) {
          arg_count.Set(gvar, prim_func->params.size());
        }
      }
      return arg_count;
    }();
    VisitExpr(lowered_main_func->body);

    // Create the runner function. Please note that the function is not legal yet
    // because the packed calls arguments are not wrapped in TVMValues. To make this happen we need
    // to run the LegalizePackedCalls pass.
    LoweredOutput ret;

    // Collect any constants extracted by external codegen.
    ret.params = std::unordered_map<std::string, tvm::runtime::NDArray>();
    Map<String, runtime::NDArray> const_name_to_constant =
        lowered_mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant)
            .value_or({});
    for (const auto& kv : const_name_to_constant) {
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    // Collect any constants extracted during lowering.
    for (const auto& kv : params_) {
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    // AoT Executor codegen works completely on TIR beyond this point, hence removing relay main
    // function and replacing it with its TIR version. We should try to make this a Pass.
    lowered_mod->Remove(lowered_mod->GetGlobalVar("main"));
    auto tir_main_func = CreateMainFunc(mod_name, lowered_main_func->params.size());
    // Extract additional information around main TIR PrimFunc arguments
    Array<String> devices = ListDevices();
    const auto main_func_params_end_iterator =
        tir_main_func->params.begin() + tir_main_func->params.size();
    const auto outputs_begin_iterator =
        main_func_params_end_iterator - return_sid_.size() - devices.size();
    Array<tir::Var> inputs = Array<tir::Var>(tir_main_func->params.begin(), outputs_begin_iterator);
    Array<TensorType> input_tensor_types;
    for (auto i : inputs) {
      input_tensor_types.push_back(io_tensor_types_[i]);
    }
    Array<tir::Var> outputs =
        Array<tir::Var>(outputs_begin_iterator, main_func_params_end_iterator - devices.size());

    lowered_mod->Update(GlobalVar(::tvm::runtime::symbol::tvm_module_main), tir_main_func);
    // Parallel for loops are not supported in AoT codegen.
    lowered_mod = tir::transform::ConvertForLoopsToSerial()(lowered_mod);

    // Check USMP option
    bool enable_usmp = false;
    if (runtime_config->name == kTvmRuntimeCrt) {
      enable_usmp = true;
    }
    if (pass_ctx->GetConfig<Bool>(kUSMPEnableOption) != nullptr) {
      enable_usmp = pass_ctx->GetConfig<Bool>(kUSMPEnableOption, Bool(false)).value();
    }

    if (enable_usmp) {
      lowered_mod = PlanMemoryWithUSMP(lowered_mod);
    } else {
      lowered_mod = PlanMemoryWithStorageRewrite(lowered_mod);
    }
    ret.function_metadata = std::move(function_metadata_);

    // Legalize AOT if needed. This means that all the packed calls
    // need to be wrapped in TVMValues (unless unpacked_api is set)
    if (call_type_ == CallType::kCPacked || call_type_ == CallType::kPacked) {
      auto pack_calls = tir::transform::LegalizePackedCalls();
      lowered_mod = pack_calls(lowered_mod);
    }

    // Collect any runtime modules generated by external codegen.
    ret.external_mods =
        lowered_mod->GetAttr<Array<tvm::runtime::Module>>(tvm::attr::kExternalMods).value_or({});

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

    // Extract USMP metadata to pass onto metadata sources
    Map<tir::Var, tir::usmp::AllocatedPoolInfo> pool_var_info;
    std::vector<tir::Var> pool_vars;
    tir_main_func =
        Downcast<tir::PrimFunc>(lowered_mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
    Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
        tir_main_func->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
    if (allocated_pool_infos) {
      for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
        int pool_var_index = allocated_pool_info->pool_var_idx.value()->value;
        pool_vars.push_back(tir_main_func->params[pool_var_index]);
        pool_var_info.Set(tir_main_func->params[pool_var_index], allocated_pool_info);
      }
    }
    Map<String, tir::usmp::PoolAllocation> io_pool_allocations =
        lowered_mod
            ->GetAttr<Map<String, tir::usmp::PoolAllocation>>(tvm::attr::kIOTensorPoolAllocations)
            .value_or({});

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

    Array<TensorType> output_tensor_types{final_aot_allocator.GetReturnTtypes()};

    ret.metadata = ExecutorCodegenMetadata(
        inputs, input_tensor_types, output_var_names, output_tensor_types, pool_vars, devices,
        runtime::kTvmExecutorAot, mod_name, interface_api, unpacked_api,
        GetModuleWorkspaceByteAlignment(mod), GetModuleConstantByteAlignment(mod), pool_var_info,
        io_pool_allocations);
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
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and Array<Target> targets";
        void* mod = args[0];
        Array<Target> targets = args[1];
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

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kRunnable; }

 private:
  void init(void* mod, const Array<Target>& targets) {
    codegen_ =
        std::make_shared<AOTExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod), targets);
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
    return (*it).second;
  }

  Array<tvm::runtime::Module> get_external_modules() { return output_.external_mods; }

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
