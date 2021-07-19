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

#include "te_compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../transforms/pass_utils.h"
#include "te_compiler.h"
#include "te_compiler_cache.h"
#include "utils.h"

namespace tvm {
namespace relay {
// TODO(@jroesch, @csullivan): declare directly elsewhere
backend::StaticMemoryPlan GraphPlanMemory(const Function& func);

namespace tec {

using namespace tvm::relay::transform;

TVM_REGISTER_OBJECT_TYPE(TECompilerNode);

class TECompilerImpl : public TECompilerNode {
 public:
  // Lower the function.
  CachedFunc Lower(const CCacheKey& key, std::function<String(String)> mangle_fn) {
    return LowerInternal(key, mangle_fn)->cached_func;
  }

  CachedFunc Lower(const CCacheKey& key, const String mod_name) {
    auto mangle_fn = [mod_name](String name) { return runtime::get_name_mangled(mod_name, name); };

    return Lower(key, mangle_fn);
  }

  // For now, build one module per function.
  PackedFunc JIT(const CCacheKey& key) final {
    auto mangle_fn = [](String name) { return name; };
    CCacheValue value = LowerInternal(key, mangle_fn);
    if (value->packed_func != nullptr) {
      return value->packed_func;
    }
    auto m = build(value->cached_func->funcs, key->target, Target(nullptr));
    value->packed_func = m.GetFunction(value->cached_func->prim_fn_var->name_hint);
    return value->packed_func;
  }

  CachedFunc LowerShapeFunc(const CCacheKey& key) final {
    return LowerShapeFuncInternal(key)->cached_func;
  }

  Map<String, IRModule> GetLoweredFunctions() {
    Map<String, IRModule> lowered_functions;
    for (const auto& it : cache_) {
      auto source_func = it.first;
      auto lowered_func = it.second;
      auto target = source_func->target;

      if (!lowered_functions.count(target->str())) {
        lowered_functions.Set(target->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }

      lowered_functions[target->str()]->Update(lowered_func->cached_func->funcs);
    }
    return lowered_functions;
  }

  Array<tvm::runtime::Module> LowerExternalFunctions() {
    Array<tvm::runtime::Module> ret;
    std::unordered_map<std::string, std::string> cached_symbol;
    std::vector<CCacheKey> cached_ext_funcs;
    for (const auto& it : cache_) {
      auto src_func = it.first->source_func;
      ICHECK(src_func.defined());
      if (src_func->GetAttr<String>(attr::kCompiler).defined()) {
        auto code_gen = src_func->GetAttr<String>(attr::kCompiler);
        std::string code_gen_name = code_gen.value();
        cached_ext_funcs.push_back(it.first);

        auto symbol_name = src_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        ICHECK(symbol_name.defined()) << "No external symbol is set for:\n"
                                      << AsText(src_func, false);

        std::string sn = symbol_name.value();
        if (cached_symbol.count(sn)) {
          cached_symbol[sn] = code_gen_name;
        } else {
          ICHECK_NE(sn, code_gen_name)
              << "Found duplicated symbol: " << sn << " for: " << code_gen_name;
        }

        std::string ext_name = "relay.ext." + code_gen_name;
        auto pf = tvm::runtime::Registry::Get(ext_name);
        ICHECK(pf) << "Failed to find the codegen tool for " << ext_name;
        // No need to keep compiler attribute at this point, functions have been
        // extracted for specific codegen.
        src_func = WithAttr(std::move(src_func), attr::kCompiler, NullValue<ObjectRef>());
        runtime::Module ext_mod = (*pf)(src_func);

        ICHECK(ext_mod.defined()) << "No external runtime is generated.";
        ret.push_back(ext_mod);
      }
    }

    // No need to cache external functions as we collected them all to create
    // external runtime modules.
    for (const auto& it : cached_ext_funcs) {
      cache_.erase(it);
    }
    return ret;
  }

  void Clear() final { cache_.clear(); }

  // List all items in the cache.
  Array<ObjectRef> ListItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<ObjectRef> items;
    for (auto& kv : cache_) {
      items.push_back(kv.first);
      items.push_back(kv.second);
    }
    return items;
  }

  /*!
   * \brief Get the cache key of the function that is being lowered currently
   * \return the cache key
   */
  CCacheKey GetCurrentCCacheKey() { return cur_ccache_key_; }

 private:
  // implement lowered func
  CCacheValue LowerInternal(const CCacheKey& key, std::function<String(String)> mangle_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 1;
      cache_[key] = value;
    }
    cur_ccache_key_ = key;

    // No need to lower external functions for now. We will invoke the external
    // codegen tool once and lower all functions together.
    if (key->source_func->GetAttr<String>(attr::kCompiler).defined()) {
      auto ir_module = IRModule();
      const auto name_node = key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(name_node.defined()) << "External function has not been attached a name yet.";
      auto func_name = GetUniqueName(name_node.value(), &name_map_);
      auto target = Target("ext_dev");
      auto global_var = GlobalVar(func_name);
      global_var->checked_type_ = key->source_func->checked_type();
      value->cached_func = CachedFunc(target, global_var, {}, {}, te::Schedule(), {}, ir_module);
      return value;
    }

    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    auto cfunc = PrimFuncFor(key->source_func, key->target, [&](std::string name) {
      auto mangled = mangle_fn(name);
      return GetUniqueName(mangled, &name_map_);
    });

    // Skip lowering for device copy node.
    const Expr body = (key->source_func)->body;
    if (const CallNode* call_node = body.as<CallNode>()) {
      if (call_node->attrs.as<DeviceCopyAttrs>()) {
        value->cached_func = cfunc;
        return value;
      }
    }

    // NOTE: array will copy on write.
    Array<te::Tensor> all_args = Array<te::Tensor>(cfunc->inputs);
    for (te::Tensor arg : cfunc->outputs) {
      all_args.push_back(arg);
    }

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    auto func_name = cfunc->prim_fn_var->name_hint;
    cfunc->funcs->Update(tvm::LowerSchedule(cfunc->schedule, all_args, func_name, binds));
    value->cached_func = cfunc;
    return value;
  }

  // implement lowered shape func
  CCacheValue LowerShapeFuncInternal(const CCacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = shape_func_cache_.find(key);
    if (it != shape_func_cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 0;
      shape_func_cache_[key] = value;
    }
    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());

    using tvm::transform::PassContext;
    With<PassContext> fresh_pass_ctx_scope(PassContext::Create());
    auto cached_func = ShapeFuncFor(key->source_func, key->target, [&](std::string name) {
      return GetUniqueName(name, &name_map_);
    });

    value->cached_func = cached_func;
    return value;
  }

  std::unordered_map<std::string, int> GetOpWeights() {
    std::unordered_map<std::string, int> weights;
    for (auto pair : cache_) {
      auto value = pair.second;
      auto name = value->cached_func->prim_fn_var->name_hint;
      weights[name] = value->use_count;
    }
    return weights;
  }

  /*! \brief compiler cache lock*/
  std::mutex mutex_;
  /*! \brief internal name map to get an unique name */
  std::unordered_map<std::string, int> name_map_;
  /*! \brief internal compiler cache */
  std::unordered_map<CCacheKey, CCacheValue> cache_;
  /*! \brief internal compiler cache for shape funcs */
  std::unordered_map<CCacheKey, CCacheValue> shape_func_cache_;
  /*! \brief the cache key of the function that is being lowered currently*/
  CCacheKey cur_ccache_key_;
};

TECompiler::TECompiler() {
  auto object = make_object<TECompilerImpl>();
  data_ = object;
}

using AnalysisRemapping = std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual>;

std::tuple<bool, int, int> IsDeviceCopy(const Function& func) {
  if (auto call_node = func->body.as<CallNode>()) {
    if (auto op_node = call_node->op.as<OpNode>()) {
      if (op_node->name == "device_copy") {
        auto attrs = call_node->attrs.as<DeviceCopyAttrs>();
        auto dst = attrs->dst_dev_type;
        auto src = attrs->src_dev_type;
        return std::tuple<bool, int, int>(true, src, dst);
      }
    }
  }

  return std::tuple<bool, int, int>(false, -1, -1);
}

class LowerTensorExpr : public ExprMutator {
 public:
  LowerTensorExpr(const IRModule& module, const TargetMap& targets, const DeviceMap& device_ctx_map,
                  ProcessFn process_fn, const String& module_name, TECompiler compiler)
      : module_(module),
        targets_(targets),
        device_context_map_(device_ctx_map),
        process_fn(process_fn),
        module_name_(module_name),
        compiler_(compiler) {}

  Expr VisitExpr_(const CallNode* call) override {
    Call expr = GetRef<Call>(call);
    Function func;

    if (call->op.as<FunctionNode>()) {
      func = GetRef<Function>(call->op.as<FunctionNode>());
    } else {
      return ExprMutator::VisitExpr_(call);
    }

    if (!func->HasNonzeroAttr(attr::kPrimitive)) {
      // Provide a callback hook which allows one-level up code generators to
      // act when we process a function.
      this->process_fn(func);
      return ExprMutator::VisitExpr_(call);
    }

    // Process inputs.
    Array<Expr> args;
    for (size_t i = 0; i < expr->args.size(); i++) {
      args.push_back(VisitExpr(expr->args[i]));
    }

    Target target;

    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      target = Target("ext_dev");
      CCacheKey key = CCacheKey(func, target);
      CachedFunc ext_func = compiler_->Lower(key, module_name_);
      ICHECK(ext_func.defined()) << "Lowering returned undefined function for "
                                 << ext_func->prim_fn_var->name_hint;

      Map<GlobalVar, tir::PrimFunc> prim_fns;

      for (auto prim_fn : ext_func->funcs->functions) {
        CHECK(prim_fn.second.as<tir::PrimFuncNode>()) << "must be a prim fn";
        prim_fns.Set(prim_fn.first, Downcast<tir::PrimFunc>(prim_fn.second));
      }

      relay::Function func_with_metadata = func;
      func_with_metadata = WithAttr(func_with_metadata, "prim_fn_var", ext_func->prim_fn_var);
      func_with_metadata = WithAttr(func_with_metadata, "prim_funcs", prim_fns);
      func_with_metadata = WithAttr(func_with_metadata, "target", ext_func->target);

      // Provide a callback hook which allows one-level up code generators to
      // act when we process a function.
      this->process_fn(func_with_metadata);

      auto ret_call = Call(ext_func->prim_fn_var, args, {});
      return std::move(ret_call);
    }

    ICHECK_GE(device_context_map_.count(expr), 0)
        << "Could not find an entry in the device context map for " << PrettyPrint(expr)
        << "The memory planning was either not performed for this precise node, or there is bug "
           "in the memory planner.";

    auto& device_context = this->device_context_map_[expr];
    auto call_dev_type = device_context.device_type;

    // Non-External Relay Function
    if (targets_.size() == 1) {
      // The homogeneous execution case, we should only have one target
      // so we just grab it.
      const auto& it = targets_.begin();
      target = (*it).second;
    } else {
      // The heterogeneous execution case we have multiple targets
      // in this case.
      //
      // We need to identify the target and translate.
      std::string call_dev_name;
      if (call_dev_type == 0) {
        call_dev_name = "llvm";
        call_dev_type = kDLCPU;
      } else {
        call_dev_name = ::tvm::runtime::DeviceName(call_dev_type);
      }

      if (targets_.count(call_dev_type) == 0) {
        std::stringstream msg;
        msg << "No target is specified for provided device name: `" << call_dev_name << "`\n\n";
        msg << call_dev_name << " mapped to device type (" << call_dev_type
            << ") which was not found in the target map.\n";
        msg << "Availible targets: \n";
        for (auto target : targets_) {
          msg << "  " << target.first << "-> " << target.second << "\n";
        }
        LOG(FATAL) << msg.str();
      }

      target = targets_[call_dev_type];
    }

    CCacheKey key = CCacheKey(func, target);
    CachedFunc lowered_func = compiler_->Lower(key, module_name_);

    Map<GlobalVar, tir::PrimFunc> prim_fns;

    for (auto prim_fn : lowered_func->funcs->functions) {
      CHECK(prim_fn.second.as<tir::PrimFuncNode>()) << "must be a prim fn";
      prim_fns.Set(prim_fn.first, Downcast<tir::PrimFunc>(prim_fn.second));
    }

    // TODO(@areusch, @jroesch): this metadata is for AOT, this should be our interface for AOT
    relay::Function func_with_metadata = func;
    func_with_metadata = WithAttr(func_with_metadata, "prim_fn_var", lowered_func->prim_fn_var);
    func_with_metadata = WithAttr(func_with_metadata, "prim_funcs", prim_fns);
    func_with_metadata = WithAttr(func_with_metadata, "target", lowered_func->target);

    // Provide a callback hook which allows one-level up code generators to
    // act when we process a function.
    this->process_fn(func_with_metadata);

    auto tir_call_attrs = make_object<TIRCallAttrs>();
    if (func->HasNonzeroAttr(attr::kReshapeOnly)) {
      tir_call_attrs->metadata.Set(attr::kReshapeOnly, tvm::Integer(1));
    }

    auto device_copy = IsDeviceCopy(func);
    if (std::get<0>(device_copy)) {
      auto source_device = std::get<1>(device_copy);
      auto dst_device = std::get<2>(device_copy);
      tir_call_attrs->metadata.Set("source_device", tvm::Integer(source_device));
      tir_call_attrs->metadata.Set("dst_device", tvm::Integer(dst_device));
    }

    tir_call_attrs->metadata.Set("relay_attrs", func->attrs);

    Expr ret_call = Call(lowered_func->prim_fn_var, args, Attrs(tir_call_attrs));
    return std::move(ret_call);
  }

  IRModule module_;
  TargetMap targets_;
  DeviceMap device_context_map_;
  ProcessFn process_fn;
  String module_name_;
  TECompiler compiler_;
};

/*!
 * \brief Obtain the Target from the device type.
 * If homogenous compilation, this will return the only target.
 * If heteregenous compilation, this will select associated using the targets_ Map.
 *
 * \param dev_type
 * \return Target
 */
Target GetTargetFromInteger(DLDeviceType dev_type, TargetMap targets) {
  if (targets.size() == 1) {
    // homogeneous execution.
    const auto& it = targets.begin();
    return (*it).second;
  } else {
    // heterogeneous execution.
    std::string call_dev_name;
    if (dev_type == 0) {
      call_dev_name = "llvm";
    } else {
      call_dev_name = runtime::DeviceName(dev_type);
    }
    if (targets.count(dev_type) == 0) {
      LOG(FATAL) << "No target is provided for device " << call_dev_name;
    }
    return targets[dev_type];
  }
}

/*!
 * \brief Update the "main" control function's metadata
 *
 * \param mod The module
 * \param targets Map of targets
 * \return function_infos Function info for each function in the module
 */

backend::FunctionInfo UpdateMainWorkspaceSize(const IRModule& mod, TargetMap targets,
                                              Map<Expr, backend::StorageInfo> storage_info_map) {
  CHECK_EQ(mod->functions.size(), 1)
      << "There should only be one function in the module passed to UpdateMainWorkspaceSize";
  Function func = Downcast<Function>(mod->Lookup("main"));

  // This is a Map<device,Map<storage_id, size>>
  std::unordered_map<DLDeviceType, std::unordered_map<int, int>, EnumClassHash> sid_workspace;
  // This is a Map<device, size_of_inputs_and_outputs>
  std::unordered_map<DLDeviceType, int, EnumClassHash> device_io;
  // This is a Map<device, size_of_constants>
  std::unordered_map<DLDeviceType, int, EnumClassHash> device_consts;

  // Initialize the mapping from all storage identifiers to workspace sizes,
  // the amount of device io, and the device constants.
  for (const auto& kv : storage_info_map) {
    backend::StorageInfo storage_info = kv.second;
    std::vector<int64_t> storage_ids = storage_info->storage_ids;
    std::vector<DLDeviceType> devices = storage_info->device_types;

    CHECK_EQ(storage_ids.size(), devices.size());
    for (uint32_t i = 0; i < devices.size(); i++) {
      sid_workspace[devices[i]][storage_ids[i]] = 0;
      device_io[devices[i]] = 0;
      device_consts[devices[i]] = 0;
    }
  }

  // Iterate the storage map to compute all the tensor sizes in the program.
  // There are 3 cases in this code:
  //
  // First we need to compute the sizes of all
  // inline constants.
  //
  // Second we compute the size of any bound variable as these are input and output
  // sizes of the program.
  //
  // Finally for all other expressions we check which storage identifier they have
  // been assigned and we compute the maximal size of the storage, as tensors can
  // share storage with other tensors which are the same size or larger.
  //
  // In this final case there is only one allocation for all tensors which share storage
  // which will be the maximal size of all tensors which were assigned to it.
  for (const auto& kv : storage_info_map) {
    Expr expr = kv.first;
    int64_t size_bytes = backend::CalculateRelayExprSizeBytes(expr->checked_type());
    backend::StorageInfo storage_info = kv.second;
    std::vector<int64_t> storage_ids = storage_info->storage_ids;
    std::vector<DLDeviceType> devices = storage_info->device_types;

    if (expr->IsInstance<ConstantNode>()) {
      for (const auto& dev : devices) {
        device_consts[dev] += size_bytes;
      }
      continue;
    } else if (expr->IsInstance<VarNode>() || expr.same_as(func->body)) {
      CHECK_GE(devices.size(), 1) << "must be at least one device";
      for (const auto& dev : devices) {
        device_io[dev] += size_bytes;
      }
      continue;
    }

    // TODO(@electriclilies): This code is never being called which means sid_workspace is not
    // updated.. This means that storage info is probably not being created correctly. Or is not
    // equivalent to what was here previously
    for (uint32_t i = 0; i < storage_ids.size(); i++) {
      // Here we record the largest size of the tensor
      // that share the same storage id, because storage_id will
      // be shared between multiple tensors that are not live simultaneously.
      if (size_bytes > sid_workspace[devices[i]][storage_ids[i]]) {
        sid_workspace[devices[i]][storage_ids[i]] = size_bytes;
      }
    }
  }

  // This is a Map<device, workspace_size>
  std::unordered_map<DLDeviceType, int, EnumClassHash> device_workspace;
  // Once we know the sizes of sids, we need to accumulate per device
  for (const auto& dev_sid_size : sid_workspace) {
    auto dev = dev_sid_size.first;
    device_workspace[dev] = 0;
    for (const auto& sid_size : dev_sid_size.second) {
      device_workspace[dev] += sid_size.second;
    }
  }

  Map<Target, Integer> workspace_sizes;
  Map<Target, Integer> io_sizes;
  Map<Target, Integer> constant_sizes;
  Map<Target, tir::PrimFunc> tir_primfuncs;
  Map<Target, Function> relay_primfuncs;

  // Initialize all target workspaces to zero
  for (const auto& kv : targets) {
    auto tgt = kv.second;
    workspace_sizes.Set(tgt, 0);
  }

  for (const auto& dev_and_size : device_workspace) {
    auto tgt = GetTargetFromInteger(dev_and_size.first, targets);
    workspace_sizes.Set(tgt, dev_and_size.second);
    relay_primfuncs.Set(tgt, func);
  }
  for (const auto& dev_and_size : device_io) {
    auto tgt = GetTargetFromInteger(dev_and_size.first, targets);
    io_sizes.Set(tgt, dev_and_size.second);
  }

  for (const auto& dev_and_size : device_consts) {
    auto tgt = GetTargetFromInteger(dev_and_size.first, targets);
    constant_sizes.Set(tgt, dev_and_size.second);
  }

  return backend::FunctionInfo(workspace_sizes, io_sizes, constant_sizes, tir_primfuncs,
                               relay_primfuncs);
}

// TODO(@electriclilies): Is the function passed in here relay_func??
// Also should this be inlined?
/*!
 * \brief A function to create the function metadata for an input function (ie calculate buffer
 * input/output sizes)
 * \param relay_func The function to calculate function metadata for
 * \param function_metadata The map that stores all the function metadatas
 */
void UpdateFunctionMetadata(Function relay_func,
                            Map<String, backend::FunctionInfo>& function_metadata) {  // NOLINT(*)
  // Originally UpdateFunctionMetadata took in CCachedFunc and looped through all the funcs stored
  // there Now the goal is to take only one func because process_fn should be controlling the
  // iteration However, to do the workspace calculations we need the primfuncs. So process_fn needs
  // to either access the cached funcs or be directly passed primfuncs This is bad and ideally we
  // don't want process_fn to look at primfuncs There's also the question now of what the function
  // metadatas are and how they are used if we can do something else to replicate the behavior of
  // the function metadatas that might be good (ie annotating functions or something).
  Map<Target, Integer> workspace_sizes;
  Map<Target, Integer> io_sizes;
  Map<Target, Integer> constant_sizes;
  Map<Target, tir::PrimFunc> tir_primfuncs;
  Map<Target, Function> relay_primfuncs;

  Optional<Map<GlobalVar, tir::PrimFunc>> prim_fns =
      relay_func->GetAttr<Map<GlobalVar, tir::PrimFunc>>("prim_funcs");
  CHECK(prim_fns) << "primitive functions not set on Relay function by TECompiler.";

  Optional<GlobalVar> prim_fn_var = relay_func->GetAttr<GlobalVar>("prim_fn_var");
  CHECK(prim_fn_var) << "prim_fn_var must be set on Relay functions by TECompiler.";

  Optional<Target> relay_target = relay_func->GetAttr<Target>("target");
  CHECK(relay_target) << "target must be set on Relay functions by the TECompiler.";

  for (const auto& kv : prim_fns.value()) {
    auto prim_fn = Downcast<tir::PrimFunc>(kv.second);
    CHECK(prim_fn.defined()) << "the primitive function must be defined";

    auto workspace_byte_alignment =
        relay_target.value()->GetAttr<Integer>("workspace_byte_alignment").value_or(16);

    Integer workspace_size = CalculateWorkspaceBytes(prim_fn, workspace_byte_alignment);

    // Workspace sizes
    Target prim_fn_target;
    if (prim_fn->attrs->dict.count("target")) {
      prim_fn_target = Downcast<Target>(prim_fn->attrs->dict["target"]);
    } else {
      prim_fn_target = relay_target.value();
    }

    workspace_sizes.Set(prim_fn_target, workspace_size);

    // Calculating size for I/O
    for (auto const& param : prim_fn->params) {
      auto p_shape = prim_fn->buffer_map[param]->shape;
      int num_of_elements = 1;
      for (const auto& dim_index_expr : p_shape) {
        if (dim_index_expr->IsInstance<IntImmNode>()) {
          num_of_elements *= dim_index_expr.as<IntImmNode>()->value;
        } else {
          // If shape is dynamic, we cannot calculate workspace in compile time.
          num_of_elements = 0;
        }
      }
      int element_size = prim_fn->buffer_map[param]->dtype.bytes();
      io_sizes.Set(prim_fn_target, element_size * num_of_elements);
    }

    constant_sizes.Set(prim_fn_target, 0);
    tir_primfuncs.Set(prim_fn_target, prim_fn);
    relay_primfuncs.Set(prim_fn_target, relay_func);
  }

  backend::FunctionInfo fi = backend::FunctionInfo(workspace_sizes, io_sizes, constant_sizes,
                                                   tir_primfuncs, relay_primfuncs);

  // The primitive function name here corresponds to the string we will use to generate
  // this Relay function at the low level.
  function_metadata.Set(prim_fn_var.value()->name_hint, fi);
}

LoweredModule LowerTE(const IRModule& module, TargetMap targets, DeviceMap device_context_map,
                      backend::StaticMemoryPlan memory_plan, const String& module_name,
                      std::function<void(Function)> process_fn) {
  TECompiler compiler;

  CHECK_EQ(module->functions.size(), 1)
      << "There should only be one function in the module passed to LowerTE";

  auto pass = CreateFunctionPass(
      [=](Function func, IRModule module, PassContext ctx) {
        LowerTensorExpr lower_te(module, targets, device_context_map, process_fn, module_name,
                                 compiler);
        return Downcast<Function>(lower_te.VisitExpr(func));
      },
      0, "LowerTensorExpr", {});

  // TODO(@electriclilies, @jroesch): remove UpdateMainWorkspaceSize
  backend::FunctionInfo func_info =
      UpdateMainWorkspaceSize(module, targets, memory_plan->expr_to_storage_info);

  auto updated_module = pass(module);

  // A temporary solution until we can rewrite the auto-scheduler task extraction code to work
  // in a more reasonable way.
  if (backend::IsAutoSchedulerEnabled()) {
    const auto* te_compiler_update_weights =
        runtime::Registry::Get("auto_scheduler.relay_integration.te_compiler_update_weights");

    ICHECK(te_compiler_update_weights != nullptr)
        << "auto_scheduler.relay_integration.te_compiler_update_weights";

    Map<String, tvm::Integer> weight_map;

    for (auto pair : compiler->GetOpWeights()) {
      weight_map.Set(pair.first, pair.second);
    }

    (*te_compiler_update_weights)(weight_map);
  }

  LoweredModule lowered_module;
  lowered_module.main_module = updated_module;
  lowered_module.per_target_module = compiler->GetLoweredFunctions();
  lowered_module.external_mods = compiler->LowerExternalFunctions();
  lowered_module.main_func_info = func_info;
  return lowered_module;
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
