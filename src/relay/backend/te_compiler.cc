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
 * \file relay/backend/te_compiler.cc
 * \brief Manages the transition from Relay "Primitive" \p Functions to TIR \p PrimFuncs. Also
 * handles invocation of external codegen.
 *
 * \p LowerTEPass handles the following (as a monolithic blob of code):
 *
 *  - Most importantly, any function with the "Primitive" attribute is first converted to TE by
 *    \p LowerToTECompute (see te_compiler_cache.cc) using each operator's 'compute' function.
 *    The TE is then 'scheduled' to TIR using the 'anchor' operator's 'schedule' function. Both
 *    of those functions come from the \p OpStrategy returned by the Python
 *    'relay.backend.lower_call' function (see te_compiler.py).
 *    The TIR is packed as a \p PrimFunc and introduced as a new global function. Calls to the
 *    original "Primitive" function are then rewritten to the form:
 *    \code
 *      call_lowered(@new_global, (... original args...), attributes)
 *    \endcode
 *
 *  - The above "Primitive" function can appear:
 *     - As a global function
 *     - As a let-bound function
 *     - As an inline function, ie the 'op' of calls.
 *    In all three cases it is possible for the same "Primitive" function to be called multiple
 *    times, and that sharing must be respected.
 *
 *  - "Primitive" functions must have a "global_symbol" attribute matching their desired or
 *    existing global name. Care is taken to ensure GlobalVars with the same name are shared.
 *
 *  - It is possible for multiple structurally equal "Primitive" functions to appear in the same
 *    \p IRModule. Only one implementation should be generated, and all calls should share that
 *    implementation.
 *
 *  - When later converting to DPS (see memory_alloc.cc) we must handle functions who's result
 *    tensor shapes depend at runtime on the input tensor shapes and/or data.
 *     - That dependency is first described in TE form (see \p MakeShapeFunc in
 *       te_compiler_cache.cc), then scheduled to yield a 'dynamic shape function' \p PrimFunc.
 *       This relies on each operator's "FShapeFunc" and "TShapeDataDependent" attributes.
 *       Since shapes are rank-1 tensors everything can be reflected back down into the regular
 *       TE/TIR forms.
 *     - Then the call_lowered attributes must record everything about the dynamic shape function
 *       later needed by memory_alloc.cc. We call this 'cross linking' the call with the shape
 *       function.
 *
 *  - Two external codegen mechanisms are supported, both triggered by "Primitive" functions which
 *    also have a "Compiler" attribute bound to $compiler:
 *     - Function-at-a-time (old style): The primitive function is passed to the function
 *       registered as 'relay.ext.$compiler'. The function returns a runtime::Module which
 *       should return true for \p ImplementsFunction for the function's global name. That
 *       module is added to the IRModule's "external_mods" attributes.
 *     - IRModule-at-a-item (new style): The \p RelayToTIRTargetHook sub-pass looks for
 *       $compiler names which correspond to TargetKind names with a \p RelayToTIR attribute.
 *       The \p Pass bound to that attribute is run, and each such 'custom' pass can do what
 *       it likes, including replacing Functions with PrimFuncs, or adding new runtime::Modules
 *       to the IRModule's "external_mods" attribute.
 *
 *  - Calls to functions added by external codegen are also rewritten to call_lowered form, and
 *    may also require cross-linking to dynamic shape functions. However, since the functions
 *    are/will be implemented by a runtime::Module all the Relay type information is no longer
 *    available. So the Relay definitions for these "Primitive" "Compiler" functions are retained
 *    in the \p IRModule, but marked with the "Extern" attribute to signal the function is now
 *    just for carrying metadata.
 *
 *  - Some operators are handled specially:
 *     - 'reshape', since it's a no-op on the underlying tensor buffer, and this is handled by
 *       condition tests in many passes.
 *     - 'debug', since it's intercepted differently depending on runtimes.
 *
 * TODO(mbs): This desperately deserves a refactor to separate all these concerns. See Relax.
 */

#include "./te_compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/transform.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../transforms/device_aware_visitors.h"
#include "./te_compiler_cache.h"
#include "./utils.h"

namespace tvm {
namespace relay {
// TODO(@jroesch, @csullivan): declare directly elsewhere
backend::StaticMemoryPlan GraphPlanMemory(const Function& func);

namespace tec {

using namespace tvm::relay::transform;

TVM_REGISTER_OBJECT_TYPE(TECompilerNode);

class TECompilerImpl : public TECompilerNode {
 public:
  explicit TECompilerImpl(Optional<IRModule> opt_mod, Optional<String> opt_mod_name)
      : global_var_supply_(GlobalVarSupply(NameSupply(opt_mod_name.value_or("")))),
        constant_name_supply_(NameSupply("")) {
    // Make sure we don't collide with any existing globals in the module.
    if (opt_mod) {
      for (const auto& kv : opt_mod.value()->functions) {
        global_var_supply_->name_supply_->ReserveName(kv.first->name_hint, false);
      }
    }
  }

  // Lower the function.
  CachedFunc Lower(const CCacheKey& key) {
    return LowerInternal(key, global_var_supply_)->cached_func;
  }

  // TODO(gigiblender): Only to be called by the global TE compiler.
  //  Remove this when the global TE compiler is removed.
  CachedFunc Lower(const CCacheKey& key, const String mod_name) {
    global_var_supply_->name_supply_->prefix_ = mod_name;
    return LowerInternal(key, global_var_supply_)->cached_func;
  }

  // For now, build one module per function.
  PackedFunc JIT(const CCacheKey& key) final {
    CCacheValue value = LowerInternal(key, GlobalVarSupply(NameSupply("")));
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

  IRModule GetLoweredFunctions() {
    VLOG(1) << "GetLoweredFunctions";
    IRModule mod;
    // Extract lowered functions from the cache
    for (const auto& it : cache_) {
      auto source_func = it.first;
      auto lowered_func = it.second;

      IRModule lowered_mod = lowered_func->cached_func->funcs;

      // Annotate functions with their target and put them in the return module
      for (const auto& kv : lowered_mod->functions) {
        const GlobalVar& var = kv.first;
        const BaseFunc& func = kv.second;

        // Only add functions that are not external functions
        if (!func->GetAttr<String>(attr::kCompiler).defined()) {
          ICHECK(func->IsInstance<tir::PrimFuncNode>())
              << "Expected all functions that are not external to be PrimFuncs, but found:"
              << std::endl
              << PrettyPrint(func);
          const tir::PrimFunc& prim_func = Downcast<tir::PrimFunc>(func);
          mod->Update(var, WithAttr(prim_func, tvm::attr::kTarget, source_func->target));
        }
      }
    }

    // Extract lowered dynamic shape functions from the shape cache
    for (const auto& it : shape_func_cache_) {
      auto source_func = it.first;
      auto lowered_func = it.second;
      auto target = source_func->target;
      IRModule lowered_mod = lowered_func->cached_func->funcs;

      // Annotate functions with their target and put them in the return module
      for (auto kv : lowered_mod->functions) {
        const GlobalVar& var = kv.first;
        const BaseFunc& func = kv.second;
        const tir::PrimFunc& prim_func = Downcast<tir::PrimFunc>(func);
        mod->Update(var, WithAttr(prim_func, tvm::attr::kTarget, source_func->target));
      }
    }

    return mod;
  }

  void AddExterns(IRModule module) {
    // Everything tagged with "Compiler" has been compiled, so remove those definitions.
    std::vector<GlobalVar> to_be_deleted;
    for (const auto& kv : module->functions) {
      if (kv.second->GetAttr<String>(attr::kCompiler).defined()) {
        to_be_deleted.push_back(kv.first);
      }
    }
    for (const auto& global_var : to_be_deleted) {
      VLOG(1) << "Removing definition for external codegened '" << global_var->name_hint << "'";
      module->Remove(global_var);
    }
    // HOWEVER we still need a Relay definition to go with those now external functions, so
    // retrieve them from the cache and mark them with "ExternalSymbol".
    for (const auto& kv1 : cache_) {
      auto src_func = kv1.first->source_func;
      ICHECK(src_func.defined());
      if (src_func->GetAttr<String>(attr::kCompiler).defined()) {
        for (const auto& kv2 : kv1.second->cached_func->funcs->functions) {
          if (const auto* function_node = kv2.second.as<FunctionNode>()) {
            // Abandon the existing function annotations.

            // Unfortunately, Optional<DictAttrs>() is indistinguishable from
            // NullValue<DictAttrs>(), and DictAttrs() is nullptr, so to erase the attributes, we
            // need pass in DictAttrs<Map<String, ObjectRef>()), which is a DictAttrs containing no
            // attributes.
            Function function =
                WithFields(GetRef<Function>(function_node), function_node->params,
                           function_node->body, function_node->ret_type, function_node->type_params,
                           /* erase attributes */ DictAttrs(Map<String, ObjectRef>()));
            // Mark function as 'extern'.
            function = WithAttr(std::move(function), attr::kExtern, Integer(1));
            module->Add(kv2.first, function);
          }
        }
      }
    }
  }

  Array<tvm::runtime::Module> LowerExternalFunctions() {
    Array<tvm::runtime::Module> ret;
    std::vector<CCacheKey> cached_ext_funcs;

    for (const auto& it : cache_) {
      auto src_func = it.first->source_func;
      ICHECK(src_func.defined());
      Optional<String> opt_compiler = src_func->GetAttr<String>(attr::kCompiler);
      if (opt_compiler.defined()) {
        Optional<String> opt_symbol_name = src_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        ICHECK(opt_symbol_name.defined()) << "No external symbol is set for:" << std::endl
                                          << PrettyPrint(src_func);
        VLOG(1) << "using external codegen '" << opt_compiler.value() << "' for name '"
                << opt_symbol_name.value() << "' and function:" << std::endl
                << PrettyPrint(src_func);
        cached_ext_funcs.push_back(it.first);

        std::string ext_name = "relay.ext." + opt_compiler.value();
        auto pf = tvm::runtime::Registry::Get(ext_name);
        ICHECK(pf) << "Failed to find the external codegen tool for " << ext_name;
        // No need to keep compiler attribute at this point, functions have been
        // extracted for specific codegen.
        src_func = WithAttr(std::move(src_func), attr::kCompiler, NullValue<ObjectRef>());
        VLOG_CONTEXT << opt_compiler.value();
        With<Target> with_target(it.first->target);
        runtime::Module ext_mod = (*pf)(src_func);
        if (ext_mod.defined()) {
          // TODO(mbs): Can this be an ICHECKs?
          if (!ext_mod->ImplementsFunction(opt_symbol_name.value())) {
            VLOG(1) << "Note that the external codegen for '" << opt_compiler.value()
                    << "' returned a runtime module which does not appear to implement '"
                    << opt_symbol_name.value() << "'";
          }
          ret.push_back(ext_mod);
        } else {
          // It is valid for the external codegen function to return null:
          //  - Unit tests can use it.
          //  - The true compilation may have already been handled by a RelayToTIR custom pass
          //    on the Target's kind. The original Relay functions will be left in place so
          //    that we can capture that their function names are now externally defined.
          VLOG(1) << "Note that no external runtime module was generated by external codegen '"
                  << opt_compiler.value() << "'";
        }
      }
    }

    // No need to cache external functions as we collected them all to create
    // external runtime modules.
    for (const auto& it : cached_ext_funcs) {
      cache_.erase(it);
    }
    return ret;
  }

  Map<GlobalVar, String> GetDeviceContexts() { return device_contexts_; }
  void SetDeviceContexts(const Map<GlobalVar, String>& device_contexts) {
    device_contexts_ = device_contexts;
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
  CCacheValue LowerInternal(const CCacheKey& key, GlobalVarSupply global_var_supply) {
    VLOG(1) << "lowering:" << std::endl
            << PrettyPrint(key->source_func) << std::endl
            << "for target:" << std::endl
            << key->target->ToDebugString();
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      VLOG(1) << "already lowered to name:" << std::endl
              << PrettyPrint(it->second->cached_func->prim_fn_var);
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 1;
      cache_[key] = value;
    }
    cur_ccache_key_ = key;

    Optional<String> opt_compiler = key->source_func->GetAttr<String>(attr::kCompiler);
    if (opt_compiler.defined()) {
      // Don't compile now since we don't have anywhere to put the resulting runtime module.
      // Instead place the original definition in the cache and wait for LowerExternalFunctions.
      IRModule ir_module({}, {});
      Optional<String> opt_global_symbol =
          key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(opt_global_symbol.defined()) << "External function has not been attached a name yet.";
      // Note that the source_func may already be bound to a global function in the module
      // we are compiling, in which case we should not attempt to make its name unique w.r.t.
      // the module's globals. Furthermore, the external codegen tool must bind the compiled
      // function to the "global_symbol" attribute on the source_func. So do not use GetUniqueName
      // here.
      auto global_var = global_var_supply->UniqueGlobalFor(opt_global_symbol.value(), false);
      global_var->checked_type_ = key->source_func->checked_type();
      ir_module->Add(global_var, key->source_func);
      value->cached_func = CachedFunc(key->target, global_var, {}, {}, te::Schedule{nullptr},
                                      tir::PrimFunc{nullptr}, {}, ir_module);
      // Collect these here as it's removed in LowerExternalFunctions()
      device_contexts_.Set(value->cached_func->prim_fn_var, opt_compiler.value());
      VLOG(1) << "preparing to use external codegen '" << opt_compiler.value()
              << "' with name:" << std::endl
              << PrettyPrint(value->cached_func->prim_fn_var) << std::endl
              << "and definitions:" << std::endl
              << PrettyPrint(value->cached_func->funcs);
      return value;
    }

    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    value->cached_func =
        PrimFuncFor(key->source_func, key->target, global_var_supply, constant_name_supply_);

    if (value->cached_func->prim_func.defined()) {
      VLOG(1) << "Lowering PrimFunc";
      IRModule lowered = tvm::LowerPrimFunc(value->cached_func->prim_func.value(),
                                            value->cached_func->prim_fn_var->name_hint, false);
      ICHECK_EQ(lowered->functions.size(), 1);
      for (const auto& kv : lowered->functions) {
        value->cached_func->funcs->Add(value->cached_func->prim_fn_var, kv.second);
      }
    } else {
      // NOTE: array will copy on write.
      Array<te::Tensor> all_args = Array<te::Tensor>(value->cached_func->inputs);
      for (te::Tensor arg : value->cached_func->outputs) {
        all_args.push_back(arg);
      }
      Array<runtime::NDArray> all_consts;
      for (auto kv : value->cached_func->constant_tensors) {
        all_args.push_back(kv.second);
        all_consts.push_back(kv.first->data);
      }
      // lower the function
      std::unordered_map<te::Tensor, tir::Buffer> binds;

      // If we have memory scopes, need to create tir::Buffer knowing this info
      size_t i = 0;  // for corresponding from tensor array
      for (Var param : key->source_func->params) {
        if (!param->virtual_device()->memory_scope.empty()) {
          for (const auto& ttype : FlattenTupleType(param->checked_type())) {
            te::Tensor x_ref = value->cached_func->inputs[i];
            // verification if we have synced params and tensors
            ICHECK(ttype->dtype == x_ref->dtype && ttype->shape.size() == x_ref->shape.size())
                << "function parameter does not correspond to prepared tensor";
            binds[x_ref] =
                tir::BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype, x_ref->op->name, -1, 0,
                                               false, param->virtual_device()->memory_scope);
          }
        }
        i++;
      }
      if (key->virtual_device != VirtualDevice::FullyUnconstrained() &&
          !key->virtual_device->memory_scope.empty() &&
          key->virtual_device->memory_scope != "global") {
        ICHECK(value->cached_func->outputs.size() == 1)
            << "Expect only one output for defined memory scope";
        te::Tensor x_ref = value->cached_func->outputs[0];
        binds[x_ref] =
            tir::BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype, x_ref->op->name, -1, 0,
                                           false, key->virtual_device->memory_scope);
      }
      auto func_name = value->cached_func->prim_fn_var->name_hint;
      VLOG(1) << "scheduling";
      IRModule scheduled_module = tvm::LowerSchedule(value->cached_func->schedule, all_args,
                                                     func_name, binds, global_var_supply);
      scheduled_module->Update(tir::transform::BindParams(all_consts)(scheduled_module));
      for (const auto& kv : scheduled_module->functions) {
        GlobalVar global_var = kv.first;
        auto func = kv.second;
        // Propagate the structural hash of the relay function to the tir
        // function so associations can be made between the two.
        Optional<String> hash = key->source_func->attrs.GetAttr<String>("hash");
        if (hash) {
          func = WithAttrs(Downcast<tir::PrimFunc>(func), {{String("hash"), hash.value()}});
        }
        value->cached_func->funcs->Add(global_var, func);
      }
      ICHECK(value->cached_func->funcs->Lookup(value->cached_func->prim_fn_var)
                 .as<tir::PrimFuncNode>());
    }
    VLOG(1) << "lowered to name:" << std::endl
            << PrettyPrint(value->cached_func->prim_fn_var) << std::endl
            << "with definitions:" << std::endl
            << PrettyPrint(value->cached_func->funcs);

    return value;
  }

  // implement lowered shape func
  CCacheValue LowerShapeFuncInternal(const CCacheKey& key) {
    VLOG(1) << "lowering dynamic shape function for:" << std::endl
            << PrettyPrint(key->source_func) << std::endl
            << "for target:" << std::endl
            << key->target->ToDebugString();
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
    value->cached_func = ShapeFuncFor(key->source_func, key->target, global_var_supply_);

    ICHECK(
        value->cached_func->funcs->Lookup(value->cached_func->prim_fn_var).as<tir::PrimFuncNode>());

    VLOG(1) << "lowered to name:" << std::endl
            << PrettyPrint(value->cached_func->prim_fn_var) << std::endl
            << "with definitions:" << std::endl
            << PrettyPrint(value->cached_func->funcs);
    return value;
  }

  Map<String, Integer> GetOpWeights() const {
    Map<String, Integer> weights;
    for (const auto& kv : cache_) {
      auto value = kv.second;
      auto name = value->cached_func->prim_fn_var->name_hint;
      weights.Set(name, value->use_count);
    }
    return weights;
  }

  // TODO(mbs): Hold the output module here and reduce the cache_ to just be from
  // Function to GlobalVar.

  /*! \brief compiler cache lock*/
  std::mutex mutex_;
  /*! \brief internal GlobalVarSupply to get unique GlobalVars  */
  GlobalVarSupply global_var_supply_;
  /*! \brief A NameSupply object for assigning unique names to constants, across different
   * invocations of PrimFuncFor. */
  NameSupply constant_name_supply_;
  /*! \brief internal compiler cache */
  std::unordered_map<CCacheKey, CCacheValue> cache_;
  /*! \brief internal compiler cache for shape funcs */
  std::unordered_map<CCacheKey, CCacheValue> shape_func_cache_;
  /*! \brief the cache key of the function that is being lowered currently*/
  CCacheKey cur_ccache_key_;
  /*! \brief Map of GlobalVar to C Device API context names */
  Map<GlobalVar, String> device_contexts_;
};

TECompiler::TECompiler(Optional<IRModule> opt_mod, Optional<String> mod_name) {
  auto object = make_object<TECompilerImpl>(std::move(opt_mod), std::move(mod_name));
  data_ = object;
}

/*! \brief The global TE compiler */
// TODO(mbs): To be terminated with extreme prejudice.
TECompiler& TECompiler::Global() {
  static TECompiler* inst =
      new TECompiler(make_object<TECompilerImpl>(Optional<IRModule>(), Optional<String>()));
  return *inst;
}
TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.use_auto_scheduler", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.use_meta_schedule", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.use_meta_schedule_dispatch", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.tir_converter", String);

TVM_REGISTER_GLOBAL("relay.backend._TECompilerGlobal").set_body_typed([]() {
  return TECompiler::Global();
});

TVM_REGISTER_GLOBAL("relay.backend._make_CCacheKey")
    .set_body_typed([](Function source_func, Target target) {
      return CCacheKey(source_func, target);
    });

TVM_REGISTER_GLOBAL("relay.backend._make_LoweredOutput")
    .set_body_typed([](tvm::Array<te::Tensor> outputs, OpImplementation impl) {
      return LoweredOutput(outputs, impl);
    });

TVM_REGISTER_GLOBAL("relay.backend._TECompilerClear").set_body_typed([](TECompiler self) {
  self->Clear();
});

TVM_REGISTER_GLOBAL("relay.backend._TECompilerLower")
    .set_body_typed([](TECompiler self, CCacheKey key, const String mod_name) {
      return self->Lower(key, mod_name);
    });

TVM_REGISTER_GLOBAL("relay.backend._TECompilerJIT")
    .set_body_typed([](TECompiler self, CCacheKey key) { return self->JIT(key); });

TVM_REGISTER_GLOBAL("relay.backend._TECompilerListItems").set_body_typed([](TECompiler self) {
  TECompilerImpl* ptr = dynamic_cast<TECompilerImpl*>(self.operator->());
  ICHECK(ptr != nullptr);
  return ptr->ListItems();
});

using AnalysisRemapping = std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual>;

/*!
 * \brief Rewrites call expressions to Relay Functions marked as "primitive"
 * to calls to the corresponding TIR PrimFunc for the appropriate target.
 *
 * \code
 * %0 = fn(...) { prim_op(...) }     OR   let %p = fn(...) { prim_op(...) }
 * ... %0(...) ...                        ... %p(...) ...
 * ==>
 * def @q(..., target=<target>) { <tir body> }
 * ... @q(...) ...
 * \endcode
 *
 * Requires FuseOps, ToANormalForm, EtaExpand and InferType to have run.
 *
 * FuseOps is needed to identify and lift all prim op calls:
 * \code
 * ... prim_op(...) ...
 * ==>
 * %0 = fn(...) { prim_op(...) }
 * ... %0(...) ...
 * \endcode
 *
 * ToANormalForm is needed so we only need to consider vars and function literals as the call
 * target.
 *
 * EtaExpand is needed to ensures all calls to primitives are direct:
 * \code
 * let %p1 = fn(...) { prim_op1(...) }
 * let %p2 = fn(...) { prim_op2(...) }
 * let %p = if (...) { %p1 } else { %p2 }
 * ... %p(...) ...
 * ==>
 * let %p1 = fn(...) { prim_op1(...) }
 * let %p2 = fn(...) { prim_op2(...) }
 * let %p = fn(...) { if (...) { %p1(...) } else { %p2(...) } }
 * ... %p(...) ...
 * \endcode
 */
class LowerTensorExprMutator : public DeviceAwareExprMutator {
 public:
  LowerTensorExprMutator(IRModule module, ProcessFn process_fn, CompilationConfig config,
                         TECompiler compiler)
      : DeviceAwareExprMutator(module),
        module_(std::move(module)),
        process_fn_(std::move(process_fn)),
        config_(std::move(config)),
        compiler_(std::move(compiler)),
        debug_op_(Op::Get("debug")) {}

  /*!
   *  \brief Returns the primitive function associated with \p expr, or nullptr if none.
   */
  BaseFunc ResolveToPrimitive(const Expr& expr) {
    // NOTE: We can't assume expr->checked_type_ is defined, so can't early exit for first-order
    // expressions.
    if (const auto* global_var_node = expr.as<GlobalVarNode>()) {
      if (!module_->ContainGlobalVar(global_var_node->name_hint)) {
        // TODO(mbs): extern function cleanup
        // Assume the function is extern and thus no longer in the IRModule.
        return {};
      } else {
        BaseFunc base_func = module_->Lookup(GetRef<GlobalVar>(global_var_node));
        return ResolveToPrimitive(base_func);
      }
    } else if (auto prim_func = expr.as<tir::PrimFunc>()) {
      return prim_func.value();
    } else if (const auto* var_node = expr.as<VarNode>()) {
      auto itr = primitive_functions_.find(var_node);
      if (itr == primitive_functions_.end()) {
        // Not bound to a primitive function.
        return {};
      } else {
        return itr->second;
      }
    } else if (const auto* function_node = expr.as<FunctionNode>()) {
      if (function_node->HasNonzeroAttr(attr::kExtern)) {
        // We have a regular call to an 'extern' function. The call itself needs to be rewritten
        // to call_lowered form, and any required dynamic shape functions generated and
        // cross-linked.
        return GetRef<Function>(function_node);
      } else if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        if (const auto* call_node = function_node->body.as<CallNode>()) {
          if (call_node->op == debug_op_) {
            // Debug 'primitives' are not lowered.
            return {};
          }
        }
        // We have a regular call to a 'primitive' function (possibly with a 'Compiler' attribute).
        // We need to lower and rewrite the call.
        return GetRef<Function>(function_node);
      } else {
        // Not marked as primitive during partitioning or TVM fusion.
        return {};
      }
    } else {
      return {};
    }
  }

  /*!
   * \brief Returns a 'call_lowered' call to \p prim_fn_var with \p args and \p span with all the
   * required attributes filled in. Generally \p prim_fn_var will correspond to the lowered or
   * externally codegen-ed form of \p original_function, where \p lowered_functions binds all
   * the required lowered functions.
   *
   * The call's attributes will capture:
   *  - Any attributes on the original_function.
   *  - All the lowered functions.
   *    TODO(mbs): Pretty sure that's no longer needed.
   *  - Details needed to cross-link the call to it's dynamic shape function, if any.
   */
  Expr MakeLoweredCall(const BaseFunc& original_function, const GlobalVar& prim_fn_var,
                       Array<Expr> args, Span span, const Target& target,
                       const Map<GlobalVar, BaseFunc>& lowered_functions,
                       const te::Schedule& sch = {}) {
    auto opt_compiler = original_function->GetAttr<String>(attr::kCompiler);

    // Add some metadata on top of the *original function* and invoke the callback so it can
    // be captured.
    // TODO(@areusch, @jroesch): this metadata is for AOT, this should be our interface for AOT
    Map<GlobalVar, tir::PrimFunc> prim_fns;
    Array<GlobalVar> all_prim_fn_vars;
    for (const auto& kv : lowered_functions) {
      if (opt_compiler) {
        // We expect the original function to have just the "Extern" attribute signaling the
        // function (will be) compiled externally.
        ICHECK(kv.second.as<FunctionNode>())
            << PrettyPrint(kv.first) << " must be bound to an (external) Function";
      } else {
        // We expect one or more PrimFuncs, one of which corresponds to 'the' lowered primitive,
        // and the rest are in support of that via tir::Calls.
        ICHECK(kv.second.as<tir::PrimFuncNode>())
            << PrettyPrint(kv.first) << " must be bound to a PrimFunc";
        prim_fns.Set(kv.first, Downcast<tir::PrimFunc>(kv.second));
        all_prim_fn_vars.push_back(kv.first);
      }
    }

    // Alas, WithAttr cannot work with base classes.
    if (auto opt = original_function.as<te::PrimFunc>()) {
      auto func_with_metadata = opt.value();
      func_with_metadata = WithAttr(func_with_metadata, "prim_fn_var", prim_fn_var);
      func_with_metadata = WithAttr(func_with_metadata, "prim_funcs", prim_fns);
      func_with_metadata = WithAttr(func_with_metadata, tvm::attr::kTarget, target);
      // Store generated Schedules of operator
      if (sch.defined() && sch->keep_schedule_record) {
        func_with_metadata = WithAttr(func_with_metadata, "schedule", sch);
      }
      this->process_fn_(func_with_metadata);
    } else {
      auto func_with_metadata = original_function.as<Function>().value();
      func_with_metadata = WithAttr(func_with_metadata, "prim_fn_var", prim_fn_var);
      func_with_metadata = WithAttr(func_with_metadata, "prim_funcs", prim_fns);
      func_with_metadata = WithAttr(func_with_metadata, tvm::attr::kTarget, target);
      // Store generated Schedules of operator
      if (sch.defined() && sch->keep_schedule_record) {
        func_with_metadata = WithAttr(func_with_metadata, "schedule", sch);
      }
      this->process_fn_(func_with_metadata);
    }

    // Now prepare the attributes of the call_lowered.
    CallLoweredAttrs call_lowered_attrs;

    // TODO(mbs): "reshape" cleanup.
    if (!opt_compiler && original_function->HasNonzeroAttr(attr::kReshapeOnly)) {
      call_lowered_attrs.metadata.Set(attr::kReshapeOnly, tvm::Integer(1));
    }

    call_lowered_attrs.metadata.Set("relay_attrs", original_function->attrs);
    call_lowered_attrs.metadata.Set("all_prim_fn_vars", all_prim_fn_vars);

    if (const auto* function_node = original_function.as<FunctionNode>()) {
      if (IsDynamic(function_node->ret_type)) {
        // Create a dynamic shape function to calculate the expected shape of the results of
        // the lowered function.
        // Shape function keys use the original function as their 'function', but the generic 'cpu'
        // target as the target since all shape functions run on the host cpu irrespective of where
        // the primitive runs.
        CCacheKey shape_key(GetRef<Function>(function_node), config_->host_virtual_device->target);
        CachedFunc lowered_shape_func = compiler_->LowerShapeFunc(shape_key);

        // Capture the shape function's global var and parameters 'states' in call
        // annotations so calling convention can be recovered.
        // TODO(mbs): Shape cleanup.
        call_lowered_attrs.metadata.Set("prim_shape_fn_var", lowered_shape_func->prim_fn_var);
        call_lowered_attrs.metadata.Set("prim_shape_fn_states",
                                        lowered_shape_func->shape_func_param_states);
        call_lowered_attrs.metadata.Set(
            "prim_shape_fn_num_inputs",
            Integer(static_cast<int>(lowered_shape_func->inputs.size())));
        call_lowered_attrs.metadata.Set(
            "prim_shape_fn_num_outputs",
            Integer(static_cast<int>(lowered_shape_func->outputs.size())));
        Array<GlobalVar> all_prim_shape_fn_vars;
        for (const auto& kv : lowered_shape_func->funcs->functions) {
          CHECK(kv.second.as<tir::PrimFuncNode>()) << "must be a prim fn";
          all_prim_shape_fn_vars.push_back(kv.first);
        }
        call_lowered_attrs.metadata.Set("all_prim_shape_fn_vars", all_prim_shape_fn_vars);
      }
    }

    return CallLowered(prim_fn_var, std::move(args), std::move(call_lowered_attrs),
                       std::move(span));
  }

  std::pair<Var, Expr> PreVisitLetBinding_(const Var& var, const Expr& value) final {
    Var new_var = Downcast<Var>(Mutate(var));
    Expr new_value = Mutate(value);
    BaseFunc prim_func = ResolveToPrimitive(new_value);

    if (prim_func.defined()) {
      // Remember let var is bound (possibly indirectly) to a primitive function.
      primitive_functions_.emplace(var.get(), prim_func);
    }
    return {new_var, new_value};
  }

  Expr PostVisitLet_(const LetNode* pre_let_node, const LetNode* post_let_node) final {
    BaseFunc prim_func = ResolveToPrimitive(post_let_node->value);
    if (prim_func.defined()) {
      // Leaving let var scope
      primitive_functions_.erase(pre_let_node->var.get());
      // Drop the let node
      return post_let_node->body;
    }
    return DeviceAwareExprMutator::PostVisitLet_(pre_let_node, post_let_node);
  }

  Expr DeviceAwareVisitExpr_(const FunctionNode* function_node) override {
    if (function_node->HasNonzeroAttr(attr::kPrimitive) ||
        function_node->HasNonzeroAttr(attr::kExtern)) {
      // Nothing to lower inside primitive/external functions.
      return GetRef<Function>(function_node);
    } else {
      return DeviceAwareExprMutator::DeviceAwareVisitExpr_(function_node);
    }
  }

  Expr DeviceAwareVisitExpr_(const CallNode* call_node) override {
    // We can see six forms of calls:
    //  1. A 'normal' Relay call to a Function with the "Primitive" attribute and not "Compiler"
    //     attribute. We will need to lower that to a global PrimFunc and rewrite the call to:
    //       call_lowered(@new_global, (arg1, ..., argn), <attributes>)
    //     If needed, the call needs to be cross-linked with any dynamic shape functions.
    //     (However, some primitives are special and handled separately.)
    //  2. A 'normal' Relay call to a Function with the "Primitive" and "Compiler" attributes. We
    //     will need to invoke the "relay.ext.<compiler>" function to yield a runtime module, and
    //     rewrite the call to the same form as above. Dynamic shape function cross-linking may
    //     also be needed.
    //  3. A 'normal' Relay call to a Function with the "Extern" attribute. This function has
    //     already been compiled by an external codegen and a definition for it exists in some
    //     runtime module. Again, we rewrite to call_lowered form, and cross-link with a dynamic
    //     shape function if needed.
    //  4. A 'normal' Relay call to a PrimFunc which has already been supplied via a global
    //     definition. We rewrite those to use the call_lowered form, but otherwise nothing else
    //     needs to be done.
    //  5. A 'call_lowered' call from an earlier invocation of this pass or otherwise deliberately
    //     inserted. It has all the required attributes, and any associated dynamic shape function
    //     has been generated and cross-linked. These calls are not changed.
    //  6. A 'normal' Relay call to a Relay Function without any special attribute. These
    //     calls are not changed.
    //
    // Note that ResolveToPrimitive will yield non-null only for cases 1-4.

    // Prepare the arguments and op.
    Array<Expr> new_args;
    for (const auto& arg : call_node->args) {
      new_args.push_back(VisitExpr(arg));
    }
    Expr new_op = VisitExpr(call_node->op);

    // Look for (possibly indirect) calls to primitives.
    BaseFunc primitive_func = ResolveToPrimitive(call_node->op);
    if (!primitive_func.defined()) {
      // Cases 5 and 6: Leave as ordinary call.
      if (auto function = call_node->op.as<Function>()) {
        process_fn_(function.value());
      }
      return WithFields(GetRef<Call>(call_node), std::move(new_op), std::move(new_args));
    }

    // Special case for case 1: device_copies are left as calls to primitive operators
    // so that each backend can handle them directly.
    // TODO(mbs): device_copy cleanup. Would be better for FuseOps to just leave device_copy alone.
    if (const auto* function_node = primitive_func.as<FunctionNode>()) {
      DeviceCopyProps device_copy_props = GetDeviceCopyProps(function_node->body);
      if (device_copy_props.body.defined()) {
        ICHECK_EQ(new_args.size(), 1);
        return DeviceCopy(new_args[0], device_copy_props.src_virtual_device,
                          device_copy_props.dst_virtual_device);
      }
    }

    ICHECK(call_node->type_args.empty()) << "lowered functions cannot be polymorphic";

    // Case 4: If the function has already been lowered we just need to update the call.
    if (auto prim_func = primitive_func.as<tir::PrimFunc>()) {
      // Function should already be Target annotated by this point
      // but the TE Compiler metadata is still needed for the callback
      // TODO(Mousius) - Robustify this to not assume we're in the GlobalVar for Target Hooks
      Optional<Target> opt_target = primitive_func->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(opt_target.defined());
      auto prim_fn_var = Downcast<GlobalVar>(call_node->op);
      Map<GlobalVar, BaseFunc> prim_fns = {{prim_fn_var, prim_func.value()}};
      return MakeLoweredCall(primitive_func, prim_fn_var, std::move(new_args), call_node->span,
                             opt_target.value(), prim_fns);
    }

    // Determine the target for lowering or external codegen.
    Target target;
    Optional<String> opt_compiler = primitive_func->GetAttr<String>(attr::kCompiler);
    if (opt_compiler.defined()) {
      // This function needs to be compiled with external codegen.
      Optional<Target> opt_target = config_->FindPrimitiveTargetForKind(opt_compiler.value());
      if (opt_target.defined()) {
        // The target is what's supplied by the compilation config for kind matching the
        // "Compiler" name.
        target = opt_target.value();
      } else {
        // Legacy fallback.
        target = Target("ext_dev");
      }
    } else {
      // The target corresponding to the call_node expression's annotation.
      VirtualDevice virtual_device = GetVirtualDevice(GetRef<Call>(call_node));
      ICHECK(!virtual_device->IsFullyUnconstrained()) << PrettyPrint(GetRef<Call>(call_node));
      target = virtual_device->target;
      ICHECK(target.defined());
    }

    if (primitive_func->HasNonzeroAttr(attr::kExtern)) {
      // Case 3: Function has already been compiled.
      GlobalVar prim_fn_var = Downcast<GlobalVar>(call_node->op);
      return MakeLoweredCall(primitive_func, prim_fn_var, std::move(new_args), call_node->span,
                             target, /*lowered_functions=*/{});
    } else {
      // Cases 1 and 2: lower the primitive function for the desired target, possibly using external
      // codegen.
      CCacheKey key(Downcast<Function>(primitive_func), target,
                    GetVirtualDevice(GetRef<Call>(call_node)));
      CachedFunc cfunc = compiler_->Lower(key);
      ICHECK(cfunc.defined());
      return MakeLoweredCall(primitive_func, cfunc->prim_fn_var, std::move(new_args),
                             call_node->span, target, cfunc->funcs->functions, cfunc->schedule);
    }
  }

  IRModule module_;
  ProcessFn process_fn_;
  /*! \brief All available targets. */
  CompilationConfig config_;
  // Map from in-scope let-bound variables to Functions known to be primitive, or PrimFuncs which
  // have already been lowered. We'll rewrite these to the fresh global vars bound to the lowered
  // primitive function as we go. Those vars will be bound in the target device-type specific
  // module we'll ultimately emit for each required device-type. Note that a primitive may be
  // lowered for multiple device types, each which will be assigned a fresh var.
  std::unordered_map<const VarNode*, BaseFunc> primitive_functions_;
  TECompiler compiler_;
  // Cache ops that need to be frequently used later to reduce lookup overhead.
  const Op& debug_op_;
};

Pass LowerTensorExpr(TECompiler compiler, ProcessFn process_fn, CompilationConfig config) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function func, IRModule module, PassContext ctx) {
        LowerTensorExprMutator lower_te(module, process_fn, config, compiler);
        return Downcast<Function>(lower_te.Mutate(func));
      };
  return CreateFunctionPass(pass_func, 0, "LowerTensorExpr", {});
}

backend::FunctionInfo UpdateMainWorkspaceSize(const IRModule& mod, const CompilationConfig& config,
                                              Map<Expr, backend::StorageInfo> storage_info_map) {
  Function func = Downcast<Function>(mod->Lookup("main"));

  VLOG_CONTEXT << "UpdateMainWorkspaceSize";
  VLOG(1) << "calculating FunctionInfo for main:" << std::endl << PrettyPrint(func);

  // This is a Map<device,Map<storage_id, size>>
  // TODO(mbs): Collapsing VirtualDevices to just device type.
  std::unordered_map<DLDeviceType, std::unordered_map<int, int>, backend::EnumClassHash>
      sid_workspace;
  // This is a Map<device, size_of_inputs_and_outputs>
  std::unordered_map<DLDeviceType, int, backend::EnumClassHash> device_io;
  // This is a Map<device, size_of_constants>
  std::unordered_map<DLDeviceType, int, backend::EnumClassHash> device_consts;

  // Initialize the mapping from all storage identifiers to workspace sizes,
  // the amount of device io, and the device constants.
  for (const auto& kv : storage_info_map) {
    const backend::StorageInfo& storage_info = kv.second;
    const std::vector<int64_t>& storage_ids = storage_info->storage_ids;
    const std::vector<VirtualDevice>& virtual_devices = storage_info->virtual_devices;
    CHECK_EQ(storage_ids.size(), virtual_devices.size());
    for (uint32_t i = 0; i < virtual_devices.size(); i++) {
      DLDeviceType device_type = virtual_devices[i]->device_type();
      sid_workspace[device_type][storage_ids[i]] = 0;
      device_io[device_type] = 0;
      device_consts[device_type] = 0;
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
    const Expr& expr = kv.first;
    const backend::StorageInfo& storage_info = kv.second;
    int64_t size_bytes = backend::CalculateRelayExprSizeBytes(expr->checked_type());
    VLOG(1) << "expression:" << std::endl
            << PrettyPrint(expr) << std::endl
            << "of type:" << std::endl
            << PrettyPrint(expr->checked_type()) << std::endl
            << "has size " << size_bytes << " and storage info:" << std::endl
            << storage_info;
    const std::vector<int64_t>& storage_ids = storage_info->storage_ids;
    const std::vector<VirtualDevice>& virtual_devices = storage_info->virtual_devices;

    if (expr->IsInstance<ConstantNode>()) {
      for (const auto& virtual_device : virtual_devices) {
        DLDeviceType device_type = virtual_device->device_type();
        ICHECK_EQ(device_consts.count(device_type), 1);
        device_consts[device_type] += size_bytes;
      }
    } else if (expr->IsInstance<VarNode>() || expr.same_as(func->body)) {
      CHECK(size_bytes == 0 || virtual_devices.size() >= 1) << "must be at least one device";
      for (const auto& virtual_device : virtual_devices) {
        DLDeviceType device_type = virtual_device->device_type();
        device_io[device_type] += size_bytes;
      }
    } else {
      // TODO(@electriclilies): This code is never being called which means sid_workspace is not
      // updated.. This means that storage info is probably not being created correctly. Or is not
      // equivalent to what was here previously
      for (uint32_t i = 0; i < storage_ids.size(); i++) {
        // Here we record the largest size of the tensor
        // that share the same storage id, because storage_id will
        // be shared between multiple tensors that are not live simultaneously.
        DLDeviceType device_type = virtual_devices[i]->device_type();
        if (size_bytes > sid_workspace[device_type][storage_ids[i]]) {
          sid_workspace[device_type][storage_ids[i]] = size_bytes;
        }
      }
    }
  }

  // This is a Map<device, workspace_size>
  std::unordered_map<DLDeviceType, int, backend::EnumClassHash> device_workspace;
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
  for (const auto& target : config->primitive_targets) {
    workspace_sizes.Set(target, 0);
  }

  for (const auto& dev_and_size : device_workspace) {
    Target target = config->FindPrimitiveTargetForDeviceOrFail(dev_and_size.first);
    workspace_sizes.Set(target, dev_and_size.second);
    relay_primfuncs.Set(target, func);
  }
  for (const auto& dev_and_size : device_io) {
    Target target = config->FindPrimitiveTargetForDeviceOrFail(dev_and_size.first);
    io_sizes.Set(target, dev_and_size.second);
  }

  for (const auto& dev_and_size : device_consts) {
    Target target = config->FindPrimitiveTargetForDeviceOrFail(dev_and_size.first);
    ICHECK_EQ(constant_sizes.count(target), 0);
    constant_sizes.Set(target, dev_and_size.second);
  }

  backend::FunctionInfo func_info(std::move(workspace_sizes), std::move(io_sizes),
                                  std::move(constant_sizes), std::move(tir_primfuncs),
                                  std::move(relay_primfuncs));
  VLOG(1) << "func_info: " << func_info;
  return std::move(func_info);
}

/*!
 * \brief A function to create the function metadata for an input function (ie calculate buffer
 * input/output sizes)
 * \param func The function to calculate function metadata for
 * \param function_metadata The map that stores all the function metadatas
 */
void UpdateFunctionMetadata(BaseFunc func,
                            Map<String, backend::FunctionInfo>& function_metadata,  // NOLINT(*)
                            Integer workspace_byte_alignment) {
  VLOG_CONTEXT << "UpdateFunctionMetadata";
  VLOG(1) << "updating function metadata for:" << std::endl << PrettyPrint(func);
  // Originally UpdateFunctionMetadata took in CCachedFunc and looped through all the funcs stored
  // there Now the goal is to take only one func because process_fn should be controlling the
  // iteration However, to do the workspace calculations we need the primfuncs. So process_fn
  // needs to either access the cached funcs or be directly passed primfuncs This is bad and
  // ideally we don't want process_fn to look at primfuncs There's also the question now of what
  // the function metadatas are and how they are used if we can do something else to replicate the
  // behavior of the function metadatas that might be good (ie annotating functions or something).
  Map<Target, Integer> workspace_sizes;
  Map<Target, Integer> io_sizes;
  Map<Target, Integer> constant_sizes;
  Map<Target, tir::PrimFunc> tir_primfuncs;
  Map<Target, Function> relay_primfuncs;

  Optional<Map<GlobalVar, tir::PrimFunc>> prim_fns =
      func->GetAttr<Map<GlobalVar, tir::PrimFunc>>("prim_funcs");
  CHECK(prim_fns) << "primitive functions not set on Relay function by TECompiler.";

  Optional<GlobalVar> prim_fn_var = func->GetAttr<GlobalVar>("prim_fn_var");
  CHECK(prim_fn_var) << "prim_fn_var must be set on Relay functions by TECompiler.";

  Optional<Target> relay_target = func->GetAttr<Target>(tvm::attr::kTarget);
  CHECK(relay_target) << "target must be set on Relay functions by the TECompiler.";

  for (const auto& kv : prim_fns.value()) {
    auto prim_fn = Downcast<tir::PrimFunc>(kv.second);
    CHECK(prim_fn.defined()) << "the primitive function must be defined";

    Integer workspace_size = CalculateWorkspaceBytes(prim_fn, workspace_byte_alignment);

    // Workspace sizes
    Target prim_fn_target;
    if (prim_fn->attrs->dict.count(tvm::attr::kTarget)) {
      prim_fn_target = Downcast<Target>(prim_fn->attrs->dict[tvm::attr::kTarget]);
    } else {
      prim_fn_target = relay_target.value();
    }

    workspace_sizes.Set(prim_fn_target, workspace_size);

    // Calculating size for I/O
    // TODO(mbs): See also the other three utils for calculating tensor bytesize.
    for (auto const& param : prim_fn->params) {
      bool not_a_buffer = prim_fn->buffer_map.count(param) == 0;
      if (not_a_buffer) {
        io_sizes.Set(prim_fn_target, 0);
        continue;
      }

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
    if (func->IsInstance<FunctionNode>()) {
      relay_primfuncs.Set(prim_fn_target, Downcast<Function>(func));
    }
  }

  backend::FunctionInfo fi = backend::FunctionInfo(
      std::move(workspace_sizes), std::move(io_sizes), std::move(constant_sizes),
      std::move(tir_primfuncs), std::move(relay_primfuncs));

  VLOG(1) << "FunctionInfo: " << PrettyPrint(prim_fn_var.value()) << " = " << PrettyPrint(fi);

  // The primitive function name here corresponds to the string we will use to generate
  // this Relay function at the low level.
  function_metadata.Set(prim_fn_var.value()->name_hint, fi);
}

/*! \brief Main lowering driving. */
IRModule LowerTE(const IRModule& module, const String& module_name, ProcessFn process_fn,
                 CompilationConfig config) {
  TECompiler compiler(module, module_name);

  // TODO(mbs): This is all unnecessarily convoluted. Better would be to accumulate the rewritten
  // module as we go (including rewritten Functions, lowered primitives, and runtime modules
  // generated by external toolchains), and use a pair of maps over vars and global vars
  // to global vars to remember which functions have already been lowered.

  // Lower all the callees in module:
  //  - Functions tagged with "Compiler" are unchanged (checked by CreateFunctionPass)
  //  - Functions tagged with "Primitive" are unchanged (checked by LowerTensorExprMutator)
  //  - Called functions tagged with "Compiler" are copied into the compiler cache with a fresh
  //    GlobalVar, and calls updated (sticking with regular Relay Call).
  //  - Calls to functions tagged with "Primitive" are compiled to PrimFuncs, and calls updated
  //    (using call_lowered convention).
  IRModule updated_module =
      LowerTensorExpr(compiler, std::move(process_fn), std::move(config))(module);

  // The Functions tagged with "Compiler" are now residing in the cache ready to be
  // compiled by LowerExternalFunctions. However we still need a record of them in the
  // IRModule so that the various executors can see which function names need to be
  // retrieved. They may, however, have been renamed.
  compiler->AddExterns(updated_module);

  // Add the lowered functions.
  IRModule lowered_module = compiler->GetLoweredFunctions();
  VLOG(1) << "capturing " << lowered_module->functions.size() << " new lowered functions";
  for (const auto& kv : lowered_module->functions) {
    if (updated_module->ContainGlobalVar(kv.first->name_hint)) {
      LOG(FATAL) << "duplicate bindings for '" << kv.first->name_hint
                 << "'. Existing is:" << std::endl
                 << PrettyPrint(updated_module->Lookup(kv.first->name_hint)) << std::endl
                 << "while new is:" << std::endl
                 << PrettyPrint(kv.second);
    }
    updated_module->Add(kv.first, kv.second);
  }

  // Invoke external codegen for all Functions in the cache tagged with "Compiler", and
  // annotate the module with the resulting runtime modules.
  // TODO(mbs): runtime modules should be first class rather than attributes.
  Array<runtime::Module> external_mods =
      module->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});
  Array<runtime::Module> new_external_mods = compiler->LowerExternalFunctions();
  VLOG(1) << "capturing " << external_mods.size() << " existing and " << new_external_mods.size()
          << " new external modules";
  for (const auto& mod : new_external_mods) {
    external_mods.push_back(mod);  // copy-on-write.
  }

  // Annotate the module with C Device API context mapping (this is until we have Targets
  // annotated for the C Device API)
  // TODO(Mousius) - Remove "device_contexts" as soon as we have the graph annotated properly with
  // Targets
  Map<GlobalVar, String> device_contexts =
      module->GetAttr<Map<GlobalVar, String>>("device_contexts", Map<GlobalVar, String>()).value();
  Map<GlobalVar, String> new_device_contexts = compiler->GetDeviceContexts();
  VLOG(1) << "capturing " << device_contexts.size() << " existing and "
          << new_device_contexts.size() << " new device contexts for external functions";
  for (const auto& kv : new_device_contexts) {
    ICHECK_EQ(device_contexts.count(kv.first), 0);
    device_contexts.Set(kv.first, kv.second);  // copy-on-write.
  }

  updated_module = WithAttrs(updated_module, {{tvm::attr::kExternalMods, std::move(external_mods)},
                                              {"device_contexts", std::move(device_contexts)}});

  if (backend::IsAutoSchedulerEnabled()) {
    // Capture all the 'operator weights', ie usage counts for each PrimFunc.
    Map<String, Integer> op_weights =
        module->GetAttr<Map<String, Integer>>("op_weights", Map<String, Integer>()).value();
    Map<String, Integer> new_op_weights = compiler->GetOpWeights();
    VLOG(1) << "capturing " << op_weights.size() << " existing and " << new_op_weights.size()
            << " new operator weights for PrimFuncs";
    for (const auto& kv : new_op_weights) {
      ICHECK_EQ(op_weights.count(kv.first), 0);
      op_weights.Set(kv.first, kv.second);  // copy-on-write.
    }
    updated_module = WithAttr(updated_module, "op_weights", std::move(op_weights));
  }

  return updated_module;
}

Map<Target, IRModule> GetPerTargetModules(IRModule mod) {
  std::unordered_map<Target, IRModule, backend::TargetStrHash, backend::TargetStrEqual>
      per_target_modules;
  for (const auto& kv : mod->functions) {
    const GlobalVar& var = kv.first;
    const BaseFunc& func = kv.second;
    if (func->IsInstance<tir::PrimFuncNode>()) {
      // Extract target
      Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(target) << "Target should be set at this point";

      // Put the function in per_target_modules
      if (!per_target_modules.count(target.value())) {
        // Initialize the IRModule for this target with the attributes from the input IRModule
        IRModule target_module = IRModule({}, {}, {}, {}, mod->attrs);
        // Add the function to the IRModule
        target_module->Add(var, func);
        per_target_modules[target.value()] = target_module;
      } else {
        // The IRModule for this target is initialized, so just add the function.
        IRModule target_module = per_target_modules.at(target.value());
        target_module->Add(var, func);
      }
    } else if (!func->IsInstance<relay::FunctionNode>()) {
      LOG(FATAL)
          << "The function types in the IRModule should be RelayFunction or PrimFunc, but got "
          << func->GetTypeKey();
    }
  }
  return per_target_modules;
}

Pass LowerTE(String module_name, CompilationConfig complilation_config, ProcessFn process_fn) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule module,
                                                                            PassContext ctx) {
    return LowerTE(module, module_name, process_fn, complilation_config);
  };

  return tvm::transform::Sequential(
      {tvm::relay::transform::RelayToTIRTargetHook(complilation_config),
       tvm::transform::CreateModulePass(pass_func, 0, "LowerTE", {"InferType"}), InferType(),
       tvm::tir::transform::ExtractPrimFuncConstants()});
}

TVM_REGISTER_GLOBAL("relay.tec.LowerTE")
    .set_body_typed([](String module_name, CompilationConfig compilation_config) {
      return LowerTE(std::move(module_name), std::move(compilation_config));
    });

}  // namespace tec
}  // namespace relay
}  // namespace tvm
