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
 * \file relay/backend/compile_engine.cc
 * \brief Internal compilation engine.
 */
#include "compile_engine.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../runtime/meta_data.h"
#include "../transforms/pass_utils.h"
#include "te_compiler_cache.h"
#include "utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_OBJECT_TYPE(CompileEngineNode);

class CompileEngineImpl : public CompileEngineNode {
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
    if (value->packed_func != nullptr) return value->packed_func;
    auto m = build(value->cached_func->funcs, key->target, Target(nullptr));
    value->packed_func = m.GetFunction(value->cached_func->prim_fn_var->name_hint);
    return value->packed_func;
  }

  CachedFunc LowerShapeFunc(const CCacheKey& key) final {
    return LowerShapeFuncInternal(key)->cached_func;
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
        ICHECK(code_gen.defined()) << "No external codegen is set";
        std::string code_gen_name = code_gen.value();
        cached_ext_funcs.push_back(it.first);

        auto symbol_name = src_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        ICHECK(symbol_name.defined()) << "No external symbol is set for:\n"
                                      << AsText(src_func, false) << "\n"
                                      << "Functions with external codegen must have the "
                                      << tvm::attr::kGlobalSymbol << " attr set.";

        std::string sn = symbol_name.value();
        if (!cached_symbol.count(sn)) {
          cached_symbol[sn] = code_gen_name;
        } else {
          ICHECK_NE(cached_symbol[sn], code_gen_name)
              << "Found duplicated symbol: " << sn << " for: " << code_gen_name;
        }

        std::string ext_name = "relay.ext." + code_gen_name;
        auto pf = tvm::runtime::Registry::Get(ext_name);
        ICHECK(pf) << "Failed to find the codegen tool for " << ext_name << "\n";
        // No need to keep compiler attribute at this point, functions have been
        // extracted for specific codegen.
        src_func = WithAttr(std::move(src_func), attr::kCompiler, NullValue<ObjectRef>());
        runtime::Module ext_mod = (*pf)(src_func);

        // todo(@zhiics, @jroesch): Should this be a user visible error?
        ICHECK(ext_mod.defined()) << "No external library was generated for " << ext_name
                                  << "even though it was requested"
                                     "by the annotated function "
                                  << PrettyPrint(src_func);

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

  // List all items in the shape_func_cache.
  Array<ObjectRef> ListShapeFuncItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<ObjectRef> items;
    for (auto& kv : shape_func_cache_) {
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
      value->use_count = 0;
      if (!backend::IsCompileEngineCacheDisabled()) {
        cache_[key] = value;
      }
    }
    cur_ccache_key_ = key;

    // No need to lower external functions for now. We will invoke the external
    // codegen tool once and lower all functions together.
    if (key->source_func->GetAttr<String>(attr::kCompiler).defined()) {
      auto ir_module = IRModule();
      const auto name_node = key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(name_node.defined()) << "External function has not been attached a name yet.";
      auto func_name = std::string(name_node.value());
      auto target = Target("ext_dev");
      auto global_var = GlobalVar(func_name);
      global_var->checked_type_ = key->source_func->checked_type();
      ir_module->Add(global_var, key->source_func);
      value->cached_func = CachedFunc(target, global_var, {}, {}, te::Schedule(), {}, ir_module);
      return value;
    }

    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    auto cfunc = PrimFuncFor(key->source_func, key->target, [&](std::string name) {
      return GetUniqueName(mangle_fn(name), &name_map_);
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
    // lower the function
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

/*! \brief The global compile engine */
CompileEngine& CompileEngine::Global() {
  // intentionally allocate raw pointer to avoid
  // free during destructuion.
  static CompileEngine* inst = new CompileEngine(make_object<CompileEngineImpl>());
  return *inst;
}

TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.use_auto_scheduler", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.disable_compile_engine_cache", Bool);

// Make LoweredOutput
TVM_REGISTER_GLOBAL("relay.backend._make_LoweredOutput")
    .set_body_typed([](tvm::Array<te::Tensor> outputs, OpImplementation impl) {
      return LoweredOutput(outputs, impl);
    });

TVM_REGISTER_GLOBAL("relay.backend._make_CCacheKey")
    .set_body_typed([](Function source_func, Target target) {
      return CCacheKey(source_func, target);
    });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineGlobal").set_body_typed([]() {
  return CompileEngine::Global();
});

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineClear").set_body_typed([](CompileEngine self) {
  self->Clear();
});

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineLower")
    .set_body_typed([](CompileEngine self, CCacheKey key, const String mod_name) {
      return self->Lower(key, mod_name);
    });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineLowerShapeFunc")
    .set_body_typed([](CompileEngine self, CCacheKey key) { return self->LowerShapeFunc(key); });

TVM_REGISTER_GLOBAL("relay.backend._CompileLowerExternalFunctions")
    .set_body_typed([](CompileEngine self) { return self->LowerExternalFunctions(); });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineJIT")
    .set_body_typed([](CompileEngine self, CCacheKey key) { return self->JIT(key); });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineListItems").set_body_typed([](CompileEngine self) {
  CompileEngineImpl* ptr = dynamic_cast<CompileEngineImpl*>(self.operator->());
  ICHECK(ptr != nullptr);
  return ptr->ListItems();
});

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineListShapeFuncItems")
    .set_body_typed([](CompileEngine self) {
      CompileEngineImpl* ptr = dynamic_cast<CompileEngineImpl*>(self.operator->());
      ICHECK(ptr != nullptr);
      return ptr->ListShapeFuncItems();
    });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineGetCurrentCCacheKey")
    .set_body_typed([](CompileEngine self) {
      CompileEngineImpl* ptr = dynamic_cast<CompileEngineImpl*>(self.operator->());
      ICHECK(ptr != nullptr);
      return ptr->GetCurrentCCacheKey();
    });

}  // namespace relay
}  // namespace tvm
