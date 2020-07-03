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
 * \file src/target/generic_func.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/node/node.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/generic_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

TVM_REGISTER_NODE_TYPE(GenericFuncNode);

struct GenericFunc::Manager {
  std::unordered_map<std::string, GenericFunc> fmap;
  // mutex
  std::mutex mutex;

  Manager() {}

  static Manager* Global() {
    static Manager inst;
    return &inst;
  }
};

GenericFunc GenericFunc::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    auto f = make_object<GenericFuncNode>();
    f->name_ = name;
    auto gf = GenericFunc(f);
    m->fmap[name] = gf;
    return gf;
  } else {
    return it->second;
  }
}

void GenericFunc::RegisterGenericFunc(GenericFunc func, const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex>(m->mutex);
  auto it = m->fmap.find(name);
  CHECK(it == m->fmap.end()) << "GenericFunc already registered " << name;
  func->name_ = name;
  m->fmap[name] = func;
}

GenericFunc& GenericFunc::set_default(const PackedFunc value, bool allow_override) {
  auto node = static_cast<GenericFuncNode*>(operator->());
  if (!allow_override) {
    CHECK(node->generic_func_ == nullptr)
        << "Generic function already registered for " << node->name_;
  }
  node->generic_func_ = value;
  return *this;
}

GenericFunc& GenericFunc::register_func(const std::vector<std::string>& tags,
                                        const PackedFunc value, bool allow_override) {
  for (auto& t : tags) {
    if (!allow_override) {
      auto iter = (*this)->dispatch_dict_.find(t);
      CHECK(iter == (*this)->dispatch_dict_.end())
          << "Tag " << t << " already registered for schedule factory " << (*this)->name_;
    }
    (*this)->dispatch_dict_[t] = value;
  }
  return *this;
}

void GenericFunc::CallPacked(TVMArgs args, TVMRetValue* ret) const {
  auto node = static_cast<const GenericFuncNode*>(get());
  auto target = Target::Current(true);
  PackedFunc func;

  if (target.defined()) {
    for (auto& k : target->GetKeys()) {
      auto iter = node->dispatch_dict_.find(k);
      if (iter != node->dispatch_dict_.end()) {
        func = iter->second;
        break;
      }
    }
  }

  if (func == nullptr) {
    CHECK(node->generic_func_ != nullptr) << "No generic function registered for " << node->name_;
    func = node->generic_func_;
  }

  func.CallPacked(args, ret);
}

TVM_REGISTER_GLOBAL("target.GenericFuncCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = GenericFunc(make_object<GenericFuncNode>());
});

TVM_REGISTER_GLOBAL("target.GenericFuncGetGlobal").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string func_name = args[0];
  *ret = GenericFunc::Get(func_name);
});

TVM_REGISTER_GLOBAL("target.GenericFuncSetDefault").set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  bool allow_override = args[2];

  generic_func.set_default(*func, allow_override);
});

TVM_REGISTER_GLOBAL("target.GenericFuncRegisterFunc").set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
  PackedFunc* func = new PackedFunc(args[1].operator PackedFunc());
  Array<runtime::String> tags = args[2];
  bool allow_override = args[3];

  std::vector<std::string> tags_vector;
  for (auto& tag : tags) {
    tags_vector.push_back(tag);
  }

  generic_func.register_func(tags_vector, *func, allow_override);
});

TVM_REGISTER_GLOBAL("target.GenericFuncCallFunc").set_body([](TVMArgs args, TVMRetValue* ret) {
  GenericFunc generic_func = args[0];
  TVMArgs func_args(&args.values[1], &args.type_codes[1], args.num_args - 1);

  generic_func.CallPacked(func_args, ret);
});

}  // namespace tvm
