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
 * \file src/relay/backend/executor.cc
 * \brief Executor Registry
 */

#include <tvm/relay/executor.h>

#include "../../node/attr_registry.h"
namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ExecutorNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ExecutorNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const Executor& executor = Downcast<Executor>(obj);
      p->stream << executor->name;
      p->stream << executor->attrs;
    });

/**********  Registry-related code  **********/

using ExecutorRegistry = AttrRegistry<ExecutorRegEntry, Executor>;

Executor Executor::Create(String name, Map<String, ObjectRef> attrs) {
  const ExecutorRegEntry* reg = ExecutorRegistry::Global()->Get(name);
  if (reg == nullptr) {
    throw Error("Executor \"" + name + "\" is not defined");
  }

  for (const auto& kv : attrs) {
    if (!reg->key2vtype_.count(kv.first)) {
      throw Error("Attribute \"" + kv.first + "\" is not available on this Executor");
    }
    std::string expected_type = reg->key2vtype_.at(kv.first).type_key;
    std::string actual_type = kv.second->GetTypeKey();
    if (expected_type != actual_type) {
      throw Error("Attribute \"" + kv.first + "\" should have type \"" + expected_type +
                  "\" but instead found \"" + actual_type + "\"");
    }
  }

  for (const auto& kv : reg->key2default_) {
    if (!attrs.count(kv.first)) {
      attrs.Set(kv.first, kv.second);
    }
  }

  return Executor(name, DictAttrs(attrs));
}

Array<String> Executor::ListExecutors() { return ExecutorRegistry::Global()->ListAllNames(); }

Map<String, String> Executor::ListExecutorOptions(const String& name) {
  Map<String, String> options;
  const ExecutorRegEntry* reg = ExecutorRegistry::Global()->Get(name);
  if (reg == nullptr) {
    throw Error("Executor \"" + name + "\" is not defined");
  }
  for (const auto& kv : reg->key2vtype_) {
    options.Set(kv.first, kv.second.type_key);
  }
  return options;
}

ExecutorRegEntry& ExecutorRegEntry::RegisterOrGet(const String& name) {
  return ExecutorRegistry::Global()->RegisterOrGet(name);
}

/**********  Register Executors and options  **********/

TVM_REGISTER_EXECUTOR("aot")
    .add_attr_option<Bool>("link-params", Bool(true))
    .add_attr_option<Bool>("unpacked-api")
    .add_attr_option<String>("interface-api")
    .add_attr_option<Integer>("workspace-byte-alignment")
    .add_attr_option<Integer>("constant-byte-alignment");

TVM_REGISTER_EXECUTOR("graph").add_attr_option<Bool>("link-params", Bool(false));

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("relay.backend.CreateExecutor").set_body_typed(Executor::Create);
TVM_REGISTER_GLOBAL("relay.backend.GetExecutorAttrs").set_body_typed([](const Executor& executor) {
  return executor->attrs->dict;
});

TVM_REGISTER_GLOBAL("relay.backend.ListExecutors").set_body_typed(Executor::ListExecutors);
TVM_REGISTER_GLOBAL("relay.backend.ListExecutorOptions")
    .set_body_typed(Executor::ListExecutorOptions);

}  // namespace relay
}  // namespace tvm
