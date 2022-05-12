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
 * \file src/relay/backend/runtime.cc
 * \brief Runtime Registry
 */

#include <tvm/relay/runtime.h>

#include "../../node/attr_registry.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(RuntimeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RuntimeNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const Runtime& runtime = Downcast<Runtime>(obj);
      p->stream << runtime->name;
    });

/**********  Registry-related code  **********/

using RuntimeRegistry = AttrRegistry<RuntimeRegEntry, Runtime>;

Runtime Runtime::Create(String name, Map<String, ObjectRef> attrs) {
  const RuntimeRegEntry* reg = RuntimeRegistry::Global()->Get(name);
  if (reg == nullptr) {
    throw Error("Runtime \"" + name + "\" is not defined");
  }

  for (const auto& kv : attrs) {
    if (!reg->key2vtype_.count(kv.first)) {
      throw Error("Attribute \"" + kv.first + "\" is not available on this Runtime");
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

  return Runtime(name, DictAttrs(attrs));
}

Array<String> Runtime::ListRuntimes() { return RuntimeRegistry::Global()->ListAllNames(); }

Map<String, String> Runtime::ListRuntimeOptions(const String& name) {
  Map<String, String> options;
  const RuntimeRegEntry* reg = RuntimeRegistry::Global()->Get(name);
  if (reg == nullptr) {
    throw Error("Runtime \"" + name + "\" is not defined");
  }
  for (const auto& kv : reg->key2vtype_) {
    options.Set(kv.first, kv.second.type_key);
  }
  return options;
}

RuntimeRegEntry& RuntimeRegEntry::RegisterOrGet(const String& name) {
  return RuntimeRegistry::Global()->RegisterOrGet(name);
}

/**********  Register Runtimes and options  **********/

TVM_REGISTER_RUNTIME(kTvmRuntimeCrt).add_attr_option<Bool>("system-lib");

TVM_REGISTER_RUNTIME(kTvmRuntimeCpp).add_attr_option<Bool>("system-lib");

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("relay.backend.CreateRuntime").set_body_typed(Runtime::Create);
TVM_REGISTER_GLOBAL("relay.backend.GetRuntimeAttrs").set_body_typed([](const Runtime& runtime) {
  return runtime->attrs->dict;
});

TVM_REGISTER_GLOBAL("relay.backend.ListRuntimes").set_body_typed(Runtime::ListRuntimes);
TVM_REGISTER_GLOBAL("relay.backend.ListRuntimeOptions").set_body_typed(Runtime::ListRuntimeOptions);

}  // namespace relay
}  // namespace tvm
