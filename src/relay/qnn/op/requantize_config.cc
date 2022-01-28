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
 * \file src/relay/qnn/op/requantize_config.cc
 * \brief QNN requantize config.
 */

#include "./requantize_config.h"

#include <dmlc/thread_local.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <stack>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMRequantizeConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  RequantizeConfig default_config;

  /*! \brief The current build config context */
  std::stack<RequantizeConfig> context_stack;

  TVMRequantizeConfigThreadLocalEntry() : default_config(make_object<RequantizeConfigNode>(true)) {}
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMRequantizeConfigThreadLocalEntry>
    TVMRequantizeConfigThreadLocalStore;

void RequantizeConfig::EnterRequantizeConfigScope(const RequantizeConfig& build_config) {
  TVMRequantizeConfigThreadLocalEntry* entry = TVMRequantizeConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void RequantizeConfig::ExitRequantizeConfigScope() {
  TVMRequantizeConfigThreadLocalEntry* entry = TVMRequantizeConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

RequantizeConfig& RequantizeConfig::Current() {
  TVMRequantizeConfigThreadLocalEntry* entry = TVMRequantizeConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(RequantizeConfigNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RequantizeConfigNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* op = static_cast<const RequantizeConfigNode*>(ref.get());
      p->stream << "requantize_config(";
      p->stream << "rounding==" << op->get_rounding() << ", ";
      p->stream << "compute_dtype==" << op->get_compute_dtype();
      p->stream << ")";
    });

TVM_REGISTER_GLOBAL("relay._requantize._GetCurrentRequantizeConfig")
    .set_body_typed([]() -> RequantizeConfig { return RequantizeConfig::Current(); });

TVM_REGISTER_GLOBAL("relay._requantize._EnterRequantizeConfigScope")
    .set_body_typed(RequantizeConfig::EnterRequantizeConfigScope);

TVM_REGISTER_GLOBAL("relay._requantize._ExitRequantizeConfigScope")
    .set_body_typed(RequantizeConfig::ExitRequantizeConfigScope);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
