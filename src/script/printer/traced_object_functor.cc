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

#include <tvm/script/printer/traced_object_functor.h>

namespace tvm {
namespace script {
namespace printer {

const runtime::PackedFunc* GetDispatchFunctionForToken(const DispatchTable& table,
                                                       const String& token, uint32_t type_index) {
  auto it = table.find(token);
  if (it == table.end()) {
    return nullptr;
  }
  const std::vector<runtime::PackedFunc>& tab = it->second;
  if (type_index >= tab.size()) {
    return nullptr;
  }
  const PackedFunc* f = &tab[type_index];
  if (f->defined()) {
    return f;
  } else {
    return nullptr;
  }
}

const runtime::PackedFunc& GetDispatchFunction(const DispatchTable& dispatch_table,
                                               const String& token, uint32_t type_index) {
  if (const runtime::PackedFunc* pf =
          GetDispatchFunctionForToken(dispatch_table, token, type_index)) {
    return *pf;
  } else if (const runtime::PackedFunc* pf =
                 GetDispatchFunctionForToken(dispatch_table, kDefaultDispatchToken, type_index)) {
    // Fallback to function with the default dispatch token
    return *pf;
  } else {
    ICHECK(false) << "ObjectFunctor calls un-registered function on type: "
                  << runtime::Object::TypeIndex2Key(type_index) << " (token: " << token << ")";
    throw;
  }
}

void SetDispatchFunction(DispatchTable* dispatch_table, const String& token, uint32_t type_index,
                         runtime::PackedFunc f) {
  std::vector<runtime::PackedFunc>* table = &(*dispatch_table)[token];
  if (table->size() <= type_index) {
    table->resize(type_index + 1, nullptr);
  }
  runtime::PackedFunc& slot = (*table)[type_index];
  if (slot != nullptr) {
    ICHECK(false) << "Dispatch for type is already registered: "
                  << runtime::Object::TypeIndex2Key(type_index);
  }
  slot = f;
}
}  // namespace printer
}  // namespace script
}  // namespace tvm
