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
 *  Implementation of DSL API
 * \file dsl_api.cc
 */
#include <dmlc/logging.h>
#include <tvm/api_registry.h>
#include <tvm/attrs.h>
#include <tvm/expr.h>
#include <vector>
#include <string>

namespace tvm {
namespace runtime {

struct APIAttrGetter : public AttrVisitor {
  std::string skey;
  TVMRetValue* ret;
  bool found_ref_object{false};

  void Visit(const char* key, double* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, int64_t* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, uint64_t* value) final {
    CHECK_LE(value[0], static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "cannot return too big constant";
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, int* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, bool* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, void** value) final {
    if (skey == key) *ret = static_cast<void*>(value[0]);
  }
  void Visit(const char* key, Type* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, std::string* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, NodeRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
  void Visit(const char* key, runtime::ObjectRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
};

struct APIAttrDir : public AttrVisitor {
  std::vector<std::string>* names;

  void Visit(const char* key, double* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, int64_t* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, uint64_t* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, bool* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, int* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, void** value) final {
    names->push_back(key);
  }
  void Visit(const char* key, Type* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, std::string* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, NodeRef* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, runtime::ObjectRef* value) final {
    names->push_back(key);
  }
};

struct NodeAPI {
  static void GetAttr(TVMArgs args, TVMRetValue* ret) {
    NodeRef ref = args[0];
    Node* tnode = const_cast<Node*>(ref.get());
    APIAttrGetter getter;
    getter.skey = args[1].operator std::string();
    getter.ret = ret;

    bool success;
    if (getter.skey == "type_key") {
      *ret = tnode->GetTypeKey();
      success = true;
    } else if (!tnode->IsInstance<DictAttrsNode>()) {
      tnode->VisitAttrs(&getter);
      success = getter.found_ref_object || ret->type_code() != kNull;
    } else {
      // specially handle dict attr
      DictAttrsNode* dnode = static_cast<DictAttrsNode*>(tnode);
      auto it = dnode->dict.find(getter.skey);
      if (it != dnode->dict.end()) {
        success = true;
        *ret = (*it).second;
      } else {
        success = false;
      }
    }
    if (!success) {
      LOG(FATAL) << "AttributeError: " << tnode->GetTypeKey()
                 << " object has no attributed " << getter.skey;
    }
  }

  static void ListAttrNames(TVMArgs args, TVMRetValue* ret) {
    NodeRef ref = args[0];
    Node* tnode = const_cast<Node*>(ref.get());
    auto names = std::make_shared<std::vector<std::string> >();
    APIAttrDir dir;
    dir.names = names.get();

    if (!tnode->IsInstance<DictAttrsNode>()) {
      tnode->VisitAttrs(&dir);
    } else {
      // specially handle dict attr
      DictAttrsNode* dnode = static_cast<DictAttrsNode*>(tnode);
      for (const auto& kv : dnode->dict) {
        names->push_back(kv.first);
      }
    }

    *ret = PackedFunc([names](TVMArgs args, TVMRetValue *rv) {
        int64_t i = args[0];
        if (i == -1) {
          *rv = static_cast<int64_t>(names->size());
        } else {
          *rv = (*names)[i];
        }
      });
  }
};

TVM_REGISTER_GLOBAL("_NodeGetAttr")
.set_body(NodeAPI::GetAttr);

TVM_REGISTER_GLOBAL("_NodeListAttrNames")
.set_body(NodeAPI::ListAttrNames);

}  // namespace runtime
}  // namespace tvm
