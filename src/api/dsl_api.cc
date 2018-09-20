/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of DSL API
 * \file dsl_api.cc
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/api_registry.h>
#include <tvm/attrs.h>
#include <vector>
#include <string>
#include <exception>
#include "../runtime/dsl_api.h"

namespace tvm {
namespace runtime {
/*! \brief entry to to easily hold returning information */
struct TVMAPIThreadLocalEntry {
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for retruning string */
  std::string ret_str;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<TVMAPIThreadLocalEntry> TVMAPIThreadLocalStore;

using TVMAPINode = NodePtr<Node>;

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
};

class DSLAPIImpl : public DSLAPI {
 public:
  void NodeFree(NodeHandle handle) const final {
    delete static_cast<TVMAPINode*>(handle);
  }
  void NodeTypeKey2Index(const char* type_key,
                        int* out_index) const final {
    *out_index = static_cast<int>(Node::TypeKey2Index(type_key));
  }
  void NodeGetTypeIndex(NodeHandle handle,
                        int* out_index) const final {
    *out_index = static_cast<int>(
        (*static_cast<TVMAPINode*>(handle))->type_index());
  }
  void NodeGetAttr(NodeHandle handle,
                   const char* key,
                   TVMValue* ret_val,
                   int* ret_type_code,
                   int* ret_success) const final {
    TVMRetValue rv;
    APIAttrGetter getter;
    TVMAPINode* tnode = static_cast<TVMAPINode*>(handle);
    getter.skey = key;
    getter.ret = &rv;
    if (getter.skey == "type_key") {
      ret_val->v_str = (*tnode)->type_key();
      *ret_type_code = kStr;
      *ret_success = 1;
      return;
    } else if (!(*tnode)->is_type<DictAttrsNode>()) {
      (*tnode)->VisitAttrs(&getter);
      *ret_success = getter.found_ref_object || rv.type_code() != kNull;
    } else {
      // specially handle dict attr
      DictAttrsNode* dnode = static_cast<DictAttrsNode*>(tnode->get());
      auto it = dnode->dict.find(key);
      if (it != dnode->dict.end()) {
        *ret_success = 1;
        rv = (*it).second;
      } else {
        *ret_success = 0;
      }
    }
    if (*ret_success) {
      if (rv.type_code() == kStr ||
          rv.type_code() == kTVMType) {
        TVMAPIThreadLocalEntry *e = TVMAPIThreadLocalStore::Get();
        e->ret_str = rv.operator std::string();
        *ret_type_code = kStr;
        ret_val->v_str = e->ret_str.c_str();
      } else {
        rv.MoveToCHost(ret_val, ret_type_code);
      }
    }
  }
  void NodeListAttrNames(NodeHandle handle,
                        int *out_size,
                        const char*** out_array) const final {
    TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
    ret->ret_vec_str.clear();
    TVMAPINode* tnode = static_cast<TVMAPINode*>(handle);
    APIAttrDir dir;
    dir.names = &(ret->ret_vec_str);

    if (!(*tnode)->is_type<DictAttrsNode>()) {
      (*tnode)->VisitAttrs(&dir);
    } else {
      // specially handle dict attr
      DictAttrsNode* dnode = static_cast<DictAttrsNode*>(tnode->get());
      for (const auto& kv : dnode->dict) {
        ret->ret_vec_str.push_back(kv.first);
      }
    }
    ret->ret_vec_charp.clear();
    for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
      ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
    }
    *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
    *out_size = static_cast<int>(ret->ret_vec_str.size());
  }
};

TVM_REGISTER_GLOBAL("dsl_api.singleton")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    static DSLAPIImpl impl;
    void* ptr = &impl;
    *rv = ptr;
  });
}  // namespace runtime
}  // namespace tvm
