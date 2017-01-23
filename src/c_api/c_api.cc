/*!
 *  Copyright (c) 2016 by Contributors
 * Implementation of C API
 * \file c_api.cc
 */
#include <tvm/c_api.h>
#include "./c_api_common.h"
#include "./c_api_registry.h"

/*! \brief entry to to easily hold returning information */
struct TVMAPIThreadLocalEntry {
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief argument stack */
  std::vector<tvm::APIVariantValue> arg_stack;
  /*! \brief return value */
  tvm::APIVariantValue ret_value;
  // clear calling stack
  inline void Clear() {
    arg_stack.clear();
    ret_value.sptr.reset();
  }
  inline void SetReturn(TVMValue* ret_val, int* ret_type_code);
};

using namespace tvm;

/*! \brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<TVMAPIThreadLocalEntry> TVMAPIThreadLocalStore;

using TVMAPINode = std::shared_ptr<Node>;

struct APIAttrGetter : public AttrVisitor {
  std::string skey;
  APIVariantValue* ret;
  bool found_node_ref{false};

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
  void Visit(const char* key, Type* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, std::string* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, NodeRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_node_ref = true;
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
  void Visit(const char* key, Type* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, std::string* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, NodeRef* value) final {
    names->push_back(key);
  }
};

int TVMListAPIFuncNames(int *out_size,
                            const char*** out_array) {
  API_BEGIN();
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
  ret->ret_vec_str = dmlc::Registry<APIFuncReg>::ListAllNames();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}

int TVMGetAPIFuncHandle(const char* fname,
                            APIFuncHandle* out) {
  API_BEGIN();
  const APIFuncReg* reg = dmlc::Registry<APIFuncReg>::Find(fname);
  CHECK(reg != nullptr) << "cannot find function " << fname;
  *out = (APIFuncHandle)reg;
  API_END();
}

int TVMGetAPIFuncInfo(APIFuncHandle handle,
                          const char **real_name,
                          const char **description,
                          int *num_doc_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions,
                          const char **return_type) {
  const auto *op = static_cast<const APIFuncReg *>(handle);
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();

  API_BEGIN();
  *real_name = op->name.c_str();
  *description = op->description.c_str();
  *num_doc_args = static_cast<int>(op->arguments.size());
  if (return_type) *return_type = nullptr;
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].name.c_str());
  }
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].type_info_str.c_str());
  }
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].description.c_str());
  }
  *arg_names = dmlc::BeginPtr(ret->ret_vec_charp);
  *arg_type_infos = dmlc::BeginPtr(ret->ret_vec_charp) + op->arguments.size();
  *arg_descriptions = dmlc::BeginPtr(ret->ret_vec_charp) + (op->arguments.size() * 2);
  API_END();
}

int TVMAPIPushStack(TVMValue arg,
                    int type_code) {
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->arg_stack.resize(ret->arg_stack.size() + 1);
  APIVariantValue& v = ret->arg_stack.back();

  v.type_code = type_code;
  switch (type_code) {
    case kInt: case kUInt: case kFloat: case kNull: {
      v.v_union = arg; break;
    }
    case kStr: {
      v.str = arg.v_str; break;
    }
    case kNodeHandle: {
      v.sptr = *static_cast<TVMAPINode*>(arg.v_handle); break;
    }
    default: LOG(FATAL) << "TVM API cannot take type " << TVMTypeCode2Str(type_code);
  }
  API_END_HANDLE_ERROR(ret->Clear());
}

int TVMAPIFuncCall(APIFuncHandle handle,
                   TVMValue* ret_val,
                   int* ret_type_code) {
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
  API_BEGIN();
  const auto *op = static_cast<const APIFuncReg *>(handle);
  op->body(ret->arg_stack, &(ret->ret_value));
  ret->SetReturn(ret_val, ret_type_code);
  ret->arg_stack.clear();
  API_END_HANDLE_ERROR(ret->Clear());
}

int TVMNodeFree(NodeHandle handle) {
  API_BEGIN();
  delete static_cast<TVMAPINode*>(handle);
  API_END();
}

int TVMNodeGetAttr(NodeHandle handle,
                   const char* key,
                   TVMValue* ret_val,
                   int* ret_type_code,
                   int* ret_success) {
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_value.type_code = kNull;
  APIAttrGetter getter;
  getter.skey = key;
  getter.ret = &(ret->ret_value);
  TVMAPINode* tnode = static_cast<TVMAPINode*>(handle);
  if (getter.skey == "type_key") {
    ret_val->v_str = (*tnode)->type_key();
    *ret_type_code = kStr;
    *ret_success = 1;
  } else {
    (*tnode)->VisitAttrs(&getter);
    if (ret->ret_value.type_code != kNull) {
      ret->SetReturn(ret_val, ret_type_code);
      *ret_success = 1;
    } else {
      *ret_success = getter.found_node_ref ? 1 : 0;
      *ret_type_code = kNull;
    }
  }
  API_END_HANDLE_ERROR(ret->Clear());
}

int TVMNodeListAttrNames(NodeHandle handle,
                         int *out_size,
                         const char*** out_array) {
  TVMAPIThreadLocalEntry *ret = TVMAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str.clear();
  TVMAPINode* tnode = static_cast<TVMAPINode*>(handle);
  APIAttrDir dir;
  dir.names = &(ret->ret_vec_str);
  (*tnode)->VisitAttrs(&dir);
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}


inline void TVMAPIThreadLocalEntry::SetReturn(TVMValue* ret_val,
                                              int* ret_type_code) {
  APIVariantValue& rv = ret_value;
  *ret_type_code = rv.type_code;
  if (rv.type_code == kNodeHandle) {
    if (rv.sptr.get() != nullptr) {
      ret_val->v_handle = new TVMAPINode(std::move(rv.sptr));
    } else {
      ret_val->v_handle = nullptr;
    }
  } else if (rv.type_code == kFuncHandle) {
    ret_val->v_handle = new runtime::PackedFunc::FType(std::move(rv.func));
  } else {
    *ret_val = rv.v_union;
  }
}
