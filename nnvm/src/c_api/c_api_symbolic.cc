/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_symbolic.cc
 * \brief C API related to symbolic graph compsition.
 */
#include <nnvm/c_api.h>
#include <nnvm/op.h>
#include <nnvm/symbolic.h>
#include "./c_api_common.h"

using namespace nnvm;

int NNSymbolListAtomicSymbolCreators(nn_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<Op>::List();
  *out_size = static_cast<nn_uint>(vec.size());
  *out_array = (AtomicSymbolCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int NNSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                const char **name,
                                const char **description,
                                nn_uint *num_doc_args,
                                const char ***arg_names,
                                const char ***arg_type_infos,
                                const char ***arg_descriptions,
                                const char **return_type) {
  const Op *op = static_cast<const Op *>(creator);

  API_BEGIN();
  *name = op->name.c_str();
  *description = op->description.c_str();
  *num_doc_args = 0;
  API_END();
}

int NNSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                               nn_uint num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  const Op* op = static_cast<const Op*>(creator);
  std::unordered_map<std::string, std::string> kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    kwargs.insert({std::string(keys[i]), std::string(vals[i])});
  }
  *s = Symbol::CreateFunctor(op, std::move(kwargs));
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}

int NNSymbolCreateVariable(const char *name, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = Symbol::CreateVariable(name);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNSymbolCreateGroup(nn_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  Symbol *s = new Symbol();
  Symbol **sym_arr = (Symbol**)symbols; // NOLINT(*)
  API_BEGIN();
  std::vector<Symbol> syms;
  for (nn_uint i = 0; i < num_symbols; ++i) {
    syms.push_back(*sym_arr[i]);
  }
  *s = Symbol::CreateGroup(syms);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNSymbolGetOutput(SymbolHandle symbol,
                      nn_uint index,
                      SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = (*static_cast<Symbol*>(symbol))[index];
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<Symbol*>(symbol)->GetInternals();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNSymbolFree(SymbolHandle symbol) {
  API_BEGIN();
  delete static_cast<Symbol*>(symbol);
  API_END();
}

int NNSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<const Symbol*>(symbol)->Copy();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNSymbolPrint(SymbolHandle symbol, const char **out_str) {
  Symbol *s = static_cast<Symbol*>(symbol);
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  s->Print(os);
  ret->ret_str = os.str();
  *out_str = (ret->ret_str).c_str();
  API_END();
}

int NNSymbolGetAttr(SymbolHandle symbol,
                    const char* key,
                    const char** out,
                    int* success) {
  Symbol *s = static_cast<Symbol*>(symbol);
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  if (s->GetAttr(key, &(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
  }
  API_END();
}

int NNSymbolSetAttrs(SymbolHandle symbol,
                     nn_uint num_param,
                     const char** keys,
                     const char** vals) {
  Symbol *s = static_cast<Symbol*>(symbol);
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    kwargs.emplace_back(
        std::make_pair(std::string(keys[i]), std::string(vals[i])));
  }
  s->SetAttrs(kwargs);
  API_END();
}

int NNSymbolListAttrs(SymbolHandle symbol,
                      int option,
                      nn_uint *out_size,
                      const char*** out) {
  Symbol *s = static_cast<Symbol*>(symbol);
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  std::unordered_map<std::string, std::string> attr =
      std::move(s->ListAttrs(static_cast<Symbol::ListAttrOption>(option)));  // NOLINT(*)

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.clear();
  for (const auto& kv : attr) {
    attr_list.push_back(kv.first);
    attr_list.push_back(kv.second);
  }
  *out_size = attr.size();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int NNSymbolListInputNames(SymbolHandle symbol,
                           int option,
                           nn_uint *out_size,
                           const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(
      s->ListInputNames(Symbol::ListInputOption(option)));
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<nn_uint>(ret->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int NNSymbolListOutputNames(SymbolHandle symbol,
                            nn_uint *out_size,
                            const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(s->ListOutputNames());
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<nn_uint>(ret->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int NNSymbolCompose(SymbolHandle sym,
                    const char *name,
                    nn_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  API_BEGIN();
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  std::string& s_name = ret->ret_str;
  std::unordered_map<std::string, const Symbol*>& kwargs
      = ret->kwarg_symbol;
  kwargs.clear();
  if (name != nullptr) {
    s_name = name;
  } else {
    s_name.clear();
  }
  Symbol* s = static_cast<Symbol*>(sym);
  if (keys == nullptr && num_args != 0) {
    kwargs.clear();
    array_view<const Symbol*> parg(
        (Symbol**)args, (Symbol**)args + num_args); // NOLINT(*)
    s->Compose(parg, kwargs, s_name);
  } else {
    for (nn_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = (Symbol*)args[i];  //  NOLINT(*)
    }
    s->Compose(array_view<const Symbol*>(), kwargs, s_name);
  }
  API_END();
}
