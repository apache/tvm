/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_graph.cc
 * \brief C API related to Graph IR.
 */
#include <nnvm/c_api.h>
#include <nnvm/op.h>
#include <nnvm/symbolic.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include "./c_api_common.h"

using namespace nnvm;

int NNGraphCreate(SymbolHandle symbol, GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  g->outputs = static_cast<Symbol*>(symbol)->outputs;
  *graph = g;
  API_END_HANDLE_ERROR(delete g);
}

int NNGraphFree(GraphHandle handle) {
  API_BEGIN();
  delete static_cast<Graph*>(handle);
  API_END();
}

int NNGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol) {
  Symbol* s = new Symbol();
  API_BEGIN();
  s->outputs = static_cast<Graph*>(graph)->outputs;
  *symbol = s;
  API_END_HANDLE_ERROR(delete s);
}

int NNGraphSetStrAttr(GraphHandle handle,
                      const char* key,
                      const char* value) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  g->attrs[std::string(key)] = std::make_shared<any>(std::string(value));
  API_END();
}

int NNGraphGetStrAttr(GraphHandle handle,
                      const char* key,
                      const char** out,
                      int *success) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string skey(key);
  auto it = g->attrs.find(skey);
  if (it != g->attrs.end()) {
    const std::string& str = nnvm::get<std::string>(*it->second.get());
    *out = str.c_str();
    *success = 1;
  } else {
    *success = 0;
  }
  API_END();
}

int NNGraphApplyPass(GraphHandle src,
                     nn_uint num_pass,
                     const char** pass_names,
                     GraphHandle *dst) {
  Graph* g = new Graph();
  API_BEGIN();
  std::vector<std::string> vpass;
  for (nn_uint i = 0; i < num_pass; ++i) {
    vpass.emplace_back(std::string(pass_names[i]));
  }
  *g = ApplyPass(*static_cast<Graph*>(src), vpass);
  *dst = g;
  API_END_HANDLE_ERROR(delete g);
}
