/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_graph.cc
 * \brief C API related to Graph IR.
 */
#include "c_api_common.h"

#include <dmlc/json.h>
#include <nnvm/c_api.h>
#include <nnvm/graph.h>
#include <nnvm/graph_annotate.h>
#include <nnvm/op.h>
#include <nnvm/pass.h>
#include <nnvm/symbolic.h>

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

int NNGraphSetNodeEntryListAttr_(GraphHandle handle,
                                 const char* key,
                                 SymbolHandle list) {
  API_BEGIN();
  Symbol* s = static_cast<Symbol*>(list);
  Graph* g = static_cast<Graph*>(handle);
  g->attrs[std::string(key)]
      = std::make_shared<any>(s->outputs);
  API_END();
}

int NNGraphSetJSONAttr(GraphHandle handle,
                       const char* key,
                       const char* json_value) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string temp(json_value);
  std::istringstream is(temp);
  dmlc::JSONReader reader(&is);
  nnvm::any value;
  reader.Read(&value);
  g->attrs[std::string(key)] = std::make_shared<any>(std::move(value));
  API_END();
}

int NNGraphGetJSONAttr(GraphHandle handle,
                      const char* key,
                      const char** json_out,
                      int *success) {
  NNAPIThreadLocalEntry *ret = NNAPIThreadLocalStore::Get();
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string skey(key);
  auto it = g->attrs.find(skey);
  if (it != g->attrs.end()) {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.Write(*it->second.get());
    ret->ret_str = os.str();
    *json_out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *success = 0;
  }
  API_END();
}

int NNGraphHasJSONAttr(GraphHandle handle, const char* key, int* has) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string skey(key);
  *has = g->attrs.find(skey) != g->attrs.end();
  API_END();
}

int NNAnnotateGraph(GraphHandle src, nn_uint num_ops, const char** op_names,
                    GraphHandle* out) {
  nnvm::Graph* g = new nnvm::Graph();
  API_BEGIN();
  nnvm::Graph* src_graph = static_cast<Graph*>(src);
  std::unordered_set<std::string> op_name_set(op_names, op_names + num_ops);
  if (!op_name_set.empty()) {
    nnvm::op::AnnotationOpPropertyPtr property =
        std::make_shared<nnvm::op::DefaultAnnotationOpProperty>(op_name_set);
    src_graph->attrs["annotation_property"] =
        std::make_shared<nnvm::any>(std::move(property));
  }
  *g = ApplyPass(std::move(*src_graph), "AnnotateGraph");
  *out = g;
  API_END_HANDLE_ERROR(delete g);
}

int NNGraphApplyPasses(GraphHandle src,
                       nn_uint num_pass,
                       const char** pass_names,
                       GraphHandle *dst) {
  Graph* g = new Graph();
  API_BEGIN();
  std::vector<std::string> vpass;
  for (nn_uint i = 0; i < num_pass; ++i) {
    vpass.emplace_back(std::string(pass_names[i]));
  }
  *g = ApplyPasses(*static_cast<Graph*>(src), vpass);
  *dst = g;
  API_END_HANDLE_ERROR(delete g);
}
