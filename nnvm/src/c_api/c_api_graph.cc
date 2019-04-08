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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_graph.cc
 * \brief C API related to Graph IR.
 */
#include <nnvm/c_api.h>
#include <nnvm/op.h>
#include <nnvm/symbolic.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <dmlc/json.h>
#include "c_api_common.h"

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
