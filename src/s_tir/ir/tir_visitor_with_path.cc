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

#include "tir_visitor_with_path.h"

#include <tvm/s_tir/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tirx;
using AccessPath = ffi::reflection::AccessPath;

void TIRVisitorWithPath::Dispatch_(const SBlockNode* op, AccessPath path) {
  std::vector<std::variant<DefContext<Var>, DefContext<IterVar>, DefContext<BufferVar>>> context;

  {
    auto iter_path = path->Attr("iter_vars");
    for (size_t i = 0; i < op->iter_vars.size(); i++) {
      context.push_back(WithDef(op->iter_vars[i], iter_path->ArrayItem(i)));
    }
  }

  // Define alloc_buffers before visiting reads/writes, since reads/writes
  // may reference buffers from alloc_buffers (e.g. after transform_layout).
  {
    auto alloc_path = path->Attr("alloc_buffers");
    for (size_t i = 0; i < op->alloc_buffers.size(); i++) {
      auto buffer_path = alloc_path->ArrayItem(i);
      auto buf = op->alloc_buffers[i];
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  {
    auto match_path = path->Attr("match_buffers");
    for (size_t i = 0; i < op->match_buffers.size(); i++) {
      Visit(op->match_buffers[i]->source, match_path->ArrayItem(i)->Attr("source"));
      auto buf = op->match_buffers[i]->buffer;
      auto buffer_path = match_path->ArrayItem(i)->Attr("buffer");

      for (auto& def : WithMatchBufferDefs(buf, buffer_path)) {
        context.push_back(std::move(def));
      }
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  // Regions may use allocation and match-buffer definitions in this block.
  Visit(op->reads, path->Attr("reads"));
  Visit(op->writes, path->Attr("writes"));

  bind_scope_.WithNewScope([&]() { Visit(op->init, path->Attr("init")); });
  bind_scope_.WithNewScope([&]() { Visit(op->body, path->Attr("body")); });

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::Dispatch_(const SBlockRealizeNode* op, AccessPath path) {
  Visit(op->iter_values, path->Attr("iter_values"));
  Visit(op->predicate, path->Attr("predicate"));
  Visit(op->block, path->Attr("block"));
}

}  // namespace s_tir
}  // namespace tvm
