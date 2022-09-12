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
#include <tvm/arith/analyzer.h>
#include <tvm/script/ir_builder/tir/ir.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

using tvm::tir::IterVar;

PrimFuncFrame PrimFunc() {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = NullOpt;
  n->args.clear();
  n->ret_type = NullOpt;
  n->buffer_map.clear();
  n->preflattened_buffer_map.clear();
  n->attrs = NullOpt;
  n->env_threads.clear();
  n->root_alloc_buffers.clear();
  return PrimFuncFrame(n);
}

BlockFrame Block(String name, bool no_realize) {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->name = name;
  n->iter_vars.clear();
  n->reads = NullOpt;
  n->writes = NullOpt;
  n->init = NullOpt;
  n->alloc_buffers.clear();
  n->match_buffers.clear();
  n->annotations = NullOpt;
  n->iter_values.clear();
  n->predicate = NullOpt;
  n->no_realize = no_realize;
  return BlockFrame(n);
}

void Evaluate(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }
TVM_REGISTER_GLOBAL("script.ir_builder.tir.PrimFunc").set_body_typed(PrimFunc);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Block").set_body_typed(Block);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Evaluate").set_body_typed(Evaluate);
}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
