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
 * \file lowered_func.cc
 */
#include <tvm/tir/lowered_func.h>

namespace tvm {
namespace tir {
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LoweredFuncNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LoweredFuncNode*>(node.get());
    p->stream << "LoweredFunc(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(LoweredFuncNode);
}  // namespace tir
}  // namespace tvm
