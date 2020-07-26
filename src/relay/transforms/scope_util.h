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
 *
 * \file tvm/relay/_transforms/pass_util.h
 * \brief Utilities for writing passes
 */
#ifndef TVM_RELAY_TRANSFORMS_SCOPE_UTIL_H_
#define TVM_RELAY_TRANSFORMS_SCOPE_UTIL_H_

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "let_list.h"

#include <memory>
#include <unordered_map>

namespace tvm {
namespace relay {

struct ScopeNode;
using Scope = std::shared_ptr<ScopeNode>;

/* Invariant: when parent is null level is 0
 *
 * Invariant: when parent is not null level is 1 + parent->level
 */
struct ScopeNode {
  size_t level;
  Scope parent;
  std::shared_ptr<LetList> ll = std::make_shared<LetList>();
  explicit ScopeNode(const Scope& parent) : level(1 + parent->level), parent(parent) {}
  ScopeNode() : level(0) {}
};

Scope ChildScope(const Scope& s) { return std::make_shared<ScopeNode>(s); }

Scope LCA(Scope lhs, Scope rhs) {
  while (lhs != rhs) {
    if (lhs->level > rhs->level) {
      lhs = lhs->parent;
    } else if (lhs->level < rhs->level) {
      rhs = rhs->parent;
    } else {
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_SCOPE_UTIL_H_
