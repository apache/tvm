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
 *  Copyright (c) 2019 by Contributors
 * \file src/tvm/ir/analysis.cc
 * \brief The set of Relay analysis passes written in C++.
 */
#include <tvm/relay/analysis.h>


namespace tvm {
namespace relay {

PointerAnalysisResult PointerAnalysisResultNode::make(Map<Expr, AbstractLocation> spawn,
                                                      Map<AbstractLocation, Expr> origin,
                                                      Map<Expr, Set<AbstractLocation>> contain,
                                                      Map<AbstractLocation, Set<Expr>> store) {
  NodePtr<PointerAnalysisResultNode> n = make_node<PointerAnalysisResultNode>();
  CHECK(spawn.defined());
  CHECK(origin.defined());
  CHECK(contain.defined());
  CHECK(store.defined());
  n->spawn = std::move(spawn);
  n->origin = std::move(origin);
  n->contain = std::move(contain);
  n->store = std::move(store);
  return PointerAnalysisResult(n);
}

AbstractLocation AbstractLocationNode::make(int id) {
  NodePtr<AbstractLocationNode> n = make_node<AbstractLocationNode>();
  n->id = id;
  return AbstractLocation(n);
}

}  // namespace relay
}  // namespace tvm
