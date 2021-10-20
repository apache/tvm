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
 * \file gradient.h
 * \brief Utility functions for Automatic Differentiation in Relay.
 */
#ifndef TVM_RELAY_TRANSFORMS_GRADIENT_H_
#define TVM_RELAY_TRANSFORMS_GRADIENT_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

#include <vector>

namespace tvm {
namespace relay {

inline Type GradRetType(const Function& f) {
  // if type annotations are provided, we will construct a ret type;
  // otherwise, leave it to be inferred
  if (!f->ret_type.defined()) {
    return Type();
  }
  std::vector<Type> vt;
  for (const auto& p : f->params) {
    if (!p->type_annotation.defined()) {
      return Type();
    }
    vt.push_back(p->type_annotation);
  }

  return TupleType({f->ret_type, TupleType(vt)});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_GRADIENT_H_
