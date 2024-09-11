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
 * \file tvm/ir/replace_global_vars.h
 *
 * \brief A utility to replace GlobalVar instances across all TVM IR
 * types in an IRMdoule.
 */
#ifndef TVM_IR_REPLACE_GLOBAL_VARS_H_
#define TVM_IR_REPLACE_GLOBAL_VARS_H_

#include <tvm/ir/module.h>

namespace tvm {
namespace transform {

/*!
 * \brief Replace GlobalVar instances across any IR type.
 *
 * \param mod The module to update
 *
 * \param replacements The map, where each entry maps from an old
 * `GlobalVar` to the new `GlobalVar` that should replace it.
 *
 * \return The updated IRModule
 */
TVM_DLL IRModule ReplaceGlobalVars(IRModule mod, Map<GlobalVar, GlobalVar> replacements);

struct GlobalVarReplacer {
  using FType = NodeFunctor<BaseFunc(const ObjectRef&, Map<GlobalVar, GlobalVar>)>;
  TVM_DLL static FType& vtable() {
    static FType inst;
    return inst;
  }
};

}  // namespace transform
}  // namespace tvm

#endif  // TVM_IR_REPLACE_GLOBAL_VARS_H_
