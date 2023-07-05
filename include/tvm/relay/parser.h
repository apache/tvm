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
#ifndef TVM_RELAY_PARSER_H_
#define TVM_RELAY_PARSER_H_

#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <string>

namespace tvm {
namespace relay {

using MetaTable = Map<String, Array<ObjectRef>>;

IRModule ParseModule(const std::string& file_name, const std::string& file_content,
                     const Optional<IRModule>& init_module = Optional<IRModule>(),
                     const MetaTable& init_meta_table = MetaTable());

/*!
 * \brief This pass pretty-prints mod then parses it back so as to establish spans and sources
 * for all Relay sub-expressions. This improves error and debugging diagnostics downstream for
 * modules constructed programaticaly rather than textually.
 */
tvm::transform::Pass AnnotateSpans();

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PARSER_H_
