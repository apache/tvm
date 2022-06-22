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
 * \file metadata.cc
 * \brief Implementations of the compiler extensions for Metadata.
 */

#include "metadata.h"

#include <tvm/node/reflection.h>

namespace tvm {
namespace target {
namespace metadata {

TVM_REGISTER_REFLECTION_VTABLE(VisitableMetadataNode,
                               ::tvm::detail::ReflectionTrait<VisitableMetadataNode>)
    .set_creator([](const std::string&) -> ObjectPtr<Object> {
      return ::tvm::runtime::make_object<VisitableMetadataNode>();
    });

TVM_REGISTER_REFLECTION_VTABLE(VisitableTensorInfoNode,
                               ::tvm::detail::ReflectionTrait<VisitableTensorInfoNode>)
    .set_creator([](const std::string&) -> ObjectPtr<Object> {
      return ::tvm::runtime::make_object<VisitableTensorInfoNode>();
    });

TVM_REGISTER_REFLECTION_VTABLE(VisitableConstantInfoMetadataNode,
                               ::tvm::detail::ReflectionTrait<VisitableConstantInfoMetadataNode>)
    .set_creator([](const std::string&) -> ObjectPtr<Object> {
      return ::tvm::runtime::make_object<VisitableConstantInfoMetadataNode>();
    });

}  // namespace metadata
}  // namespace target
}  // namespace tvm
