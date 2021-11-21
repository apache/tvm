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
 * \file relay/op/annotation/annotation.h
 * \brief Helpers for working with various 'annotation' attributes.
 */
#ifndef TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_
#define TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/ndarray.h>

#include <vector>

namespace tvm {
namespace relay {

/*! \brief Wraps \p data in a "stop_fusion" annotation. */
Expr StopFusion(Expr data);

/*! \brief Wraps \p data in a "cast_hint" annotation for \p dtype. */
Expr CastHint(Expr data, DataType dtype);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_
