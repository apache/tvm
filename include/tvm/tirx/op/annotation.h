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
 * \file tvm/tirx/op/annotation.h
 * \brief Annotation operations for TIRx.
 */
#ifndef TVM_TIRX_OP_ANNOTATION_H_
#define TVM_TIRX_OP_ANNOTATION_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Marker where a transform should replace generated kernel initialization.
 */
TVM_DLL const Op& kernel_replace_point_op();

/*!
 * \brief Annotate a predicate not be considered as target condition of loop partition.
 *
 * Arguments, in order:
 * - args[0]: predicate, The predicate.
 */
TVM_DLL const Op& ignore_loop_partition_op();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_ANNOTATION_H_
