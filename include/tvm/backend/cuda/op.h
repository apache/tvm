/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file tvm/backend/cuda/op.h
 * \brief CUDA-owned operators.
 */
#ifndef TVM_BACKEND_CUDA_OP_H_
#define TVM_BACKEND_CUDA_OP_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace backend {
namespace cuda {

/*!
 * \name Frontend-only NVIDIA IKET annotations
 *
 * These CUDA device intrinsics have opaque call effects and no registered attrs
 * schema or attrs defaults. Marker, range-start, range-end and range-push
 * annotations have a registered variadic operand tail; the Python convenience
 * builders and IKET lowering use it for one optional scalar payload. Below, C denotes
 * tvm.backend.cuda.op and T is the TIRx script namespace.
 *
 * Match a structurally valid marker call and inspect its operands:
 * \code
 * if (const auto* call = expr.as<tvm::CallNode>();
 *     call && call->op.same_as(tvm::backend::cuda::iket_mark_op())) {
 *   tvm::Expr name = call->args[0];
 *   if (call->args.size() > 1) {
 *     tvm::Expr payload = call->args[1];
 *   }
 *   tvm::Type result_type = call->ty;
 * }
 * \endcode
 * The other getters use the same identity-matching pattern with the operand
 * layouts documented below.
 * \{
 */

/*!
 * \brief Get the NVIDIA IKET marker operator, tirx.cuda.iket_mark.
 * Python: C.cuda_iket_mark(name, payload=None).
 * Script: T.cuda.iket.mark(name, payload=None).
 * Operands: [0] name, the marker name; [1...] args, the optional payload tail.
 * Result: void. Effects: opaque marker annotation.
 */
TVM_DLL const Op& iket_mark_op();

/*!
 * \brief Get the NVIDIA IKET token-range start operator, tirx.cuda.iket_range_start.
 * Python: C.cuda_iket_range_start(name, payload=None).
 * Script: T.cuda.iket.range_start(name, payload=None).
 * Operands: [0] name, the range name; [1...] args, the optional payload tail.
 * Result: uint32 range token. Effects: opaque range-start annotation.
 */
TVM_DLL const Op& iket_range_start_op();

/*!
 * \brief Get the NVIDIA IKET token-range end operator, tirx.cuda.iket_range_end.
 * Python: C.cuda_iket_range_end(token, payload=None).
 * Script: T.cuda.iket.range_end(token, payload=None).
 * Operands: [0] token, an IntExpr identifying the range to end;
 * [1...] args, the optional payload tail. IKET lowering requires a uint32 token.
 * Result: void. Effects: opaque range-end annotation.
 */
TVM_DLL const Op& iket_range_end_op();

/*!
 * \brief Get the NVIDIA IKET stack-range push operator, tirx.cuda.iket_range_push.
 * Python: C.cuda_iket_range_push(name, payload=None).
 * Script: T.cuda.iket.range_push(name, payload=None).
 * Operands: [0] name, the range name; [1...] args, the optional payload tail.
 * Result: void. Effects: opaque stack-range push annotation.
 */
TVM_DLL const Op& iket_range_push_op();

/*!
 * \brief Get the NVIDIA IKET stack-range pop operator, tirx.cuda.iket_range_pop.
 * Python: C.cuda_iket_range_pop(). Script: T.cuda.iket.range_pop().
 * Operands: none; IKET lowering enforces the empty operand list.
 * Result: void. Effects: opaque stack-range pop annotation.
 */
TVM_DLL const Op& iket_range_pop_op();

/*!
 * \brief Get the NVIDIA IKET sentinel-token operator, tirx.cuda.iket_sentinel_token.
 * Python: C.cuda_iket_sentinel_token(name). Script: T.cuda.iket.sentinel_token(name).
 * Operands: [0] name, the event name.
 * Result: uint32 sentinel token. Registered effects: opaque. The sentinel carries
 * token-flow identity; IKET lowering emits no runtime event for it.
 */
TVM_DLL const Op& iket_sentinel_token_op();

/*! \} */

}  // namespace cuda
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_CUDA_OP_H_
