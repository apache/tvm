/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file annotate.h
 * \brief The functions to make Relax annotate operator calls for SmoothQuant quantization
 *        algorithm.
 */
#ifndef TVM_RELAX_OP_TENSOR_ANNOTATE_H_
#define TVM_RELAX_OP_TENSOR_ANNOTATE_H_

#include <tvm/relax/attrs/annotate.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Multiply elements from a tensor by a scale (if it operates like "multiply") or pass input
 * as is (if it operates like "identity").
 * \param x The source tensor.
 * \param m Scale multiplier.
 * \param k Kind of argument to be annotated. Can be one of: activation or weight.
 * \param md Execution mode for the op. Can be one of: "identity", "multiply" or "quantize".
 * \return The computed result.
 */
Expr smooth(Expr x, Expr m, int k, String md);

/*!
 * \brief Helper op, that does the following transformation:
 *  abs()->max()->squeeze() for the input tensor.
 * \param x The source tensor.
 * \param k Kind of argument to be annotated. Can be one of: activation or weight.
 * \return The computed result.
 */
Expr absmax(Expr x, int k);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_ANNOTATE_H_
