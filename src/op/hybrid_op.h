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
 * \brief Helper utilities to implement hybrid_op.
 * \file hybrid_op.h
 */
#ifndef TVM_OP_HYBRID_OP_H_
#define TVM_OP_HYBRID_OP_H_

#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../pass/ir_util.h"
#include "../pass/arg_binder.h"
#include "../schedule/message_passing.h"


namespace tvm {
namespace op {

/*!
 * \brief Find all the iteration variables in the given statement body.
 * \param stmt The body to be inspected.
 */
std::vector<IterVar> GatherLoopVars(Stmt stmt);

/*!
 * \brief Replace the tensor reference (especially in Provide's) in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
Stmt ReplaceProvideTensor(Stmt stmt,
                          const std::unordered_map<Tensor, Tensor>& replace);

/*!
 * \brief Apply the schedule manipulation on the function body.
 * \param stmt The statement to be processed.
 * \param dom_map The extents of the iterative variables may be used.
 * \param stage The schedule information to be applied.
 */
Stmt ApplySchedule(const Stage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map, Stmt stmt);

/*!
 * \brief Apply loop splits and fuses in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param dom_map The extents of the iterative variables may be used.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopShapes(const Stage &stage,
                     const std::unordered_map<IterVar, Range>& dom_map, Stmt stmt);


/*!
 * \brief Apply loop annotation in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param rebased The map specifies the rebase, a.k.a rename, relationship of these variables.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopAnnotations(const Stage &stage,
                          const std::unordered_map<IterVar, IterVar>& rebased, Stmt stmt);

/*!
 * \brief Apply loop order in the schedule on the function body.
 * \param stage The schedule information to be applied.
 * \param dom_map The extents of the iterative variables may be used.
 * \param rebased The map specifies the rebase, a.k.a rename, relationship of these variables.
 * \param stmt The statement to be processed.
 */
Stmt ApplyLoopOrder(const Stage &stage,
                    const std::unordered_map<IterVar, Range> &dom_map,
                    const std::unordered_map<IterVar, IterVar> &rebased, Stmt stmt);

}  // namespace op
}  // namespace tvm

#endif  // TVM_OP_HYBRID_OP_H_
