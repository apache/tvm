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
 * \file conjunctive_normal_form.h
 *
 * \brief Centralized location for simplifying into specific forms
 */

#ifndef TVM_ARITH_CONJUNCTIVE_NORMAL_FORM_H_
#define TVM_ARITH_CONJUNCTIVE_NORMAL_FORM_H_

#include <tvm/tir/expr.h>

namespace tvm {
namespace arith {

class Analyzer;

/*! \brief Convert boolean expression to AND of ORs and simplify
 *
 * \param expr The PrimExpr to be simplified
 *
 * \param analyzer The analyzer with which to simplify
 *
 * \return The simplified expression
 */
PrimExpr SimplifyAsAndOfOrs(const PrimExpr& expr, Analyzer* analyzer);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_CONJUNCTIVE_NORMAL_FORM_H_
