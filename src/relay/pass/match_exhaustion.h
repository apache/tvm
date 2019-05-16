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
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass/match_exhaustion.h
 * \brief Header of definitions for match exhaustion.
 */

#ifndef TVM_RELAY_PASS_MATCH_EXHAUSTION_H_
#define TVM_RELAY_PASS_MATCH_EXHAUSTION_H_

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/pass.h>

namespace tvm {
namespace relay {

/*!
 * \brief Tests whether all match expressions in the given program
 * are exhaustive.
 * \return Returns a list of cases that are not handled by the match
 * expression.
 */
Array<Pattern> CheckMatchExhaustion(const Match& match, const Module& mod);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_MATCH_EXHAUSTION_H_
