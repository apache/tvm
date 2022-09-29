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
 * \file install_debug_spans_ops.h
 * \brief List of stmts and exprs supported by the debug info pass
 */

#ifndef TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_
#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_

#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_EXPRS \
  X(Call)                                                      \
  X(Add)                                                       \
  X(Sub)                                                       \
  X(Mul)                                                       \
  X(Div)                                                       \
  X(Mod)                                                       \
  X(FloorDiv)                                                  \
  X(FloorMod)                                                  \
  X(Min)                                                       \
  X(Max)                                                       \
  X(EQ)                                                        \
  X(NE)                                                        \
  X(LT)                                                        \
  X(LE)                                                        \
  X(GT)                                                        \
  X(GE)                                                        \
  X(And)                                                       \
  X(Or)                                                        \
  X(Reduce)                                                    \
  X(Cast)                                                      \
  X(Not)                                                       \
  X(Select)                                                    \
  X(Ramp)                                                      \
  X(Broadcast)                                                 \
  X(Shuffle)                                                   \
  X(IntImm)                                                    \
  X(FloatImm)                                                  \
  X(StringImm)

#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_STMTS \
  X(AttrStmt)                                                  \
  X(IfThenElse)                                                \
  X(LetStmt)                                                   \
  X(For)                                                       \
  X(While)                                                     \
  X(Allocate)                                                  \
  X(AllocateConst)                                             \
  X(DeclBuffer)                                                \
  X(Store)                                                     \
  X(BufferStore)                                               \
  X(BufferRealize)                                             \
  X(AssertStmt)                                                \
  X(ProducerStore)                                             \
  X(ProducerRealize)                                           \
  X(Prefetch)                                                  \
  X(SeqStmt)                                                   \
  X(Evaluate)                                                  \
  X(BlockRealize)

#endif  // TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_
