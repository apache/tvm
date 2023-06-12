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
#ifndef TVM_TIR_UTILS_H_
#define TVM_TIR_UTILS_H_

namespace tvm {
namespace tir {

/*!
 * \brief A helper macro to convert an sref to the statement it points to,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be cast
 * \param Type The type to be cast to, can be Block or For
 */
#define TVM_SREF_AS_OR_ERR(Result, SRef, Type) \
  SRef->StmtAs<Type>();                        \
  ICHECK(Result)

/*!
 * \brief A helper macro to convert an sref to the block it points to,
 *
 * Throws an internal error if downcasting fails.  The variable name
 * in the parent scope is used for the error message.
 *
 * \param SRef The SRef to be cast
 */
#define TVM_SREF_TO_BLOCK(SRef)                                                                    \
  [&]() {                                                                                          \
    auto result = TVM_SREF_AS_OR_ERR(result, (SRef), ::tvm::tir::BlockNode)                        \
                  << "TypeError: Expects StmtSRef `" << #SRef << "` points to `Block`, but gets: " \
                  << ((SRef)->stmt ? (SRef)->stmt->GetTypeKey() : "None");                         \
    return result;                                                                                 \
  }()

/*!
 * \brief A helper macro to convert an sref to the for-loop it points to
 *
 * Throws an internal error if downcasting fails.  The variable name
 * in the parent scope is used for the error message.
 *
 * \param SRef The SRef to be cast
 */
#define TVM_SREF_TO_FOR(SRef)                                                                     \
  [&]() {                                                                                         \
    auto result = TVM_SREF_AS_OR_ERR(result, (SRef), ::tvm::tir::ForNode)                         \
                  << "TypeError: Expects StmtSRef `" << #SRef << "` points to `Loop`, but gets: " \
                  << ((SRef)->stmt ? (SRef)->stmt->GetTypeKey() : "None");                        \
    return result;                                                                                \
  }()

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param From The ObjectRef to be downcast
 * \param Type The type to be downcast to
 */
#define TVM_TYPE_AS_OR_ERR(Result, From, Type) \
  From.as<Type>();                             \
  ICHECK(Result)

/*!
 * \brief Downcast a TVM ObjectRef to its corresponding container using `ObjectRef::as<Type>`,
 * throwing an internal error if downcast fails.
 * \param From The ObjectRef to be downcast
 * \param Type The type to be downcast to
 */
#define TVM_TYPE_AS(From, Type)                                                               \
  [&]() {                                                                                     \
    auto result = TVM_TYPE_AS_OR_ERR(result, (From), Type)                                    \
                  << "TypeError: Expects `" << #From << "` to have type `" << Type::_type_key \
                  << "`, but gets: " << ((From).defined() ? (From)->GetTypeKey() : "None");   \
    return result;                                                                            \
  }()

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_UTILS_H_
