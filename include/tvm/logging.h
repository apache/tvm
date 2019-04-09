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
 * \file tvm/logging.h
 * \brief logging utilities on top of dmlc-core
 */
#ifndef TVM_LOGGING_H_
#define TVM_LOGGING_H_

// a technique that enables overriding macro names on the number of parameters. This is used
// to define other macros below
#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

/*!
 * \brief COND_X calls COND_X_N where N is the number of parameters passed to COND_X
 * X can be any of CHECK_GE, CHECK_EQ, CHECK, or LOG (defined dmlc-core/include/dmlc/logging.h.)
 * COND_X (but not COND_X_N) are supposed to be used outside this file.
 * The first parameter of COND_X (and therefore, COND_X_N), which we call 'quit_on_assert',
 * is a boolean. The rest of the parameters of COND_X is the same as the parameters of X.
 * quit_on_assert determines the overall behaviour of COND_X. If it's true COND_X
 * quits the program on assertion failure. If it's false, then it moves on and somehow reports
 * the assertion failure back to the macro caller in an appropriate manner (e.g, 'return false'
 * in a function, or 'continue' or 'break' in a loop)
 * The default behavior when quit_on_assertion is false, is to 'return false'. If this is not
 * desirable, the macro caller can pass one more last parameter to COND_X to tell COND_X what
 * to do when when quit_on_assertion is false and the assertion fails.
 *
 * Rationale: These macros were designed to implement functions that have two behaviours
 * in a concise way. Those behaviours are quitting on assertion failures, or trying to
 * move on from assertion failures. Note that these macros hide lots of control flow in them,
 * and therefore, makes the logic of the whole code slightly harder to understand. However,
 * in pieces of code that use these macros frequently, it will significantly shorten the
 * amount of code needed to be read, and we won't need to clutter the main logic of the
 * function by repetitive control flow structure. The first problem
 * mentioned will be improved over time as the developer gets used to the macro.
 *
 * Here is an example of how to use it
 * \code
 * bool f(..., bool quit_on_assertion) {
 *   int a = 0, b = 0;
 *   ...
 *   a = ...
 *   b = ...
 *   // if quit_on_assertion is true, if a==b, continue, otherwise quit.
 *   // if quit_on_assertion is false, if a==b, continue, otherwise 'return false' (default behaviour)
 *   COND_CHECK_EQ(quit_on_assertion, a, b) << "some error message when  quiting"
 *   ...
 *   for (int i = 0; i < N; i++) {
 *     a = ...
 *     b = ...
 *     // if quit_on_assertion is true, if a==b, continue, otherwise quit.
 *     // if quit_on_assertion is false, if a==b, continue, otherwise 'break' (non-default
 *     // behaviour, therefore, has to be explicitly specified)
 *     COND_CHECK_EQ(quit_on_assertion, a, b, break) << "some error message when  quiting"
 *   }
 * }
 * \endcode
 */
#define COND_CHECK_GE(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_GE_5, COND_CHECK_GE_4, COND_CHECK_GE_3)(__VA_ARGS__)
#define COND_CHECK_EQ(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_EQ_5, COND_CHECK_EQ_4, COND_CHECK_EQ_3)(__VA_ARGS__)
#define COND_CHECK(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_5, COND_CHECK_4, COND_CHECK_3, COND_CHECK_2)(__VA_ARGS__)
#define COND_LOG(...) \
  GET_MACRO(__VA_ARGS__, COND_LOG_5, COND_LOG_4, COND_LOG_3, COND_LOG_2)(__VA_ARGS__)

// Not supposed to be used by users directly.
#define COND_CHECK_OP(quit_on_assert, x, y, what, op) \
  if (!quit_on_assert) { \
    if (!((x) op (y))) \
      what; \
  } \
  else /* NOLINT(*) */ \
    CHECK_##op(x, y)

#define COND_CHECK_EQ_4(quit_on_assert, x, y, what) COND_CHECK_OP(quit_on_assert, x, y, what, ==)
#define COND_CHECK_GE_4(quit_on_assert, x, y, what) COND_CHECK_OP(quit_on_assert, x, y, what, >=)

#define COND_CHECK_3(quit_on_assert, x, what) \
  if (!quit_on_assert) { \
    if (!(x)) \
      what; \
  } \
  else /* NOLINT(*) */ \
    CHECK(x)

#define COND_LOG_3(quit_on_assert, x, what) \
  if (!quit_on_assert) { \
    what; \
  } \
  else /* NOLINT(*) */ \
    LOG(x)

#define COND_CHECK_EQ_3(quit_on_assert, x, y) COND_CHECK_EQ_4(quit_on_assert, x, y, return false)
#define COND_CHECK_GE_3(quit_on_assert, x, y) COND_CHECK_GE_4(quit_on_assert, x, y, return false)
#define COND_CHECK_2(quit_on_assert, x) COND_CHECK_3(quit_on_assert, x, return false)
#define COND_LOG_2(quit_on_assert, x) COND_LOG_3(quit_on_assert, x, return false)

#endif   // TVM_LOGGING_H_
