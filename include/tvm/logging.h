/*!
 *  Copyright (c) 2018 by Contributors
 * \file logging.h
 * \brief logging utilities on top of dmlc-core
 */
#ifndef TVM_LOGGING_H_
#define TVM_LOGGING_H_

// a technique that enables overriding macro names on the number of parameters
#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define COND_CHECK_GE(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_GE_5, COND_CHECK_GE_4, COND_CHECK_GE_3)(__VA_ARGS__)
#define COND_CHECK_EQ(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_EQ_5, COND_CHECK_EQ_4, COND_CHECK_EQ_3)(__VA_ARGS__)
#define COND_CHECK(...) \
  GET_MACRO(__VA_ARGS__, COND_CHECK_5, COND_CHECK_4, COND_CHECK_3, COND_CHECK_2)(__VA_ARGS__)
#define COND_LOG(...) \
  GET_MACRO(__VA_ARGS__, COND_LOG_5, COND_LOG_4, COND_LOG_3, COND_LOG_2)(__VA_ARGS__)

// lint triggers an error for "else" and "CHECK_##op(x,y)" being on the same line
// 1) I can't put CHECK in braces because I want to write things like
// COND_CHECK_OP  << "error message"
// 2) I can't put // NOLINT in the middle of the macro definition to avoid the lint error
// What can I do?
#define COND_CHECK_OP(cond, x, y, what, op) \
  if (!cond) { \
    if (!((x) op (y))) \
      what; \
  } \
  else \
    CHECK_##op(x, y)

#define COND_CHECK_EQ_4(cond, x, y, what) COND_CHECK_OP(cond, x, y, what, ==)
#define COND_CHECK_GE_4(cond, x, y, what) COND_CHECK_OP(cond, x, y, what, >=)

#define COND_CHECK_3(cond, x, what) \
  if (!cond) { \
    if (!(x)) \
      what; \
  } \
  else \
    CHECK(x)

#define COND_LOG_3(cond, x, what) \
  if (!cond) { \
    what; \
  } \
  else \
    LOG(x)

#define COND_CHECK_EQ_3(cond, x, y) COND_CHECK_EQ_4(cond, x, y, return false)
#define COND_CHECK_GE_3(cond, x, y) COND_CHECK_GE_4(cond, x, y, return false)
#define COND_CHECK_2(cond, x) COND_CHECK_3(cond, x, return false)
#define COND_LOG_2(cond, x) COND_LOG_3(cond, x, return false)

#endif   // TVM_LOGGING_H_
