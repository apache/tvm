/*!
 *  Copyright (c) 2019 by Contributors
 * \file int_op_overflow.h
 * \brief Utility functions to detect if an integer op will overflow.
 */
#ifndef TVM_ARITHMETIC_INT_OP_OVERFLOW_H_
#define TVM_ARITHMETIC_INT_OP_OVERFLOW_H_

#include <limits>

namespace tvm {
namespace arith {

/*!
 * \brief Check if an integer op with operand x, y will overflow.
 * \param x The left operand.
 * \param y The left operand.
 * \param min_value The minimum value of the domain.
 * \param max_value The maximum value of the domain.
 * \return Whether overflow can happen.
 * \tparam Op The integer operator.
 */
template<typename Op>
inline bool WillOverflow(int64_t x,
                         int64_t y,
                         int64_t min_value,
                         int64_t max_value) {
  return false;
}

template<>
bool WillOverflow<ir::Add>(int64_t x,
                           int64_t y,
                           int64_t min_value,
                           int64_t max_value) {
  if ((y > 0) && (x > max_value - y)) return true;
  if ((y < 0) && (x < min_value - y)) return true;
  return false;
}

template<>
bool WillOverflow<ir::Sub>(int64_t x,
                           int64_t y,
                           int64_t min_value,
                           int64_t max_value) {
  if ((y > 0) && (x < min_value + y)) return true;
  if ((y < 0) && (x > max_value + y)) return true;
  return false;
}

template<>
bool WillOverflow<ir::Mul>(int64_t x,
                           int64_t y,
                           int64_t min_value,
                           int64_t max_value) {
  if (y == 0) return false;
  if (y > 0) {
    if (x < min_value / y)  return true;
    if (x > max_value / y)  return true;
  } else {
    if (y == -1 && x == std::numeric_limits<int64_t>::min()) return true;
    if (x > min_value / y)  return true;
    if (x < max_value / y)  return true;
  }
  return false;
}

template<>
bool WillOverflow<ir::Mod>(int64_t x,
                           int64_t y,
                           int64_t min_value,
                           int64_t max_value) {
  return y == 0;
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_INT_OP_OVERFLOW_H_
