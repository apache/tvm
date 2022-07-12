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
 * \file src/relay/collage/cost.h
 * \brief Represents the estimated cost of a candidate partition.
 */
#ifndef TVM_RELAY_COLLAGE_COST_H_
#define TVM_RELAY_COLLAGE_COST_H_

#include <tvm/runtime/logging.h>

#include <cmath>
#include <limits>
#include <string>

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief The assumed cost for a candidate partition. Generally average execution time in seconds.
 * However other cost functions are possible, for example to introduce a penalty for high memory
 * use, etc.
 */
class Cost {
 public:
  Cost() = delete;

  static Cost Zero() { return Cost(0.0); }

  /*!
   * \brief Returns the distinguished 'invalid' cost signaling a candidate partition is not
   * supported by the intended target, for example because the sub-graph has an unsupported operator
   * or the intermediate memory required exceeds some system limit.
   */
  static Cost Invalid() { return Cost(std::numeric_limits<double>::infinity()); }

  bool is_invalid() const { return std::isinf(value_) && value_ > 0.0; }

  /*!
   * \brief Returns the distinguished 'unknown' cost, signaling fixed priorities should be used to
   * choose the best partitions. This can be used to disable tuning and fallback to fixed rules,
   * much as TVM will use an un-tuned kernel if no tuning records are available.
   */
  static Cost Unknown() { return Cost(std::numeric_limits<double>::quiet_NaN()); }

  bool is_unknown() const { return std::isnan(value_); }

  /*! \brief Returns cost with given finite, non-negative value. */
  static Cost Value(double value) {
    ICHECK(!std::isnan(value) && !std::isinf(value) && value >= 0.0);
    return Cost(value);
  }

  bool is_value() const { return !std::isnan(value_) && !std::isinf(value_); }

  double value() const {
    ICHECK(is_value());
    return value_;
  }

  /*! \brief Return true if the less-than relation is defined for this and that. */
  bool are_comparable(Cost that) const { return !std::isnan(value_) && !std::isnan(that.value_); }

  /*! \brief Returns sum of this and that. */
  Cost operator+(Cost that) const { return Cost(value_ + that.value_); }

  /*! \brief Returns difference of this and that. */
  Cost operator-(Cost that) const { return Cost(value_ - that.value_); }

  /*! \brief Returns true if this is cheaper than that, assuming they are comparable. */
  bool operator<(Cost that) const { return value_ < that.value_; }

  std::string ToString() const;

 private:
  explicit Cost(double value) : value_(value) {}

  /*!
   * \brief Non-negative value or:
   *   - +inf if candidate partition is not feasible.
   *   - NaN if candidate partition has an unknown cost (priority may be used to break ties).
   */
  double value_ = 0.0;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COST_H_
