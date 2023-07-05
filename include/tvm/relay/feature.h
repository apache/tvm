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
 * \file tvm/relay/feature.h
 * \brief Detect features used in Expr/Module.
 */
#ifndef TVM_RELAY_FEATURE_H_
#define TVM_RELAY_FEATURE_H_

#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>

#include <bitset>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Different kinds of relay feature a program might use. */
enum Feature : int {
  fVar = 0,
  fGlobalVar = 1,
  fConstant = 2,
  fTuple = 3,
  fTupleGetItem = 4,
  fFunction = 5,
  fOp = 6,
  fCall = 7,
  fLet = 8,
  fIf = 9,
  fRefCreate = 10,
  fRefRead = 11,
  fRefWrite = 12,
  fConstructor = 13,
  fMatch = 14,
  /*! \brief Whether any non-atom fragment of the program is shared, making the program a graph. */
  fGraph = 15,
  /*! \brief Whether there is local fixpoint in the program. */
  fLetRec = 16
};

constexpr size_t feature_count = 17;

/*!
 * \brief A finite set of Feature.
 */
class FeatureSet {
 public:
  FeatureSet(const FeatureSet&) = default;
  /*! \brief A singleton set containing a single Feature. */
  explicit FeatureSet(Feature ft) { bs_.set(static_cast<size_t>(ft)); }
  explicit FeatureSet(const tvm::Array<tvm::Integer>& ft) {
    for (Integer i : ft) {
      *this += Feature(i.IntValue());
    }
  }
  explicit operator Array<Integer>() const {
    Array<Integer> ret;
    for (size_t i = 0; i < feature_count; ++i) {
      if (bs_[i]) {
        ret.push_back(Integer(i));
      }
    }
    return ret;
  }
  /*! \brief A set that contain all the Feature. */
  static FeatureSet All() {
    FeatureSet fs;
    fs.bs_.flip();
    return fs;
  }
  /*! \brief The empty set. Contain no Feature. */
  static FeatureSet No() {
    FeatureSet fs;
    return fs;
  }
  template <typename T>
  FeatureSet& operator+=(const T& rhs) {
    bs_ |= FeatureSet(rhs).bs_;
    return *this;
  }
  /*! \brief Set union. */
  template <typename T>
  FeatureSet operator+(const T& rhs) const {
    FeatureSet fs(*this);
    fs += rhs;
    return fs;
  }
  template <typename T>
  FeatureSet& operator-=(const T& rhs) {
    bs_ &= ~(FeatureSet(rhs)).bs_;
    return *this;
  }
  /*! \brief Set difference. */
  template <typename T>
  FeatureSet operator-(const T& rhs) const {
    FeatureSet fs(*this);
    fs -= rhs;
    return fs;
  }
  /*!
   * \brief Is this a subset of rhs?
   *
   * \param rhs another FeatureSet.
   *
   * \return true only if this is a subset of rhs.
   */
  bool is_subset_of(const FeatureSet& rhs) const { return ((*this) - rhs).bs_.none(); }

  /*!
   * \brief return a string representation.
   */
  std::string ToString() const;

 private:
  std::bitset<feature_count> bs_;
  FeatureSet() = default;
  explicit FeatureSet(const std::bitset<feature_count>& bs) : bs_(bs) {}
};

/*!
 * \brief Calculate the feature of the program.
 *
 * \param expr The expression.
 *
 * \return The FeatureSet.
 */
FeatureSet DetectFeature(const RelayExpr& expr);

/*!
 * \brief Calculate the feature of the program.
 *
 * \param mod The module.
 *
 * \return The FeatureSet.
 */
FeatureSet DetectFeature(const IRModule& mod);

/*!
 * \brief Calculate the feature of the program.
 *
 * \param expr The expression.
 * \param mod The module.
 *
 * \return The FeatureSet.
 */
inline FeatureSet DetectFeature(const Expr& expr, const IRModule& mod) {
  return DetectFeature(expr) + DetectFeature(mod);
}

/*!
 * \brief Check the feature of the program.
 *
 * \param expr The expression.
 * \param fs The feature set of the program.
 */
void CheckFeature(const RelayExpr& expr, const FeatureSet& fs);

/*!
 * \brief Check the feature of the program.
 *
 * \param mod The module.
 * \param fs The feature set of the program.
 */
void CheckFeature(const IRModule& mod, const FeatureSet& fs);

/*!
 * \brief Check the feature of the program.
 *
 * \param expr The expression.
 * \param mod The module.
 * \param fs The feature set of the program.
 */
inline void CheckFeature(const RelayExpr& expr, const IRModule& mod, const FeatureSet& fs) {
  CheckFeature(expr, fs);
  CheckFeature(mod, fs);
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_FEATURE_H_
