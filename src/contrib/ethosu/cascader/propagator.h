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
 * \file src/contrib/ethosu/cascader/propagator.h
 * \brief Propagator class for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PROPAGATOR_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PROPAGATOR_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class Propagator;
class StripeConfig;

/*! \brief Node to represent a Propagator */
class PropagatorNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \return The transform matrix to apply to the StripeConfigs */
  const std::vector<std::vector<float>> GetTransform() const { return transform_; }
  /*! \return The offset vector to apply to the StripeConfigs */
  const std::vector<int> GetOffset() const { return offset_; }
  /*! \return The number of input dimensions */
  size_t GetInputDims() const { return offset_.size(); }
  /*! \return The number of output dimensions */
  size_t GetOutputDims() const { return transform_[0].size() - 1; }
  /*!
   * \brief Propagate a StripeConfig through the transform and offset matrices.
   * \param stripe_config The StripeConfig to propagate.
   * \return The transformed StripeConfig.
   * \note The propagation proceeds as follows:
   *
   * Both the stripe shape and extent have 1 appended to them (so they pick up
   * constant factors from the affine transform) and are then multiplied by the
   * transform matrix. The result is then ceil-rounded and has the trailing 1
   * stripped to give the new shape and extent.
   *
   * The strides has 0 appended to it (so it doesn't pick up constant factors)
   * and is then multiplied by the transform matrix. The trailing 0 is stripped.
   *
   * For the remaining three values we introduce the concept of the 'binarized'
   * transform matrix. This is the transform matrix but with every non-zero element
   * set to 1. It represents how axes get re-ordered as part of the propagation.
   *
   * [2,   0,   0, 1]            [1, 0, 0, 1]
   * [0,   0, 0.4, 2]  binarize  [0, 0, 1, 1]
   * [0, 1.5,   0, 0]   ---->    [0, 1, 0, 0]
   * [0,   0,   0, 1]            [0, 0, 0, 1]
   *
   * The order has 0 appended to it and is multiplied by the 'binarized' transform
   * matrix. The trailing 0 is then stripped.
   *
   * The stripes has 0 appended to it and multiplied by the 'binarized' transform
   * matrix. The trailing 0 is then stripped and any remaining 0 elements that
   * were introduced by the transform are set instead to 1.
   *
   * The stripe offset is multiplied by the 'binarized' transform matrix and is
   * then summed with the propagator offset.
   */
  StripeConfig propagate(const StripeConfig& stripe_config) const;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.Propagator";
  TVM_DECLARE_FINAL_OBJECT_INFO(PropagatorNode, Object);

 protected:
  friend class Propagator;

  /*! \brief The transform matrix to apply to the StripeConfigs */
  std::vector<std::vector<float>> transform_;
  /*! \brief The offset vector to apply to the StripeConfigs */
  std::vector<int> offset_;
};

/*!
 * \brief A class to transform StripeConfigs according to the data dependencies
 between Part outputs and inputs. The dependency is represented as an affine
 transformation matrix + an offset vector. Using this, an output StripeConfig
 can be propagated through a Part to arrive at the input StripeConfigs.
 * \note The transform matrix should be a 2D affine transform matrix.
 * As an example, consider a (1, 1, 2, 32) output stripe for an NHWC pooling
 * operation with a 3x3 pool size:
 *
 * [1, 0, 0, 0, 0]     [ 1]     [ 1]
 * [0, 1, 0, 0, 2]     [ 1]     [ 3]
 * [0, 0, 1, 0, 2]  x  [ 2]  =  [ 4]
 * [0, 0, 0, 1, 0]     [32]     [32]
 * [0, 0, 0, 0, 1]     [ 1]     [ 1]
 *
 * Using the appropriate affine matrix we see that the required input data to
 * produce that output stripe is a (1, 3, 4, 32) stripe. These matrices should
 * be derived for the Parts to relate input and output data dependencies.
 *
 * The offset is a 1D vector representing the first tensor element to read.
 * Often this is just the 0 element, but for an operator such as pad it may be
 * negative. For instance, a symmetric padding by 1 of a 2D tensor would require
 * the offset vector [-1, -1]. Additionally, positive offsets may be required
 * for operators like strided_slice where only part of a tensor is read from.
 */
class Propagator : public ObjectRef {
 public:
  Propagator(const std::vector<std::vector<float>>& transform, const std::vector<int>& offset);

  TVM_DEFINE_OBJECT_REF_METHODS(Propagator, ObjectRef, PropagatorNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PROPAGATOR_H_
