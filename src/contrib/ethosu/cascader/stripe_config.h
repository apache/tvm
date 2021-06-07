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
 * \file src/contrib/ethosu/cascader/stripe_config.h
 * \brief StripeConfig object for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_STRIPE_CONFIG_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_STRIPE_CONFIG_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <map>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class StripeConfig;
class PropagatorNode;

/*! \brief Node to represent a StripeConfig */
class StripeConfigNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Get the shape of the stripe config.
   * \return The shape of the stripe config.
   * \note The shape refers to the size of the stripes in each dimension.
   */
  inline std::vector<int> GetShape() const { return shape_; }
  /*!
   * \brief Get the extent of the stripe config.
   * \return The extent of the stripe config.
   * \note The extent refers to the extent over which a StripeConfig operates.
   * Specifically, it is the extent in each axis between the lowest value read
   * by a stripe and the highest value read by a stripe.
   */
  inline std::vector<int> GetExtent() const { return extent_; }
  /*!
   * \brief Get the strides of the stripe config.
   * \return The strides of the stripe config.
   * \note The strides refer to the stride between stripes in each axis.
   * The strides are represented as a float rather than an int to account for
   * cases of 'fractional striding'. This may happen, for instance, with an
   * upscaling operation where elements of the affine transformation matrix
   * are not integers. In this case we can't simply round the strides as the
   * error will compound when we need to multiply the strides by the number of
   * stripes along a given axis.
   */
  inline std::vector<float> GetStrides() const { return strides_; }
  /*!
   * \brief Get the order of the stripe config.
   * \return The order of the stripe config.
   * \note The order refers to order in which the axes are iterated over.
   * The first (outermost) axis is labelled as 1 with the rest increasing
   * according to the axis' position. Any axis labelled with 0 isn't iterated over.
   * For example, [1, 3, 2] would mean axis 0 is the outermost iteration axis,
   * then axis 2, then finally axis 1.
   */
  inline std::vector<int> GetOrder() const { return order_; }
  /*!
   * \brief Get the stripes of the stripe config.
   * \return The stripes of the stripe config.
   * \note The stripes refer to the number of stripes in each axis.
   * There must be at least one stripe in any given axis.
   */
  inline std::vector<int> GetStripes() const { return stripes_; }
  /*!
   * \brief Get the offset of the stripe config.
   * \return The offset of the stripe config.
   * \note The offset refers to the offset of the first stripe
   * from the first element of the tensor. For example, in a 2D padding operation
   * that is padding by 1 in every dimension, the offset would be [-1, -1].
   */
  inline std::vector<int> GetOffset() const { return offset_; }
  /*! \return The hash of the StripeConfigNode */
  size_t GetHash() const { return hash_; }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.StripeConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(StripeConfigNode, Object);

 protected:
  friend class StripeConfig;
  friend class PropagatorNode;

  /*! \brief Compute the hash of the StripeConfigNode */
  void ComputeHash_();

  /*! \brief The shape of the stripes */
  std::vector<int> shape_;
  /*! \brief The extent of region to stripe over */
  std::vector<int> extent_;
  /*! \brief The strides of the stripes */
  std::vector<float> strides_;
  /*! \brief The order of the striping axes */
  std::vector<int> order_;
  /*! \brief The number of stripes in each axis */
  std::vector<int> stripes_;
  /*! \brief The offset of the first stripe */
  std::vector<int> offset_;
  /*! \brief The hash of the StripeConfigNode */
  std::size_t hash_{0};
};

/*!
 * \brief An object to describe how a tensor should be computed as a series
 of n-dimensional tiles, or 'stripes'.
 * \note The StripeConfig is a verbose way of specifying how to tile a tensor.
 * We can imagine taking a 2D tensor of size (12, 12) and wanting to compute
 * it in tiles of (4, 4). The tile is referred to as a stripe here to generalize
 * this to n-dimensional tiles.
 *
 * The size of that stripe in each axis is the 'shape'. The strides is how far
 * you should move between stripes, so also (4, 4) for a simple non-overlappping
 * tiling. However, we explore some overlapping scheduling options so shape != strides
 * in general. The 'extent' is simply (12, 12), the region over which we're conducting
 * our tiling.
 *
 * The 'order' tells us which axis to iterate over first and which second and the
 * 'stripes' tells us how many stripes we need to compute in each of those axes.
 *
 * Finally, the 'offset' tells us where to start the first stripe. In this simple
 * case the offset is just (0, 0), but in something like a padding operation we
 * may want to start from a negative index, which is captured by the offset.
 */
class StripeConfig : public ObjectRef {
 public:
  StripeConfig(const std::vector<int>& shape, const std::vector<int>& extent,
               const std::vector<float>& strides, const std::vector<int>& order,
               const std::vector<int>& stripes, const std::vector<int>& offset);
  /*!
   * \brief Check if two StripeConfigs are equals to each other.
   * \param other StripeConfig to be checked.
   * \return Whether the two StripeConfigs equal each other.
   */
  bool operator==(const StripeConfig& other) const;

  TVM_DEFINE_OBJECT_REF_METHODS(StripeConfig, ObjectRef, StripeConfigNode);
};

/*!
 * \brief Count the number of stripes of each shape that are executed for a given
 StripeConfig.
 * \param stripe_config The StripeConfig to count the stripes for.
 * \param enable_sliding_window Whether to assume the sliding window optimization.
 * \return A map between stripe shapes and the number of stripes of that shape that need
 * executing.
 */
std::map<std::vector<int>, int> CountStripes(const StripeConfig& stripe_config,
                                             bool enable_sliding_window);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

// Hash and equal function for StripeConfig
namespace std {

/*! \brief The equal_to function for tvm::contrib::ethosu::cascader::StripeConfig */
template <>
struct equal_to<::tvm::contrib::ethosu::cascader::StripeConfig> {
  bool operator()(const ::tvm::contrib::ethosu::cascader::StripeConfig& lhs,
                  const ::tvm::contrib::ethosu::cascader::StripeConfig& rhs) const {
    return lhs == rhs;
  }
};

/*! \brief The hash function for tvm::contrib::ethosu::cascader::StripeConfig */
template <>
struct hash<::tvm::contrib::ethosu::cascader::StripeConfig> {
  std::size_t operator()(
      const ::tvm::contrib::ethosu::cascader::StripeConfig& stripe_config) const {
    return stripe_config->GetHash();
  }
};

}  // namespace std

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_STRIPE_CONFIG_H_
