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
 * \file src/contrib/ethosu/cascader/block_config.h
 * \brief BlockConfig object for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_BLOCK_CONFIG_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_BLOCK_CONFIG_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class BlockConfig;

/*! \brief Node to represent a BlockConfig */
class BlockConfigNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Get the shape of input block.
   * \return The input shape of the block config.
   */
  inline std::vector<int> GetInputBlockShape() const { return input_shape_; }

  /*!
   * \brief Get the shape of output block.
   * \return The output shape of the block config.
   */
  inline std::vector<int> GetOutputBlockShape() const { return output_shape_; }

  /*!
   * \brief Get the number of cycles required to output this block
   * \return The output cycles
   */
  inline int GetOutputCycles() const { return output_cycles_; }

  /*!
   * \brief Get the number of cycles required to compute this block
   * \return The compute cycles
   */
  inline int GetComputeCycles() const { return compute_cycles_; }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.BlockConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockConfigNode, Object);

 protected:
  friend class BlockConfig;

  /*! \brief The shape of the input block */
  std::vector<int> input_shape_;
  /*! \brief The shape of the output block */
  std::vector<int> output_shape_;
  /*! \brief Cycles required to compute this block */
  int compute_cycles_;
  /*! \brief Cycles required to output this block */
  int output_cycles_;
};

/*!
 * \brief An object that contains a an output block shape as well as the output and compute cycles
 * required to compute this block
 */
class BlockConfig : public ObjectRef {
 public:
  BlockConfig(const std::vector<int>& input_shape, const std::vector<int>& output_shape,
              int compute_cycles, int output_cycles);

  TVM_DEFINE_OBJECT_REF_METHODS(BlockConfig, ObjectRef, BlockConfigNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_BLOCK_CONFIG_H_
