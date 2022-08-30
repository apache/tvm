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
 * \file src/contrib/ethosu/cascader/graph.h
 * \brief Graph objects (Tensor and Part) for the Ethos-U cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_GRAPH_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_GRAPH_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "block_config.h"
#include "propagator.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class Tensor;
class Part;
class StripeConfig;

/*!
 * \brief The buffering mode to use when realizing a tensor.
 * RECOMPUTE - The 'default' behaviour of TVM. Overlapping stripes will be recomputed.
 * ROLLING - Apply both the sliding window and storage folding optimizations to the tensor
 * realization.
 */
enum BufferMode { RECOMPUTE, ROLLING };

/*! \brief A struct to hold a Tensor Expression subgraph */
struct TESubgraph {
  /*! \brief The input te::Tensors to the subgraph */
  std::vector<te::Tensor> input_tensors;
  /*! \brief The output te::Tensor of the subgraph */
  te::Tensor output_tensor;
};

/*! \brief Node to hold performance information for a Part */
class PerformanceInfoNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \brief The cycles to compute a block */
  int64_t compute_cycles;
  /*! \brief The number of bytes read per input tensor */
  std::vector<int64_t> read_bytes;
  /*! \brief The number of bytes written to the output tensor */
  int64_t write_bytes;
  /*! \brief The block config used for this performance point */
  BlockConfig block_config;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.PerformanceInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(PerformanceInfoNode, Object);
};

/*!
 * \brief A class to hold the performance information for a Part.
 * \note The performance information for a Part is composed of 3 factors: the compute cycles,
 * the number of bytes read from each input tensor and the number of bytes written to the output
 * tensor. Bytes read/written is reported in favour of read/write bandwidth cycles so the
 * calculation of the performance information can be re-used with different memory homing.
 */
class PerformanceInfo : public ObjectRef {
 public:
  PerformanceInfo(int64_t compute_cycles, std::vector<int64_t> read_bytes, int64_t write_bytes,
                  BlockConfig block_config) {
    auto n = make_object<PerformanceInfoNode>();
    n->compute_cycles = compute_cycles;
    n->read_bytes = std::move(read_bytes);
    n->write_bytes = write_bytes;
    n->block_config = block_config;
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(PerformanceInfo, ObjectRef, PerformanceInfoNode);
};

/*! \brief Node to represent a Tensor */
class TensorNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \return The shape of the tensor */
  std::vector<int> GetShape() const { return shape_; }
  /*! \return The data type of the tensor */
  DataType GetDataType() const { return dtype_; }
  /*! \return Whether the tensor stores a constant value */
  bool IsConstant() const { return is_constant_; }
  /*! \return The compression ratio of the tensor */
  float GetCompressionRatio() const { return compression_ratio_; }
  /*! \return The producers of the tensor */
  const std::vector<Part> GetProducers() const { return producers_; }
  /*! \return The consumers of the tensor */
  const std::vector<Part> GetConsumers() const { return consumers_; }
  /*! \return The size of the tensor in bytes */
  int GetSize() const { return size_ * compression_ratio_; }

  /*! \brief Add a producer of the tensor */
  inline void AddProducer(const Part& part) { producers_.push_back(part); }
  /*! \brief Add a consumer of the tensor */
  inline void AddConsumer(const Part& part) { consumers_.push_back(part); }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.Tensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorNode, Object);

 protected:
  friend class Tensor;

  /*! \brief The shape of the tensor */
  std::vector<int> shape_;
  /*! \brief The data type of the tensor */
  DataType dtype_;
  /*! \brief Whether the tensor stores a constant value */
  bool is_constant_;
  /*! \brief The compression ratio of the tensor */
  float compression_ratio_;
  /*! \brief The producers of the tensor */
  std::vector<Part> producers_;
  /*! \brief The consumers of the tensor */
  std::vector<Part> consumers_;
  /*! \brief The size of the tensor in bytes */
  int size_;
};

/*!
 * \brief A class to describe a Tensor in a Cascader graph.
 * \note Cascader graphs consist of two object types: Tensors and Parts. This class
 * defines the Tensors which represent the tensors that are consumed and produced
 * as part of the graph. They are augmented with information about their 'kind'
 * (input/output/constant/intermediate), their default memory home (which memory they
 * are expected to be allocated in) and a compression ratio where applicable (weights
 * for instance are compressed).
 */
class Tensor : public ObjectRef {
 public:
  Tensor(const std::vector<int>& shape, DataType dtype, bool is_constant, float compression_ratio);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Tensor, ObjectRef, TensorNode);
};

/*! \brief Node to represent a Part */
class PartNode : public Object {
 public:
  virtual void VisitAttrs(AttrVisitor* v);

  /*! \return The TE subgraph represented by the Part */
  const TESubgraph GetSubgraph() const { return subgraph_; }
  /*! \return The output->input propagators */
  const std::vector<Propagator> GetPropagators() const { return propagators_; }
  /*! \return Whether the Part is inline */
  bool IsInline() const { return in_line_; }
  /*! \return The input tensors */
  const std::vector<Tensor> GetInputTensors() const { return input_tensors_; }
  /*! \return The output tensor */
  const Tensor GetOutputTensor() const { return output_tensor_; }

  /*! \brief Add a producer of the tensor */
  void SetInput(uint64_t input_index, const Tensor& input_tensor);
  /*! \brief Add a consumer of the tensor */
  void SetOutput(const Tensor& output_tensor) { output_tensor_ = output_tensor; }
  /*!
   * \brief Calculate the input stripe configs for a given output stripe config using the
   * Propagators. \param output_stripe_config The output stripe config to propagate. \return The
   * calculated input stripe configs.
   */
  std::vector<StripeConfig> CalculateInputStripeConfigs(const StripeConfig& output_stripe_config);
  /*!
   * \brief Get the preferred alignment in each axis for a stripe of the Part.
   * \note This is used to bias the selection of StripeConfigs towards those that are integer
   * multiples of a tensor intrinsic used to compute the Part.
   */
  virtual const std::vector<int> GetStripeAlignHint() const;
  /*!
   * \brief Get the performance information for a given output stripe config.
   * \param output_stripe_config The output stripe config to compute the performance for.
   * \param is_rolling Whether the output config should be computed as a rolling buffer.
   * \return The performance information containing the compute cycles and read/write bytes.
   */
  virtual const PerformanceInfo GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                                   BufferMode buffer_mode) = 0;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.Part";
  TVM_DECLARE_BASE_OBJECT_INFO(PartNode, Object);

 protected:
  friend class Part;

  /*! \brief The Tensor Expression subgraph represented by the Part */
  TESubgraph subgraph_;
  /*! \brief The output->input propagators */
  std::vector<Propagator> propagators_;
  /*! \brief Whether the Part is computed in-line */
  bool in_line_;
  /*! \brief The input tensors */
  std::vector<Tensor> input_tensors_;
  /*! \brief The output tensor */
  Tensor output_tensor_;
};

/*!
 * \brief A class to describe a Part in a Cascader graph.
 * \note Cascader graphs consist of two object types: Tensors and Parts. This class
 * defines the Parts which represent the operations which produce and consume Tensors.
 *
 * A Part can represent one or more Tensor Expression compute operations but the subgraph
 * it represents must have only a single output. Multiple TE compute operations should be
 * represented under a single Part if the intermediate tensors between them won't be
 * realized. This is a common pattern in Ethos-U where a sequence of TE compute operations
 * are used to represent a single hardware primitive operation.
 *
 * Parts contain a Propagator per input which describes how a given output stripe config
 * should be transformed into an input stripe config for each input. This is essential
 * to analyse both the performance of Parts (determining the data that will be read) and
 * in cascading Parts together (determining compatible stripe config choices).
 *
 * A Part can be marked as 'in_line', in which case it is assumed that it doesn't need to
 * allocate space for its output tensor.
 *
 * This is only a base class and concrete Parts must be derived from it, implementing a
 * function to model the performance of the Part as well as to determine its compute
 * quantum.
 */
class Part : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Part, ObjectRef, PartNode);
};

/*! \brief Node to represent a CascaderGraph */
class CascaderGraphNode : public Object {
 public:
  CascaderGraphNode() {}
  CascaderGraphNode(std::vector<Tensor> input_tensors, std::vector<Tensor> output_tensors);

  void VisitAttrs(AttrVisitor* v);

  /*! \return The input Tensors of the CascaderGraph */
  std::vector<Tensor> GetInputTensors() const { return input_tensors_; }
  /*! \return The output Tensors of the CascaderGraph */
  std::vector<Tensor> GetOutputTensors() const { return output_tensors_; }
  /*! \return The order of the Parts in the CascaderGraph */
  std::vector<Part> GetPartOrder() const { return part_order_; }
  /*!
   * \brief Get the ID of a Part in the CascaderGraph.
   * \param part The Part to get the ID of.
   * \return The ID of the Part in the CascaderGraph.
   * \note Each Part is given a unique ID within the CascaderGraph.
   */
  int GetPartID(const Part& part) const;
  /*!
   * \brief Get the ID of a Tensor in the CascaderGraph.
   * \param tensor The Tensor to get the ID of.
   * \return The ID of the Tensor in the CascaderGraph.
   * \note Each Tensor is given a unique ID within the CascaderGraph.
   */
  int GetTensorID(const Tensor& tensor) const;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.CascaderGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(CascaderGraphNode, Object);

 protected:
  /*!
   * \brief Initialize the CascaderGraph by defining a topological ordering.
   * \note This will traverse the Parts and Tensors using a depth-first
   * visiting pattern and use the traversal order to initialize both the
   * 'order' vectors and the ID maps. The order vectors define the ordering
   * that the cascader expects the CascaderGraph to be executed in, but reversed.
   * The ID maps assign a unique integer ID to each Part and Tensor corresponding
   * to their position in their respective order vector.
   */
  void Init_();

  /*! \brief The input Tensors of the CascaderGraph */
  std::vector<Tensor> input_tensors_;
  /*! \brief The output Tensors of the CascaderGraph */
  std::vector<Tensor> output_tensors_;
  /*! \brief The order of the Tensors in the CascaderGraph */
  std::vector<Tensor> tensor_order_;
  /*! \brief The order of the Parts in the CascaderGraph */
  std::vector<Part> part_order_;
  /*! \brief A map between Parts in the CascaderGraph and their IDs */
  std::unordered_map<Part, int, ObjectPtrHash, ObjectPtrEqual> part_id_map_;
  /*! \brief A map between Tensors in the CascaderGraph and their IDs */
  std::unordered_map<Tensor, int, ObjectPtrHash, ObjectPtrEqual> tensor_id_map_;
};

/*!
 * \brief A class to describe a graph of Parts and Tensors used by the cascader.
 * \note This class describes a graph consisting of two object types: Tensors and Parts.
 * It defines a topological ordering on the graph such that each Part and Tensor has a
 * position in the ordering. This ordering is used by the Plan and Proposal generation
 * algorithms. It is also the ordering the Parts are expected to be executed in.
 *
 * In addition to defining an ordering, the Parts and Tensors are also all given unique
 * IDs which they can be referred to by.
 */
class CascaderGraph : public ObjectRef {
 public:
  CascaderGraph(std::vector<Tensor> input_tensors, std::vector<Tensor> output_tensors);

  TVM_DEFINE_OBJECT_REF_METHODS(CascaderGraph, ObjectRef, CascaderGraphNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_GRAPH_H_
