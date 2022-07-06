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

#ifndef TVM_META_SCHEDULE_FEATURE_EXTRACTOR_H_
#define TVM_META_SCHEDULE_FEATURE_EXTRACTOR_H_

#include <tvm/meta_schedule/measure_candidate.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*! \brief Extractor for features from measure candidates for use in cost model. */
class FeatureExtractorNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~FeatureExtractorNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Extract features from the given measure candidate.
   * \param context The tuning context for feature extraction.
   * \param candidates The measure candidates to extract features from.
   * \return The feature ndarray extracted.
   */
  virtual Array<tvm::runtime::NDArray> ExtractFrom(const TuneContext& context,
                                                   const Array<MeasureCandidate>& candidates) = 0;

  static constexpr const char* _type_key = "meta_schedule.FeatureExtractor";
  TVM_DECLARE_BASE_OBJECT_INFO(FeatureExtractorNode, Object);
};

/*! \brief The feature extractor with customized methods on the python-side. */
class PyFeatureExtractorNode : public FeatureExtractorNode {
 public:
  /*!
   * \brief Extract features from the given measure candidate.
   * \param context The tuning context for feature extraction.
   * \param candidates The measure candidates to extract features from.
   * \return The feature ndarray extracted.
   */
  using FExtractFrom = runtime::TypedPackedFunc<Array<tvm::runtime::NDArray>(
      const TuneContext& context, const Array<MeasureCandidate>& candidates)>;
  /*!
   * \brief Get the feature extractor as string with name.
   * \return The string of the feature extractor.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;

  /*! \brief The packed function to the `ExtractFrom` function. */
  FExtractFrom f_extract_from;
  /*! \brief The packed function to the `AsString` function. */
  FAsString f_as_string;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_extract_from` is not visited
    // `f_as_string` is not visited
  }

  Array<tvm::runtime::NDArray> ExtractFrom(const TuneContext& context,
                                           const Array<MeasureCandidate>& candidates) final;

  static constexpr const char* _type_key = "meta_schedule.PyFeatureExtractor";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyFeatureExtractorNode, FeatureExtractorNode);
};

/*!
 * \brief Managed reference to FeatureExtractorNode
 * \sa FeatureExtractorNode
 */
class FeatureExtractor : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a feature extractor that extracts features from each BufferStore
   * \param buffers_per_store The number of buffers in each BufferStore; Pad or truncate if
   * necessary.
   * \param arith_intensity_curve_num_samples The number of samples used in the arithmetic intensity
   * curve.
   * \param cache_line_bytes The number of bytes in a cache line.
   * \param extract_workload Whether to extract features in the workload in tuning context or not.
   * \return The feature extractor created.
   */
  TVM_DLL static FeatureExtractor PerStoreFeature(int buffers_per_store = 5,
                                                  int arith_intensity_curve_num_samples = 10,
                                                  int cache_line_bytes = 64,
                                                  bool extract_workload = false);
  /*!
   * \brief Create a feature extractor with customized methods on the python-side.
   * \param f_extract_from The packed function of `ExtractFrom`.
   * \param f_as_string The packed function of `AsString`.
   * \return The feature extractor created.
   */
  TVM_DLL static FeatureExtractor PyFeatureExtractor(
      PyFeatureExtractorNode::FExtractFrom f_extract_from,
      PyFeatureExtractorNode::FAsString f_as_string);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FeatureExtractor, ObjectRef, FeatureExtractorNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_FEATURE_EXTRACTOR_H_
