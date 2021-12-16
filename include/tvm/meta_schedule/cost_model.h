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

#ifndef TVM_META_SCHEDULE_COST_MODEL_H_
#define TVM_META_SCHEDULE_COST_MODEL_H_

#include <tvm/meta_schedule/search_strategy.h>

#include <vector>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*! \brief Cost model. */
class CostModelNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~CostModelNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Load the cost model from given file location.
   * \param path The file path.
   */
  virtual void Load(const String& path) = 0;

  /*!
   * \brief Save the cost model to given file location.
   * \param path The file path.
   */
  virtual void Save(const String& path) = 0;

  /*!
   * \brief Update the cost model given running results.
   * \param tune_context The tuning context.
   * \param candidates The measure candidates.
   * \param results The running results of the measure candidates.
   */
  virtual void Update(const TuneContext& tune_context, const Array<MeasureCandidate>& candidates,
                      const Array<RunnerResult>& results) = 0;

  /*!
   * \brief Predict the normalized score (the larger the better) of given measure candidates.
   * \param tune_context The tuning context.
   * \param candidates The measure candidates.
   * \return The predicted normalized score.
   */
  virtual std::vector<double> Predict(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) = 0;

  static constexpr const char* _type_key = "meta_schedule.CostModel";
  TVM_DECLARE_BASE_OBJECT_INFO(CostModelNode, Object);
};

/*! \brief The cost model with customized methods on the python-side. */
class PyCostModelNode : public CostModelNode {
 public:
  /*!
   * \brief Load the cost model from given file location.
   * \param path The file path.
   */
  using FLoad = runtime::TypedPackedFunc<void(String)>;
  /*!
   * \brief Save the cost model to given file location.
   * \param path The file path.
   */
  using FSave = runtime::TypedPackedFunc<void(String)>;
  /*!
   * \brief Update the cost model given running results.
   * \param tune_context The tuning context.
   * \param candidates The measure candidates.
   * \param results The running results of the measure candidates.
   * \return Whether cost model was updated successfully.
   */
  using FUpdate = runtime::TypedPackedFunc<void(const TuneContext&, const Array<MeasureCandidate>&,
                                                const Array<RunnerResult>&)>;
  /*!
   * \brief Predict the running results of given measure candidates.
   * \param tune_context The tuning context.
   * \param candidates The measure candidates.
   * \param p_addr The address to save the the estimated running results.
   */
  using FPredict = runtime::TypedPackedFunc<void(const TuneContext&, const Array<MeasureCandidate>&,
                                                 void* p_addr)>;
  /*!
   * \brief Get the cost model as string with name.
   * \return The string representation of the cost model.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;

  /*! \brief The packed function to the `Load` function. */
  FLoad f_load;
  /*! \brief The packed function to the `Save` function. */
  FSave f_save;
  /*! \brief The packed function to the `Update` function. */
  FUpdate f_update;
  /*! \brief The packed function to the `Predict` function. */
  FPredict f_predict;
  /*! \brief The packed function to the `AsString` function. */
  FAsString f_as_string;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_load` is not visited
    // `f_save` is not visited
    // `f_update` is not visited
    // `f_predict` is not visited
    // `f_as_string` is not visited
  }

  void Load(const String& path) {
    ICHECK(f_load != nullptr) << "PyCostModel's Load method not implemented!";
    f_load(path);
  }

  void Save(const String& path) {
    ICHECK(f_save != nullptr) << "PyCostModel's Save method not implemented!";
    f_save(path);
  }
  void Update(const TuneContext& tune_context, const Array<MeasureCandidate>& candidates,
              const Array<RunnerResult>& results) {
    ICHECK(f_update != nullptr) << "PyCostModel's Update method not implemented!";
    f_update(tune_context, candidates, results);
  }

  std::vector<double> Predict(const TuneContext& tune_context,
                              const Array<MeasureCandidate>& candidates) {
    ICHECK(f_predict != nullptr) << "PyCostModel's Predict method not implemented!";
    std::vector<double> result(candidates.size(), 0.0);
    f_predict(tune_context, candidates, result.data());
    return result;
  }

  static constexpr const char* _type_key = "meta_schedule.PyCostModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyCostModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to CostModelNode
 * \sa CostModelNode
 */
class CostModel : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a feature extractor with customized methods on the python-side.
   * \param f_load The packed function of `Load`.
   * \param f_save The packed function of `Save`.
   * \param f_update The packed function of `Update`.
   * \param f_predict The packed function of `Predict`.
   * \param f_as_string The packed function of `AsString`.
   * \return The feature extractor created.
   */
  TVM_DLL static CostModel PyCostModel(PyCostModelNode::FLoad f_load,        //
                                       PyCostModelNode::FSave f_save,        //
                                       PyCostModelNode::FUpdate f_update,    //
                                       PyCostModelNode::FPredict f_predict,  //
                                       PyCostModelNode::FAsString f_as_string);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CostModel, ObjectRef, CostModelNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_COST_MODEL_H_
