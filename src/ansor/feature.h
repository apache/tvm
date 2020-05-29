/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/search_task.h
 * \brief Meta inforamtion for a search task
 */

#ifndef TVM_ANSOR_FEATURE_H_
#define TVM_ANSOR_FEATURE_H_

// #include <tvm/build_module.h>
#include <string>
#include <vector>
#include "compute_dag.h"
#include "measure.h"

namespace tvm {
namespace ansor {

/*! \brief Get PerStmt feature from a tvm IR stmt */
void GetPerStmtFeature(const Stmt& stmt,
                       int cache_line_size,
                       int max_n_bufs,
                       std::vector<float>* ret);

/* \brief Get the name of every element in the feature vector. Use this for debug and inspection */
void GetPerStmtFeatureName(int max_n_bufs, std::vector<std::string> *ret);


/*! \brief Get PerStmt feature from states */
void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const SearchTask& task,
                                  int max_n_bufs,
                                  int skip_first_n_feature_extraction,
                                  std::vector<std::vector<float> >* features);

/*! \brief Get PerStmt feature from states */
void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const std::vector<SearchTask>& tasks,
                                  int max_n_bufs,
                                  int skip_first_n_feature_extraction,
                                  std::vector<std::vector<float> >* features);

/*! \brief Get PerStmt feature from a log file */
void GetPerStmtFeaturesFromFile(const std::string& filename,
                                int n_lines,
                                int max_n_bufs,
                                std::vector<std::vector<float> >* features,
                                std::vector<float>* normalized_throughputs,
                                std::vector<int>* task_ids);

/*! \brief Get PerStmt feature from measure pairs */
void GetPerStmtFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                        const Array<MeasureResult>& results,
                                        int max_n_bufs,
                                        int skip_first_n_feature_extraction,
                                        std::vector<std::vector<float> >* features,
                                        std::vector<float>* normalized_throughputs,
                                        std::vector<int>* task_ids);

}   // namespace ansor
}   // namespace tvm

#endif  // TVM_ANSOR_FEATURE_H_
