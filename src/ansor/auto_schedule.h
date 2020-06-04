/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/search_task.h
 * \brief Meta information for a search task
 */

#ifndef TVM_ANSOR_AUTO_SCHEDULE_H_
#define TVM_ANSOR_AUTO_SCHEDULE_H_

#include "measure.h"

namespace tvm {
namespace ansor {

/*! \brief Tuning and measurement options */
class TuneOption;
class TuneOptionNode : public Object {
 public:
  int n_trials;              // Number of total measurement trials
  int early_stopping;        // Stops early the tuning if no improvement after n
                             // measurements
  int num_measure_per_iter;  // The number of programs to be measured at each
                             // iteration
  int verbose;               // Verbosity level. 0 means silent.
  Builder builder;           // Builder which builds the program
  Runner runner;             // Runner which runs the program and measure time
                             // costs
  Array<MeasureCallback> callbacks;  // Callback functions

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("n_trials", &n_trials);
    v->Visit("early_stopping", &early_stopping);
    v->Visit("num_measure_per_iter", &num_measure_per_iter);
    v->Visit("verbose", &verbose);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("callbacks", &callbacks);
  }

  static TuneOption make(int n_trials, int early_stopping,
                         int num_measure_per_iter, int verbose, Builder builder,
                         Runner runner, Array<MeasureCallback> callbacks);

  static constexpr const char* _type_key = "ansor.TuneOption";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuneOptionNode, Object);
};
TVM_DEFINE_COW_NODE_REF(TuneOption, ObjectRef, TuneOptionNode);

/*! \brief Auto schedule for a compute declaration */
State AutoSchedule(SearchTask task, SearchPolicy search_policy,
                   TuneOption tune_option);

std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    std::string workload_key, Target target, Target target_host,
    SearchPolicy search_policy, HardwareParams hardware_params,
    TuneOption tune_option);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_AUTO_SCHEDULE_H_