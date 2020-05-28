/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/search_task.h
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs
 */

#ifndef TVM_ANSOR_MEASURE_H_
#define TVM_ANSOR_MEASURE_H_

// #include <tvm/build_module.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "search_task.h"
#include "loop_state.h"

namespace tvm {
namespace ansor {

class SearchPolicy;
class MeasureInput; class BuildResult; class MeasureResult;
class Builder; class Runner; class MeasureCallback; class ProgramMeasurer;

extern const char *ErrorNoToStr[];

enum MeasureErrorNO {
  kNoError = 0,              // No error
  kInstantiationError = 1,   // Errors happen when apply transform steps from init state
  kCompileHostError = 2,     // Errors happen when compiling code on host (when build module)
  kCompileDeviceError = 3,   // Errors happen when compiling code on device (when load module)
  kRuntimeDeviceError = 4,   // Errors happen when run program on device
  kWrongAnswerError = 5,     // Answer is wrong when compared to a reference output
  kBuildTimeoutError = 6,    // Timeout during compilation
  kRunTimeoutError = 7,      // Timeout during run
  kUnknonwError = 8,         // Unknown error
};

// Inputs and results of one measurement

/* \brief Store the input of a meansurement */
class MeasureInputNode: public Object {
 public:
  SearchTask task;
  State state;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("state", &state);
  }

  static MeasureInput make(SearchTask task, State state);
  MeasureInput copy() const;  // Do deep copy

  static constexpr const char* _type_key = "ansor.MeasureInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureInputNode, Object);
};
TVM_DEFINE_NODE_REF(MeasureInput, MeasureInputNode);

/* \brief Store the input of a build */
class BuildResultNode: public Object {
 public:
  std::string filename;
  Array<te::Tensor> args;
  int error_no;
  std::string error_msg;
  double time_cost;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("filename", &filename);
    v->Visit("args", &args);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("time_cost", &time_cost);
  }

  static BuildResult make(std::string filename, Array<te::Tensor> args,
                          int error_no, std::string error_msg, double time_cost);

  static constexpr const char* _type_key = "ansor.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, Object);
};
TVM_DEFINE_NODE_REF(BuildResult, BuildResultNode);

/* \brief Store the results of a measurement */
class MeasureResultNode: public Object {
 public:
  Array<PrimExpr> costs;
  int error_no;
  std::string error_msg;
  double all_cost;
  double timestamp;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("costs", &costs);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("all_cost", &all_cost);
    v->Visit("timestamp", &timestamp);
  }

  MeasureResult copy() const;  // Do deep copy

  static MeasureResult make(Array<PrimExpr> costs, int error_no, std::string error_msg,
                            double all_cost, double timestamp);

  static constexpr const char* _type_key = "ansor.MeasureResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureResultNode, Object);
};
TVM_DEFINE_NODE_REF(MeasureResult, MeasureResultNode);


// Measure callback
class MeasureCallbackNode: public Object {
 public:
  virtual void callback(const SearchPolicy& policy,
                        const Array<MeasureInput>& inputs,
                        const Array<MeasureResult>& results) = 0;
  static constexpr const char *_type_key = "ansor.MeasureCallback";
  TVM_DECLARE_BASE_OBJECT_INFO(MeasureCallbackNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(MeasureCallback, MeasureCallbackNode);


// Base class for builder and runner

/* \brief Builder that builds the programs */
class BuilderNode: public Object {
 public:
  int n_parallel;
  int timeout;

  virtual Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) = 0;

  static constexpr const char* _type_key = "ansor.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(Builder, BuilderNode);

/* \brief Runner that runs the built programs and measure the time cost */
class RunnerNode: public Object {
 public:
  int timeout;

  virtual Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                                   const Array<BuildResult>& build_results,
                                   int verbose) = 0;

  static constexpr const char* _type_key = "ansor.Runner";
  TVM_DECLARE_BASE_OBJECT_INFO(RunnerNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(Runner, RunnerNode);


// Implementation of various builders and runners
/* \brief LocalBuilder use local CPU cores to build programs in parallel */
class LocalBuilderNode: public BuilderNode {
 public:
  std::string build_func;

  static Builder make(int timeout, int n_parallel, const std::string& build_func);

  Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) final;

  static constexpr const char* _type_key = "ansor.LocalBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(LocalBuilderNode, BuilderNode);
};

class RPCRunnerNode : public RunnerNode {
 public:
  std::string key;
  std::string host;
  int port;
  int priority;
  int n_parallel;
  int number;
  int repeat;
  int min_repeat_ms;
  double cooldown_interval;

  static Runner make(const std::string& key, const std::string& host, int port,
                     int priority, int timeout, int n_parallel, int number,
                     int repeat, int min_repeat_ms, double cooldown_interval);

  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results,
                           int verbose) final;

  static constexpr const char* _type_key = "ansor.RPCRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(RPCRunnerNode, RunnerNode);
};

/* \brief LocalRunner use local CPU/GPU to runs programs in serial and measure the time cost */
class LocalRunnerNode: public RunnerNode {
 public:
  int number;
  int repeat;
  int min_repeat_ms;
  double cooldown_interval;

  static Runner make(int timeout, int number, int repeat,
                     int min_repeat_ms, double cooldown_interval);

  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results,
                           int verbose) final;

  static constexpr const char* _type_key = "ansor.LocalRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(LocalRunnerNode, RunnerNode);
};


/*!
 * \brief Measurer measures the time costs of tvm programs
 * This class combines Builder and Runner, and provides a simpler API
 */
class ProgramMeasurerNode: public Object {
 public:
  static const int DEFAULT_MAX_CONTINOUS_ERROR = 150;

  int ct;
  int error_ct;   // continuous error counter
  std::unordered_map<std::string, double> best_flops;
  std::unordered_map<std::string, State> best_state;
  std::unordered_map<std::string, int> best_ct;

  Builder builder;
  Runner runner;
  Array<MeasureCallback> callbacks;
  int verbose;
  int max_continous_error;

  static ProgramMeasurer make(Builder builder, Runner runner,
                              Array<MeasureCallback> callbacks,
                              int verbose,
                              int max_continous_error = -1);

  /*! \brief Reset book keeping variables */
  void Reset();

  /*! \biref Do measurement */
  void Measure(const SearchTask& task,
               const SearchPolicy& policy,
               const std::vector<MeasureInput>& inputs,
               std::vector<MeasureResult>* results,
               int batch_size = -1);

  /*! \biref Do measurement silently */
  void SilentMeasure(const SearchTask& task,
                     const std::vector<MeasureInput>& inputs,
                     std::vector<MeasureResult>* results);

  static constexpr const char* _type_key = "ansor.ProgramMeasurer";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProgramMeasurerNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(ProgramMeasurer, ProgramMeasurerNode);


}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_MEASURE_H_
