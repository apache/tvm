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
 * \file auto_scheduler/measure.cc
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs.
 */

#include <tvm/auto_scheduler/measure.h>
#include <tvm/runtime/registry.h>

#include <algorithm>

#include "search_policy/empty_policy.h"
#include "search_policy/sketch_policy.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedMeasureCallbackNode);
TVM_REGISTER_OBJECT_TYPE(ProgramRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramBuilderNode);
TVM_REGISTER_OBJECT_TYPE(ProgramMeasurerNode);
TVM_REGISTER_OBJECT_TYPE(LocalBuilderNode);
TVM_REGISTER_OBJECT_TYPE(LocalRunnerNode);
TVM_REGISTER_OBJECT_TYPE(RPCRunnerNode);

static const char* ErrorNoToStr[] = {
    "NoError",
    "InstantiationError",
    "CompileHostError",
    "CompileDeviceError",
    "RuntimeDeviceError",
    "WrongAnswerError",
    "BuildTimeoutError",
    "RunTimeoutError",
    "UnknownError",
};

/********** Measure input and result **********/
MeasureInput::MeasureInput(SearchTask task, State state) {
  auto node = make_object<MeasureInputNode>();
  node->task = std::move(task);
  node->state = std::move(state);
  data_ = std::move(node);
}

MeasureInput MeasureInputNode::copy() const {
  auto node = make_object<MeasureInputNode>();
  node->task = task;
  node->state = state;
  return MeasureInput(node);
}

BuildResult::BuildResult(String filename, Array<te::Tensor> args, int error_no, String error_msg,
                         double time_cost) {
  auto node = make_object<BuildResultNode>();
  node->filename = std::move(filename);
  node->args = std::move(args);
  node->error_no = error_no;
  node->error_msg = std::move(error_msg);
  node->time_cost = time_cost;
  data_ = std::move(node);
}

MeasureResult::MeasureResult(Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                             double timestamp) {
  auto node = make_object<MeasureResultNode>();
  node->costs = std::move(costs);
  node->error_no = error_no;
  node->error_msg = std::move(error_msg);
  node->all_cost = all_cost;
  node->timestamp = timestamp;
  data_ = std::move(node);
}

MeasureResult MeasureResultNode::copy() const {
  auto node = make_object<MeasureResultNode>();
  node->costs = costs;
  node->error_no = error_no;
  node->error_msg = error_msg;
  node->all_cost = all_cost;
  node->timestamp = timestamp;
  return MeasureResult(node);
}

/********** LocalBuilder **********/
LocalBuilder::LocalBuilder(int timeout, int n_parallel, const String& build_func) {
  auto node = make_object<LocalBuilderNode>();
  node->timeout = timeout;
  node->n_parallel = n_parallel;
  node->build_func = build_func;
  data_ = std::move(node);
}

Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.local_builder.build")) {
    Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  }
  LOG(FATAL) << "auto_scheduler.local_builder.build is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** LocalRunner **********/
LocalRunner::LocalRunner(int timeout, int number, int repeat, int min_repeat_ms,
                         double cooldown_interval, bool enable_cpu_cache_flush, int device) {
  ObjectPtr<LocalRunnerNode> node = make_object<LocalRunnerNode>();
  node->timeout = timeout;
  node->number = number;
  node->repeat = repeat;
  node->min_repeat_ms = min_repeat_ms;
  node->cooldown_interval = cooldown_interval;
  node->enable_cpu_cache_flush = enable_cpu_cache_flush;
  node->device = device;
  data_ = std::move(node);
}

Array<MeasureResult> LocalRunnerNode::Run(const Array<MeasureInput>& inputs,
                                          const Array<BuildResult>& build_results, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.local_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, timeout, number, repeat, min_repeat_ms, cooldown_interval,
             enable_cpu_cache_flush, verbose, device);
    return results;
  }
  LOG(FATAL) << "auto_scheduler.local_runner.run is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** RPCRunner **********/
RPCRunner::RPCRunner(const String& key, const String& host, int port, int priority, int n_parallel,
                     int timeout, int number, int repeat, int min_repeat_ms,
                     double cooldown_interval, bool enable_cpu_cache_flush, int device) {
  auto node = make_object<RPCRunnerNode>();
  node->key = key;
  node->host = host;
  node->port = port;
  node->priority = priority;
  node->timeout = timeout;
  node->n_parallel = n_parallel;
  node->number = number;
  node->repeat = repeat;
  node->min_repeat_ms = min_repeat_ms;
  node->cooldown_interval = cooldown_interval;
  node->enable_cpu_cache_flush = enable_cpu_cache_flush;
  node->device = device;
  data_ = std::move(node);
}

Array<MeasureResult> RPCRunnerNode::Run(const Array<MeasureInput>& inputs,
                                        const Array<BuildResult>& build_results, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.rpc_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, key, host, port, priority, n_parallel, timeout, number, repeat,
             min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, verbose, device);
    return results;
  } else {
    LOG(FATAL) << "auto_scheduler.rpc_runner.run is not registered. "
               << "This is a function registered in Python, "
               << "make sure the TVM Python runtime has been loaded successfully.";
  }
  return Array<MeasureResult>();
}

/********** MeasureCallback **********/
PythonBasedMeasureCallback::PythonBasedMeasureCallback(PackedFunc callback_func) {
  auto node = make_object<PythonBasedMeasureCallbackNode>();
  node->callback_func = std::move(callback_func);
  data_ = std::move(node);
}

void PythonBasedMeasureCallbackNode::Callback(const SearchPolicy& policy,
                                              const Array<MeasureInput>& inputs,
                                              const Array<MeasureResult>& results) {
  if (auto* sketch_policy = static_cast<SketchPolicyNode*>(policy.operator->())) {
    callback_func(GetRef<SketchPolicy>(sketch_policy), inputs, results);
  } else if (auto* empty_policy = static_cast<EmptyPolicyNode*>(policy.operator->())) {
    callback_func(GetRef<EmptyPolicy>(empty_policy), inputs, results);
  } else {
    LOG(FATAL) << "Unrecognized search policy type. Expect SketchPolicy or EmptyPolicy";
  }
}

/********** ProgramMeasurer **********/
ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Optional<Array<MeasureCallback>> callbacks, int verbose,
                                 int max_continuous_error) {
  auto node = make_object<ProgramMeasurerNode>();
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->callbacks = std::move(callbacks);
  node->verbose = verbose;
  node->max_continuous_error = max_continuous_error < 0
                                   ? ProgramMeasurerNode::DEFAULT_MAX_CONTINUOUS_ERROR
                                   : max_continuous_error;
  data_ = std::move(node);
}

void ProgramMeasurerNode::Reset() {
  ct = error_ct = 0;
  best_flops.clear();
  best_ct.clear();
  best_state.clear();
  has_valid.clear();
}

Array<MeasureResult> ProgramMeasurerNode::Measure(const SearchTask& task,
                                                  const SearchPolicy& policy,
                                                  const Array<MeasureInput>& inputs,
                                                  int batch_size) {
  auto t_begin = std::chrono::high_resolution_clock::now();

  Array<MeasureResult> results;
  results.reserve(inputs.size());

  if (batch_size == -1) {
    // set default batch size
    batch_size = builder->n_parallel * 2;
  }

  int old_verbosity = verbose;

  StdCout(verbose) << "Get " << inputs.size() << " programs to measure:" << std::endl;

  for (size_t i = 0; i < inputs.size(); i += batch_size) {
    Array<MeasureInput> input_batch(inputs.begin() + i,
                                    inputs.begin() + std::min(i + batch_size, inputs.size()));
    Array<MeasureResult> result_batch;

    // build and run
    SilentMeasure(task, input_batch, &result_batch);

    // update current best state according to the new measure result
    for (size_t j = 0; j < input_batch.size(); ++j) {
      const String& workload_key = input_batch[j]->task->workload_key;
      double flops;

      if (result_batch[j]->error_no == 0) {
        flops = task->compute_dag->flop_ct / FloatArrayMean(result_batch[j]->costs);
        error_ct = 0;
        has_valid.insert(workload_key);
      } else {
        flops = 0.0;
        error_ct++;
      }

      if (flops > best_flops[workload_key]) {
        best_flops[workload_key] = flops;
        best_state[workload_key] = input_batch[j]->state;
        best_ct[workload_key] = ct;
      }

      ct++;
      StdCout(verbose, 2) << std::fixed << std::setprecision(2) << Chars('=', 50) << "\n"
                          << "No: " << ct << "\tGFLOPS: " << flops / 1e9 << " / "
                          << best_flops[workload_key] / 1e9 << "\tresults: " << result_batch[j]
                          << "\n"
                          << Chars('=', 50) << "\n"
                          << input_batch[j]->state << "\n";
    }

    // Call callback functions
    if (callbacks) {
      for (const auto& callback : callbacks.value()) {
        callback->Callback(policy, input_batch, result_batch);
      }
    }

    // Store result batch
    for (auto& res : result_batch) {
      results.push_back(res);
    }

    if (error_ct > max_continuous_error) {
      LOG(WARNING) << "Too many errors happened during tuning. Switching to debug mode."
                   << std::endl;
      verbose = 2;
    } else {
      verbose = old_verbosity;
    }
  }

  PrintTimeElapsed(t_begin, "measurement", verbose);

  return results;
}

void ProgramMeasurerNode::SilentMeasure(const SearchTask& task, const Array<MeasureInput>& inputs,
                                        Array<MeasureResult>* results) {
  results->clear();
  results->reserve(inputs.size());

  // Call builder and runner
  Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
  Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);

  // Store result batch
  for (auto& res : result_batch) {
    results->push_back(res);
  }
}

/********** Printing functions **********/
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureInputNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "MeasureInput()";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MeasureResultNode*>(ref.get());
      if (node->error_no == static_cast<int>(MeasureErrorNO::kNoError)) {
        p->stream << "MeasureResult(cost:[";
        auto old_config = p->stream.precision(4);
        for (size_t i = 0; i < node->costs.size(); ++i) {
          auto pf = node->costs[i].as<FloatImmNode>();
          ICHECK(pf != nullptr);
          p->stream << pf->value;
          if (i != node->costs.size() - 1) {
            p->stream << ",";
          }
        }
        p->stream.precision(old_config);
        p->stream << "], ";
        p->stream << "error_no:" << 0 << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      } else {
        p->stream << "MeasureResult("
                  << "error_type:" << ErrorNoToStr[node->error_no] << ", "
                  << "error_msg:" << node->error_msg << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BuildResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BuildResultNode*>(ref.get());
      p->stream << "BuildResult(" << node->filename << ", " << node->error_no << ", "
                << node->time_cost << ")";
    });

/********** Measure interface API for ffi **********/
TVM_REGISTER_GLOBAL("auto_scheduler.MeasureInput").set_body_typed([](SearchTask task, State state) {
  return MeasureInput(task, state);
});

TVM_REGISTER_GLOBAL("auto_scheduler.BuildResult")
    .set_body_typed([](String filename, Array<te::Tensor> args, int error_no, String error_msg,
                       double time_cost) {
      return BuildResult(filename, args, error_no, error_msg, time_cost);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.MeasureResult")
    .set_body_typed([](Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                       double timestamp) {
      return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.PythonBasedMeasureCallback")
    .set_body_typed([](PackedFunc callback_func) {
      return PythonBasedMeasureCallback(callback_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramMeasurer")
    .set_body_typed([](ProgramBuilder builder, ProgramRunner runner,
                       Array<MeasureCallback> callbacks, int verbose, int max_continuous_error) {
      return ProgramMeasurer(builder, runner, callbacks, verbose, max_continuous_error);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramBuilderBuild")
    .set_body_typed([](const ProgramBuilder& builder, const Array<MeasureInput>& inputs,
                       int verbose) { return builder->Build(inputs, verbose); });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramRunnerRun")
    .set_body_typed([](const ProgramRunner& runner, const Array<MeasureInput>& inputs,
                       const Array<BuildResult>& build_results,
                       int verbose) { return runner->Run(inputs, build_results, verbose); });

TVM_REGISTER_GLOBAL("auto_scheduler.LocalBuilder")
    .set_body_typed([](int timeout, int n_parallel, const String& build_func) {
      return LocalBuilder(timeout, n_parallel, build_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.LocalRunner")
    .set_body_typed([](int timeout, int number, int repeat, int min_repeat_ms,
                       double cooldown_interval, bool enable_cpu_cache_flush, int device) {
      return LocalRunner(timeout, number, repeat, min_repeat_ms, cooldown_interval,
                         enable_cpu_cache_flush, device);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.RPCRunner")
    .set_body_typed([](const String& key, const String& host, int port, int priority,
                       int n_parallel, int timeout, int number, int repeat, int min_repeat_ms,
                       double cooldown_interval, bool enable_cpu_cache_flush, int device) {
      return RPCRunner(key, host, port, priority, n_parallel, timeout, number, repeat,
                       min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, device);
    });

}  // namespace auto_scheduler
}  // namespace tvm
