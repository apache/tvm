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
 * \file ansor/measure.cc
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs
 */

#include "measure.h"
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <algorithm>
#include <iomanip>
#include <utility>
#include <vector>

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_OBJECT_TYPE(RunnerNode);
TVM_REGISTER_OBJECT_TYPE(BuilderNode);
TVM_REGISTER_OBJECT_TYPE(LocalBuilderNode);
TVM_REGISTER_OBJECT_TYPE(RPCRunnerNode);
TVM_REGISTER_OBJECT_TYPE(LocalRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramMeasurerNode);

const char* ErrorNoToStr[] = {
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

// Measure input and result
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

BuildResult::BuildResult(std::string filename, Array<te::Tensor> args,
                         int error_no, std::string error_msg,
                         double time_cost) {
  auto node = make_object<BuildResultNode>();
  node->filename = std::move(filename);
  node->args = std::move(args);
  node->error_no = error_no;
  node->error_msg = std::move(error_msg);
  node->time_cost = time_cost;
  data_ = std::move(node);
}

MeasureResult::MeasureResult(Array<PrimExpr> costs, int error_no,
                             std::string error_msg, double all_cost,
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

// LocalBuilder
LocalBuilder::LocalBuilder(int timeout, int n_parallel,
                           const std::string& build_func) {
  auto node = make_object<LocalBuilderNode>();
  node->timeout = timeout;
  node->n_parallel = n_parallel;
  node->build_func = build_func;
  data_ = std::move(node);
}

Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs,
                                           int verbose) {
  if (const auto* f = runtime::Registry::Get("ansor.local_builder.build")) {
    Array<BuildResult> results =
        (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  } else {
    LOG(FATAL) << "ansor.local_builder.build is not registered";
  }
  return Array<BuildResult>();
}

// RPC Runner
RPCRunner::RPCRunner(const std::string& key, const std::string& host, int port,
                     int priority, int timeout, int n_parallel, int number,
                     int repeat, int min_repeat_ms, double cooldown_interval) {
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
  data_ = std::move(node);
}

Array<MeasureResult> RPCRunnerNode::Run(const Array<MeasureInput>& inputs,
                                        const Array<BuildResult>& build_results,
                                        int verbose) {
  if (const auto* f = runtime::Registry::Get("ansor.rpc_runner.run")) {
    Array<MeasureResult> results = (*f)(
        inputs, build_results, key, host, port, priority, timeout, n_parallel,
        number, repeat, min_repeat_ms, cooldown_interval, verbose);
    return results;
  } else {
    LOG(FATAL) << "ansor.rpc_runner.run is not registered";
  }
  return Array<MeasureResult>();
}

// Local Runner
LocalRunner::LocalRunner(int timeout, int number, int repeat,
                         int min_repeat_ms, double cooldown_interval) {
  ObjectPtr<LocalRunnerNode> node = make_object<LocalRunnerNode>();
  node->timeout = timeout;
  node->number = number;
  node->repeat = repeat;
  node->min_repeat_ms = min_repeat_ms;
  node->cooldown_interval = cooldown_interval;
  data_ = std::move(node);
}

Array<MeasureResult> LocalRunnerNode::Run(
    const Array<MeasureInput>& inputs, const Array<BuildResult>& build_results,
    int verbose) {
  if (const auto* f = runtime::Registry::Get("ansor.local_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, timeout, number, repeat, min_repeat_ms,
             cooldown_interval, verbose);
    return results;
  } else {
    LOG(FATAL) << "ansor.local_runner.run is not registered";
  }
  return Array<MeasureResult>();
}

// Program Measurer
ProgramMeasurer::ProgramMeasurer(Builder builder, Runner runner,
                                 Array<MeasureCallback> callbacks, int verbose,
                                 int max_continous_error) {
  auto node = make_object<ProgramMeasurerNode>();
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->callbacks = std::move(callbacks);
  node->verbose = verbose;
  node->max_continous_error = max_continous_error < 0 ?
      ProgramMeasurerNode::DEFAULT_MAX_CONTINOUS_ERROR : max_continous_error;
  data_ = std::move(node);
}

void ProgramMeasurerNode::Reset() {
  ct = error_ct = 0;
  best_flops.clear();
  best_ct.clear();
  best_state.clear();
}

void ProgramMeasurerNode::Measure(const SearchTask& task,
                                  const SearchPolicy& policy,
                                  const std::vector<MeasureInput>& inputs,
                                  std::vector<MeasureResult>* results,
                                  int batch_size) {
  results->clear();
  results->reserve(inputs.size());

  if (batch_size == -1) {
    // set default batch size
    batch_size = builder->n_parallel * 2;
  }

  StdCout(verbose) << "Get " << inputs.size()
                   << " programs for measure. (This may take a while)"
                   << std::endl;

  for (size_t i = 0; i < inputs.size(); i += batch_size) {
    std::vector<MeasureInput> input_batch(
        inputs.begin() + i,
        inputs.begin() + std::min(i + batch_size, inputs.size()));
    std::vector<MeasureResult> result_batch;

    // build and run
    SilentMeasure(task, input_batch, &result_batch);

    // update current best state according to the new measure result
    for (size_t j = 0; j < input_batch.size(); ++j) {
      double flops;
      if (result_batch[j]->error_no == 0) {
        flops =
            task->compute_dag->flop_ct / FloatArrayMean(result_batch[j]->costs);
        error_ct = 0;
      } else {
        flops = 0.0;
        error_ct++;
      }

      const std::string& workload_key = input_batch[j]->task->workload_key;
      if (flops > best_flops[workload_key]) {
        best_flops[workload_key] = flops;
        best_state[workload_key] = input_batch[j]->state;
        best_ct[workload_key] = ct;
      }

      ct++;
      if (verbose >= 1) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "===============================================\n";
        std::cout << "No: " << ct << "\tGFLOPS: " << flops / 1e9 << " / "
                  << best_flops[workload_key] / 1e9
                  << "\tresults: " << result_batch[j] << "\n";
        std::cout << "===============================================\n";
        std::cout << input_batch[j]->state << "\n";
      }
    }

    // Call callback functions
    for (const auto& callback : callbacks) {
      callback->callback(policy, input_batch, result_batch);
    }

    // Store result batch
    for (auto& res : result_batch) {
      results->push_back(res);
    }

    if (error_ct > max_continous_error) {
      LOG(FATAL) << "Too many errors happened during tuning";
    }
  }
}

void ProgramMeasurerNode::SilentMeasure(const SearchTask& task,
                                        const std::vector<MeasureInput>& inputs,
                                        std::vector<MeasureResult>* results) {
  // Close the thread pool to avoid the conflits with python environment
  ThreadPool::Global().Abort();

  results->clear();
  results->reserve(inputs.size());
  Array<MeasureInput> input_batch(inputs.begin(), inputs.end());

  // Call builder and runner
  Array<BuildResult> build_res_batch = builder->Build(input_batch, verbose);
  Array<MeasureResult> result_batch =
      runner->Run(input_batch, build_res_batch, verbose);

  // Store result batch
  for (auto& res : result_batch) {
    results->push_back(res);
  }
}

// Printing functions
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<MeasureInputNode>([](const ObjectRef& ref, ReprPrinter* p) {
  p->stream << "MeasureInput()";
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<MeasureResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const MeasureResultNode*>(ref.get());
  if (node->error_no == kNoError) {
    p->stream << "MeasureResult(cost:[";
    auto old_config = p->stream.precision(4);
    for (size_t i = 0; i < node->costs.size(); ++i) {
      auto pf = node->costs[i].as<FloatImmNode>();
      CHECK(pf != nullptr);
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
  p->stream << "BuildResult(" << node->filename << ", " << node->error_no
            << ", " << node->time_cost << ")";
});

TVM_REGISTER_GLOBAL("ansor.MeasureInput").set_body_typed([](SearchTask task, State state) {
  return MeasureInput(task, state);
});

TVM_REGISTER_GLOBAL("ansor.BuildResult")
.set_body_typed([](std::string filename, Array<te::Tensor> args,
                   int error_no, std::string error_msg, double time_cost) {
  return BuildResult(filename, args, error_no, error_msg, time_cost);
});

TVM_REGISTER_GLOBAL("ansor.MeasureResult")
.set_body_typed([](Array<PrimExpr> costs, int error_no, std::string error_msg,
                   double all_cost, double timestamp) {
  return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
});

TVM_REGISTER_GLOBAL("ansor.BuilderBuild")
.set_body_typed([](const Builder& builder, const Array<MeasureInput>& inputs, int verbose) {
  return builder->Build(inputs, verbose);
});

TVM_REGISTER_GLOBAL("ansor.RunnerRun")
.set_body_typed([](const Runner& runner, const Array<MeasureInput>& inputs,
                   const Array<BuildResult>& build_results, int verbose) {
  return runner->Run(inputs, build_results, verbose);
});

TVM_REGISTER_GLOBAL("ansor.LocalBuilder")
.set_body_typed([](int timeout, int n_parallel, const std::string& build_func) {
  return LocalBuilder(timeout, n_parallel, build_func);
});

TVM_REGISTER_GLOBAL("ansor.LocalRunner")
.set_body_typed([](int timeout, int number, int repeat,
                   int min_repeat_ms, double cooldown_interval) {
  return LocalRunner(timeout, number, repeat, min_repeat_ms, cooldown_interval);
});

TVM_REGISTER_GLOBAL("ansor.RPCRunner")
.set_body_typed([](const std::string& key, const std::string& host, int port,
                   int priority, int timeout, int n_parallel, int number,
                   int repeat, int min_repeat_ms, double cooldown_interval){
  return RPCRunner(key, host, port, priority, timeout, n_parallel, number,
                   repeat, min_repeat_ms, cooldown_interval);
});

TVM_REGISTER_GLOBAL("ansor.ProgramMeasurer")
.set_body_typed([](Builder builder, Runner runner,
                   Array<MeasureCallback> callbacks, int verbose,
                   int max_continous_error = -1) {
  return ProgramMeasurer(builder, runner, callbacks, verbose,
                         max_continous_error);
});

}  // namespace ansor
}  // namespace tvm
