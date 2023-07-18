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
 * \file graph_executor_debug.cc
 */
#include "./graph_executor_debug.h"

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

#include "../../rpc/rpc_session.h"

namespace tvm {
namespace runtime {
std::string GraphExecutorDebug::RunIndividual(int number, int repeat, int min_repeat_ms,
                                              int limit_zero_time_iterations,
                                              int cooldown_interval_ms, int repeats_to_cooldown) {
  // warmup run
  GraphExecutor::Run();
  std::string tkey = module_->type_key();
  std::vector<std::vector<double>> time_sec_per_op(op_execs_.size());
  if (tkey == "rpc") {
    // RPC modules rely on remote timing which implements the logic from the else branch.
    for (size_t index = 0; index < op_execs_.size(); ++index) {
      time_sec_per_op[index] =
          RunOpRPC(index, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                   cooldown_interval_ms, repeats_to_cooldown);
    }
  } else {
    int op = 0;
    for (size_t index = 0; index < op_execs_.size(); ++index) {
      std::string result_str =
          RunIndividualNode(index, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                            cooldown_interval_ms, repeats_to_cooldown);
      const double* blob_ptr = reinterpret_cast<const double*>(result_str.data());
      for (int i = 0; i < repeat; ++i, ++blob_ptr) {
        time_sec_per_op[index].push_back(*blob_ptr);
      }
      if (op_execs_[index]) {
        LOG(INFO) << "Op #" << op << " " << GetNodeName(index) << ":";
        for (size_t cur_repeat = 0; cur_repeat < time_sec_per_op[index].size(); cur_repeat++) {
          const auto& data = time_sec_per_op[index][cur_repeat];
          LOG(INFO) << "Iteration: " << cur_repeat << ": " << (data * 1e6) << " us/iter";
        }
        ++op;
      }
    }
  }

  std::ostringstream os;
  int64_t size = time_sec_per_op.size();
  os.write(reinterpret_cast<char*>(&size), sizeof(int64_t));
  for (size_t index = 0; index < time_sec_per_op.size(); ++index) {
    for (auto& repeat_data : time_sec_per_op[index]) {
      // To have good behavior when calculating total time, etc.
      double data = std::isnan(repeat_data) ? 0 : repeat_data;
      os.write(reinterpret_cast<char*>(&data), sizeof(double));
    }
  }
  return os.str();
}

std::string GraphExecutorDebug::RunIndividualNode(int node_index, int number, int repeat,
                                                  int min_repeat_ms, int limit_zero_time_iterations,
                                                  int cooldown_interval_ms,
                                                  int repeats_to_cooldown) {
  std::string tkey = module_->type_key();

  if (tkey == "rpc") {
    LOG(FATAL) << "RPC measurements should not use RunIndividualNode!";
  }

  if (!op_execs_[node_index]) {
    // don't return anything...
    std::ostringstream os;
    double zero = 0;
    for (int i = 0; i < repeat; ++i) {
      os.write(reinterpret_cast<char*>(&zero), sizeof(double));
    }
    return os.str();
  }

  // assume host runs things which is first device
  Device& d = devices_[0];
  PackedFunc time_evaluator = profiling::WrapTimeEvaluator(
      TypedPackedFunc<void()>([this, node_index]() { this->RunOpHost(node_index); }), d, number,
      repeat, min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown);
  return time_evaluator();
}

std::vector<double> GraphExecutorDebug::RunOpRPC(int index, int number, int repeat,
                                                 int min_repeat_ms, int limit_zero_time_iterations,
                                                 int cooldown_interval_ms,
                                                 int repeats_to_cooldown) {
  std::vector<double> results(repeat, 0);
  // Right now we expect either "tvm_op" for nodes which run PackedFunc or "null" for nodes
  // which represent inputs/parameters to the graph. Other types may be supported in the
  // future, but consideration would be needed as to how to do that over RPC before we support
  // it here.
  if (nodes_[index].op_type != "tvm_op") {
    CHECK_EQ(nodes_[index].op_type, "null")
        << "Don't know how to run op type " << nodes_[index].op_type
        << " remotely over RPC right now";

    // NOTE: GraphExecutorDebug expects graph nodes to have an "op" attribute of "tvm_op" or
    // "null" and "null" is a placeholder node for a parameter or input.
    return results;
  }

  const Device& dev = data_entry_[entry_id(index, 0)]->device;
  TVMOpParam param = nodes_[index].param;
  std::string name = param.func_name;
  uint32_t num_inputs = param.num_inputs;
  uint32_t num_outputs = param.num_outputs;

  PackedFunc time_eval =
      runtime::Registry::Get("runtime.RPCTimeEvaluator")
          ->
          operator()(module_, name, static_cast<int>(dev.device_type), dev.device_id, number,
                     repeat, min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
                     repeats_to_cooldown, /*cache_flush_bytes=*/0, "");

  int num_flat_args = num_inputs + num_outputs;
  auto values = std::make_unique<TVMValue[]>(num_flat_args);
  auto type_codes = std::make_unique<int[]>(num_flat_args);
  TVMArgsSetter setter(values.get(), type_codes.get());
  int offs = 0;
  const auto& inode = nodes_[index];
  for (const auto& e : inode.inputs) {
    uint32_t eid = this->entry_id(e);
    DLTensor* arg = const_cast<DLTensor*>(data_entry_[eid].operator->());
    setter(offs, arg);
    offs++;
  }
  for (uint32_t i = 0; i < num_outputs; ++i) {
    uint32_t eid = this->entry_id(index, i);
    DLTensor* arg = const_cast<DLTensor*>(data_entry_[eid].operator->());
    setter(offs, arg);
    offs++;
  }
  TVMRetValue rv;
  time_eval.CallPacked(TVMArgs(values.get(), type_codes.get(), num_flat_args), &rv);
  std::string results_str = rv.operator std::string();
  const double* blob_ptr = reinterpret_cast<const double*>(results_str.data());
  for (int i = 0; i < repeat; ++i, ++blob_ptr) {
    results[i] = *blob_ptr;
  }

  std::ostringstream os;
  for (auto& repeat_data : results) {
    os << std::to_string(repeat_data) << ", ";
  }
  LOG(INFO) << "Got op timing: " << os.str();
  return results;
}

Timer GraphExecutorDebug::RunOpHost(int index) {
  const Device& dev = data_entry_[entry_id(index, 0)]->device;
  Timer t = Timer::Start(dev);
  op_execs_[index]();
  t->Stop();
  return t;
}

/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphExecutorDebug::GetFunction(const String& name,
                                           const ObjectPtr<Object>& sptr_to_self) {
  // return member functions during query.
  if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
      } else {
        this->DebugGetNodeOutput(args[0], args[1]);
      }
    });
  } else if (name == "execute_node") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->ExecuteNode(args[0]); });
  } else if (name == "get_node_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetNodeOutput(args[0], args[1]);
    });
  } else if (name == "run_individual") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int number = args[0];
      int repeat = args[1];
      int min_repeat_ms = args[2];
      int limit_zero_time_iterations = args[3];
      int cooldown_interval_ms = args[4];
      int repeats_to_cooldown = args[5];
      ICHECK_GT(number, 0);
      ICHECK_GT(repeat, 0);
      ICHECK_GE(min_repeat_ms, 0);
      ICHECK_GE(limit_zero_time_iterations, 0);
      ICHECK_GE(cooldown_interval_ms, 0);
      ICHECK_GT(repeats_to_cooldown, 0);
      std::string blob =
          this->RunIndividual(number, repeat, min_repeat_ms, limit_zero_time_iterations,
                              cooldown_interval_ms, repeats_to_cooldown);
      TVMByteArray arr;
      arr.size = blob.length();
      arr.data = blob.data();
      *rv = arr;
    });
  } else if (name == "run_individual_node") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int node_index = args[0];
      int number = args[1];
      int repeat = args[2];
      int min_repeat_ms = args[3];
      int limit_zero_time_iterations = args[4];
      int cooldown_interval_ms = args[5];
      int repeats_to_cooldown = args[6];
      ICHECK_GE(node_index, 0);
      ICHECK_LT(node_index, nodes_.size());
      ICHECK_GT(number, 0);
      ICHECK_GT(repeat, 0);
      ICHECK_GE(min_repeat_ms, 0);
      ICHECK_GE(limit_zero_time_iterations, 0);
      ICHECK_GE(cooldown_interval_ms, 0);
      ICHECK_GT(repeats_to_cooldown, 0);
      std::string blob = this->RunIndividualNode(node_index, number, repeat, min_repeat_ms,
                                                 limit_zero_time_iterations, cooldown_interval_ms,
                                                 repeats_to_cooldown);
      TVMByteArray arr;
      arr.size = blob.length();
      arr.data = blob.data();
      *rv = arr;
    });
  } else if (name == "profile") {
    return TypedPackedFunc<profiling::Report(Array<profiling::MetricCollector>)>(
        [sptr_to_self, this](Array<profiling::MetricCollector> collectors) {
          // We cannot send Arrays over rpc, so in order to support profiling
          // on remotes, we accept a nullptr for collectors.
          if (collectors.defined()) {
            return this->Profile(collectors);
          } else {
            return this->Profile({});
          }
        });
  } else if (name == "profile_rpc") {
    // We cannot return a Report over RPC because TMV RPC mechanism only
    // supports a subset of Object classes. Instead we serialize it on the
    // remote (here) and deserialize it on the other end.
    return TypedPackedFunc<std::string()>([sptr_to_self, this]() {
      PackedFunc profile = GetFunction("profile", sptr_to_self);
      profiling::Report report = profile(Array<profiling::MetricCollector>());
      return report->AsJSON();
    });
  } else {
    return GraphExecutor::GetFunction(name, sptr_to_self);
  }
}

int GraphExecutorDebug::GetNodeIndex(const std::string& name) const {
  for (size_t nid = 0; nid < GetNumOfNodes(); ++nid) {
    if (GetNodeName(nid) == name) {
      return static_cast<int>(nid);
    }
  }
  LOG(FATAL) << "cannot find " << name << " among nodex";
  return -1;
}

void GraphExecutorDebug::ExecuteNode(int node) {
  ICHECK_LT(static_cast<size_t>(node), op_execs_.size());

  int start_ind;
  int end_ind;
  if (node < last_executed_node_) {
    start_ind = 0;
    end_ind = node;
  } else if (node > last_executed_node_) {
    start_ind = last_executed_node_ + 1;
    end_ind = node;
  } else {
    return;
  }

  for (int i = start_ind; i <= end_ind; i++) {
    if (op_execs_[i]) op_execs_[i]();
  }
  last_executed_node_ = end_ind;
}

void GraphExecutorDebug::DebugGetNodeOutput(int index, DLTensor* data_out) {
  ICHECK_LT(static_cast<size_t>(index), op_execs_.size());
  uint32_t eid = index;

  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
    if (static_cast<int>(i) == index) break;
  }

  data_entry_[eid].CopyTo(data_out);
}

NDArray GraphExecutorDebug::GetNodeOutput(int node, int out_ind) {
  ICHECK_EQ(node, last_executed_node_);
  ICHECK_LT(entry_id(node, out_ind), data_entry_.size());
  return data_entry_[entry_id(node, out_ind)].CopyTo({kDLCPU, 0});
}

profiling::Report GraphExecutorDebug::Profile(Array<profiling::MetricCollector> collectors) {
  std::vector<profiling::MetricCollector> cs(collectors.begin(), collectors.end());
  profiling::Profiler prof(devices_, cs, {{String("Executor"), String("Graph")}});

  // warm up. 1 iteration does not seem enough.
  for (int i = 0; i < 3; i++) {
    GraphExecutor::Run();
  }

  prof.Start();
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) {
      // get argument shapes
      std::vector<NDArray> shapes;
      for (const auto& e : nodes_[i].inputs) {
        uint32_t eid = entry_id(e);
        shapes.push_back(data_entry_[eid]);
      }
      for (uint32_t j = 0; j < nodes_[i].param.num_outputs; ++j) {
        uint32_t eid = entry_id(i, j);
        shapes.push_back(data_entry_[eid]);
      }

      uint32_t eid = entry_id(i, 0);
      const Device& dev = data_entry_[eid]->device;

      std::unordered_map<std::string, ObjectRef> metrics;
      for (auto p : nodes_[i].param.attrs) {
        if (std::string(p.first).find("layout") != std::string::npos) {
          metrics[p.first] = p.second;
        }
      }
      if (nodes_[i].param.attrs.find("hash") != nodes_[i].param.attrs.end()) {
        metrics["Hash"] = Downcast<String>(nodes_[i].param.attrs.at("hash"));
      }
      metrics["Argument Shapes"] = profiling::ShapeString(shapes);
      prof.StartCall(nodes_[i].param.func_name, dev, metrics);
      op_execs_[i]();
      prof.StopCall();
    }
  }
  prof.Stop();
  return prof.Report();
}

/*!
 * \brief GraphExecutorDebugCreate Get the function based on input.
 * \param sym_json The graph symbol in json format.
 * \param m Compiled module which will be loaded.
 * \param devs All devices.
 */
Module GraphExecutorDebugCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                                const std::vector<Device>& devs,
                                PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphExecutorDebug>();
  exec->Init(sym_json, m, devs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_executor_debug.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_executor.create is "
                                 "at least 4, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int dev_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) {
    lookup_linked_param_func = args[2];
    dev_start_arg++;
  }

  *rv = GraphExecutorDebugCreate(args[0], args[1], GetAllDevice(args, dev_start_arg),
                                 lookup_linked_param_func);
});
}  // namespace runtime
}  // namespace tvm
