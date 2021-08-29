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
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <sstream>

#include "../graph_executor.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph executor with debug .
 *
 *  This is the extension of GraphExecutor class used for debugging
 *  TVM runtime PackedFunc API.
 */
class GraphExecutorDebug : public GraphExecutor {
 public:
  /*!
   * \brief Run each operation in the graph and get the time per op for all ops.
   * \param number The number of times to run this function for taking average.
   * \param repeat The number of times to repeat the measurement.
   *        In total, the function will be invoked (1 + number x repeat) times,
   *        where the first one is warmed up and will be discarded in case
   *        there is lazy initialization.
   * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
   *        By default, one `repeat` contains `number` runs. If this parameter is set,
   *        the parameters `number` will be dynamically adjusted to meet the
   *        minimum duration requirement of one `repeat`.
   * \return Comma seperated string containing the elapsed time per op for the last
   *         iteration only, because returning a long string over rpc can be expensive.
   */
  std::string RunIndividual(int number, int repeat, int min_repeat_ms) {
    // warmup run
    GraphExecutor::Run();
    std::string tkey = module_->type_key();
    std::vector<double> time_sec_per_op(op_execs_.size(), 0);
    if (tkey == "rpc") {
      // RPC modules rely on remote timing which implements the logic from the else branch.
      for (size_t index = 0; index < op_execs_.size(); ++index) {
        time_sec_per_op[index] += RunOpRPC(index, number, repeat, min_repeat_ms);
      }
    } else {
      for (int i = 0; i < repeat; ++i) {
        std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds>
            tbegin, tend;
        double duration_ms = 0.0;
        do {
          std::fill(time_sec_per_op.begin(), time_sec_per_op.end(), 0);
          if (duration_ms > 0.0) {
            number = static_cast<int>(std::max((min_repeat_ms / (duration_ms / number) + 1),
                                               number * 1.618));  // 1.618 is chosen by random
          }
          tbegin = std::chrono::high_resolution_clock::now();
          std::vector<std::vector<Timer>> op_timers;
          for (size_t index = 0; index < op_execs_.size(); index++) {
            op_timers.push_back({});
          }
          for (int k = 0; k < number; k++) {
            for (size_t index = 0; index < op_execs_.size(); ++index) {
              if (op_execs_[index]) {
                op_timers[index].push_back(RunOpHost(index));
              }
            }
          }
          for (size_t index = 0; index < op_execs_.size(); ++index) {
            for (auto t : op_timers[index]) {
              time_sec_per_op[index] += t->SyncAndGetElapsedNanos() / 1e9;
            }
          }
          tend = std::chrono::high_resolution_clock::now();
          duration_ms =
              std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count() *
              1000;
        } while (duration_ms < min_repeat_ms);

        LOG(INFO) << "Iteration: " << i;
        int op = 0;
        for (size_t index = 0; index < time_sec_per_op.size(); index++) {
          if (op_execs_[index]) {
            time_sec_per_op[index] /= number;
            LOG(INFO) << "Op #" << op++ << " " << GetNodeName(index) << ": "
                      << time_sec_per_op[index] * 1e6 << " us/iter";
          }
        }
      }
    }

    std::ostringstream os;
    for (size_t index = 0; index < time_sec_per_op.size(); index++) {
      os << time_sec_per_op[index] << ",";
    }
    return os.str();
  }

  double RunOpRPC(int index, int number, int repeat, int min_repeat_ms) {
    // Right now we expect either "tvm_op" for nodes which run PackedFunc or "null" for nodes which
    // represent inputs/parameters to the graph. Other types may be supported in the future, but
    // consideration would be needed as to how to do that over RPC before we support it here.
    if (nodes_[index].op_type != "tvm_op") {
      CHECK_EQ(nodes_[index].op_type, "null")
          << "Don't know how to run op type " << nodes_[index].op_type
          << " remotely over RPC right now";

      // NOTE: GraphExecutorDebug expects graph nodes to have an "op" attribute of "tvm_op" or
      // "null" and "null" is a placeholder node for a parameter or input.
      return 0;
    }

    const Device& dev = data_entry_[entry_id(index, 0)]->device;
    TVMOpParam param = nodes_[index].param;
    std::string name = param.func_name;
    uint32_t num_inputs = param.num_inputs;
    uint32_t num_outputs = param.num_outputs;

    PackedFunc time_eval = runtime::Registry::Get("runtime.RPCTimeEvaluator")
                               ->
                               operator()(module_, name, static_cast<int>(dev.device_type),
                                          dev.device_id, number, repeat, min_repeat_ms, "");

    int num_flat_args = num_inputs + num_outputs;
    std::unique_ptr<TVMValue> values(new TVMValue[num_flat_args]);
    std::unique_ptr<int> type_codes(new int[num_flat_args]);
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
    std::string results = rv.operator std::string();
    const double* results_arr = reinterpret_cast<const double*>(results.data());
    LOG(INFO) << "Got op timing: " << results_arr[0];
    return results_arr[0];
  }

  Timer RunOpHost(int index) {
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
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name) const {
    for (size_t nid = 0; nid < GetNumOfNodes(); ++nid) {
      if (GetNodeName(nid) == name) {
        return static_cast<int>(nid);
      }
    }
    LOG(FATAL) << "cannot find " << name << " among nodex";
    return -1;
  }

  /*!
   * \brief Execute index-th node in the network.
   *
   * This method will do a partial run of the graph
   * up to index-th node.
   *
   * \param node: The index of the node.
   */
  void ExecuteNode(int node) {
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

  /*!
   * \brief Returns index-th output of node.
   *
   * This method will return index-th out_ind output
   * of index-th node in the network.
   *
   * \param node: The index of the node.
   * \param out_ind: The index of the output.
   * \return Output array.
   */
  NDArray GetNodeOutput(int node, int out_ind) {
    ICHECK_EQ(node, last_executed_node_);
    ICHECK_LT(entry_id(node, out_ind), data_entry_.size());
    return data_entry_[entry_id(node, out_ind)].CopyTo({kDLCPU, 0});
  }

  /*!
   * \brief Copy index-th node to data_out.
   *
   * This method will do a partial run of the graph
   * from begining upto the index-th node and return output of index-th node.
   * This is costly operation and suggest to use only for debug porpose.
   *
   * \param index: The  index of the node.
   * \param data_out the node data.
   */
  void DebugGetNodeOutput(int index, DLTensor* data_out) {
    ICHECK_LT(static_cast<size_t>(index), op_execs_.size());
    uint32_t eid = index;

    for (size_t i = 0; i < op_execs_.size(); ++i) {
      if (op_execs_[i]) op_execs_[i]();
      if (static_cast<int>(i) == index) break;
    }

    data_entry_[eid].CopyTo(data_out);
  }

  /*!
   * \brief Profile execution time of the module.
   *
   * We run the entire module while recording overall and per-op timing
   * information. The module may be run multiple times to ensure everything is
   * warmed up. This function is a more correct reflection of actual runtime of
   * the module compared to GraphRuntimeDebug::RunIndividual as it runs the
   * entire graph in order.
   *
   * \param collectors Optional user defined `MetricCollector`s to use with this profiling run.
   *
   * \returns A table of per-op runtimes and total times.
   */
  profiling::Report Profile(Array<profiling::MetricCollector> collectors) {
    std::vector<profiling::MetricCollector> cs(collectors.begin(), collectors.end());
    profiling::Profiler prof(devices_, cs);

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

 private:
  int last_executed_node_ = -1;
};

/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphExecutorDebug::GetFunction(const std::string& name,
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
      ICHECK_GT(number, 0);
      ICHECK_GT(repeat, 0);
      ICHECK_GE(min_repeat_ms, 0);
      *rv = this->RunIndividual(number, repeat, min_repeat_ms);
    });
  } else if (name == "profile") {
    return TypedPackedFunc<profiling::Report(Array<profiling::MetricCollector>)>(
        [sptr_to_self, this](Array<profiling::MetricCollector> collectors) {
          return this->Profile(collectors);
        });
  } else {
    return GraphExecutor::GetFunction(name, sptr_to_self);
  }
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
