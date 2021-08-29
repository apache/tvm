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
 * \file src/runtime/profiling.cc
 * \brief Runtime profiling including timers.
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>

namespace tvm {
namespace runtime {

class DefaultTimerNode : public TimerNode {
 public:
  virtual void Start() {
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    start_ = std::chrono::high_resolution_clock::now();
  }
  virtual void Stop() {
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    duration_ = std::chrono::high_resolution_clock::now() - start_;
  }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~DefaultTimerNode() {}

  explicit DefaultTimerNode(Device dev) : device_(dev) {}
  static constexpr const char* _type_key = "DefaultTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(DefaultTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
  Device device_;
};

TVM_REGISTER_OBJECT_TYPE(DefaultTimerNode);
TVM_REGISTER_OBJECT_TYPE(TimerNode);

Timer DefaultTimer(Device dev) { return Timer(make_object<DefaultTimerNode>(dev)); }

class CPUTimerNode : public TimerNode {
 public:
  virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~CPUTimerNode() {}

  static constexpr const char* _type_key = "CPUTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CPUTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};
TVM_REGISTER_OBJECT_TYPE(CPUTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.cpu").set_body_typed([](Device dev) {
  return Timer(make_object<CPUTimerNode>());
});

Timer Timer::Start(Device dev) {
  auto f = Registry::Get(std::string("profiling.timer.") + DeviceName(dev.device_type));
  if (f == nullptr) {
    Timer t = DefaultTimer(dev);
    t->Start();
    return t;
  } else {
    Timer t = f->operator()(dev);
    t->Start();
    return t;
  }
}

TVM_REGISTER_GLOBAL("profiling.start_timer").set_body_typed(Timer::Start);

namespace profiling {

void Profiler::Start(const std::vector<Device>& devs) {
  CHECK(global_timers_.empty()) << "You can only call Start once per Profiler.";
  for (auto dev : devs) {
    global_timers_.emplace_back(dev, Timer::Start(dev));
  }
}

void Profiler::StartCall(String name, Device dev,
                         std::unordered_map<std::string, ObjectRef> extra_metrics) {
  in_flight_.push(CallFrame{dev, name, Timer::Start(dev), extra_metrics});
}

void Profiler::StopCall(std::unordered_map<std::string, ObjectRef> extra_metrics) {
  CallFrame cf = in_flight_.top();
  cf.timer->Stop();
  for (auto& p : extra_metrics) {
    cf.extra_metrics[p.first] = p.second;
  }
  in_flight_.pop();
  calls_.push_back(cf);
}

void Profiler::Stop() {
  // Stop all global timers. We wait to synchronize until we are making the report.
  for (auto p : global_timers_) {
    p.second->Stop();
  }
}

String ShapeString(const std::vector<NDArray>& shapes) {
  std::stringstream sizes;
  for (const NDArray& ary : shapes) {
    if (sizes.tellp() > 0) {
      sizes << ", ";
    }
    auto shape = ary.Shape();
    sizes << ary.DataType() << "[";
    for (size_t i = 0; i < shape.size(); i++) {
      if (i != 0) {
        sizes << ", ";
      }
      sizes << shape[i];
    }
    sizes << "]";
  }
  return String(sizes.str());
}

String ReportNode::AsCSV() const {
  // get unique headers
  std::unordered_set<std::string> unique_headers;

  for (auto row : calls) {
    for (auto p : row) {
      unique_headers.insert(p.first);
    }
  }

  std::vector<std::string> headers;
  for (auto x : unique_headers) {
    headers.push_back(x);
  }

  std::stringstream s;

  for (size_t i = 0; i < headers.size(); i++) {
    std::string header = headers[i];
    s << header;
    if (i < headers.size() - 1) {
      s << ",";
    }
  }
  s << std::endl;
  for (auto row : calls) {
    for (size_t i = 0; i < headers.size(); i++) {
      std::string header = headers[i];
      auto it = row.find(header);
      if (it != row.end()) {
        std::string val;
        if ((*it).second.as<CountNode>()) {
          s << (*it).second.as<CountNode>()->value;
        } else if ((*it).second.as<DurationNode>()) {
          s << (*it).second.as<DurationNode>()->microseconds;
        } else if ((*it).second.as<PercentNode>()) {
          s << (*it).second.as<PercentNode>()->percent;
        } else if ((*it).second.as<StringObj>()) {
          s << "\"" << Downcast<String>((*it).second) << "\"";
        }
      }
      if (i < headers.size() - 1) {
        s << ",";
      }
    }
    s << std::endl;
  }
  return s.str();
}

String ReportNode::AsTable(bool sort, bool aggregate) const {
  // aggregate calls by op hash (or op name if hash is not set) + argument shapes
  std::vector<Map<String, ObjectRef>> aggregated_calls;
  if (aggregate) {
    std::unordered_map<std::string, std::vector<size_t>> aggregates;
    for (size_t i = 0; i < calls.size(); i++) {
      auto& frame = calls[i];
      auto it = frame.find("Hash");
      std::string name = Downcast<String>(frame["Name"]);
      if (it != frame.end()) {
        name = Downcast<String>((*it).second);
      }
      if (frame.find("Argument Shapes") != frame.end()) {
        name += Downcast<String>(frame["Argument Shapes"]);
      }

      if (aggregates.find(name) == aggregates.end()) {
        aggregates[name] = {i};
      } else {
        aggregates[name].push_back(i);
      }
    }
    for (const auto& p : aggregates) {
      std::unordered_map<String, ObjectRef> aggregated;
      for (auto i : p.second) {
        for (auto& metric : calls[i]) {
          auto it = aggregated.find(metric.first);
          if (it == aggregated.end()) {
            aggregated[metric.first] = metric.second;
          } else {
            if (metric.second.as<DurationNode>()) {
              aggregated[metric.first] = ObjectRef(
                  make_object<DurationNode>(it->second.as<DurationNode>()->microseconds +
                                            metric.second.as<DurationNode>()->microseconds));
            } else if (metric.second.as<CountNode>()) {
              aggregated[metric.first] = ObjectRef(make_object<CountNode>(
                  it->second.as<CountNode>()->value + metric.second.as<CountNode>()->value));
            } else if (metric.second.as<PercentNode>()) {
              aggregated[metric.first] =
                  ObjectRef(make_object<PercentNode>(it->second.as<PercentNode>()->percent +
                                                     metric.second.as<PercentNode>()->percent));
            } else if (metric.second.as<StringObj>()) {
              // Don't do anything. Assume the two strings are the same.
            } else {
              LOG(FATAL) << "Can only aggregate metrics with types DurationNode, CountNode, "
                            "PercentNode, and StringObj, but got "
                         << metric.second->GetTypeKey();
            }
          }
        }
      }
      aggregated_calls.push_back(aggregated);
    }
  } else {
    for (auto call : calls) {
      aggregated_calls.push_back(call);
    }
  }

  // sort rows by duration
  if (sort) {
    std::sort(aggregated_calls.begin(), aggregated_calls.end(),
              [&](const Map<String, ObjectRef>& a, const Map<String, ObjectRef>& b) {
                return a.at("Duration (us)").as<DurationNode>()->microseconds >
                       b.at("Duration (us)").as<DurationNode>()->microseconds;
              });
  }

  // compute columnwise sums
  std::unordered_map<String, ObjectRef> col_sums;
  for (auto call : aggregated_calls) {
    for (auto p : call) {
      if (p.second.as<CountNode>()) {
        int64_t val = p.second.as<CountNode>()->value;
        auto it = col_sums.find(p.first);
        if (it != col_sums.end()) {
          val += it->second.as<CountNode>()->value;
        }
        col_sums[p.first] = ObjectRef(make_object<CountNode>(val));
      } else if (p.second.as<DurationNode>()) {
        double val = p.second.as<DurationNode>()->microseconds;
        auto it = col_sums.find(p.first);
        if (it != col_sums.end()) {
          val += it->second.as<DurationNode>()->microseconds;
        }
        col_sums[p.first] = ObjectRef(make_object<DurationNode>(val));
      } else if (p.second.as<PercentNode>()) {
        double val = p.second.as<PercentNode>()->percent;
        auto it = col_sums.find(p.first);
        if (it != col_sums.end()) {
          val += it->second.as<PercentNode>()->percent;
        }
        col_sums[p.first] = ObjectRef(make_object<PercentNode>(val));
      }
    }
  }
  col_sums["Name"] = String("Sum");
  aggregated_calls.push_back({{String("Name"), String("----------")}});  // separator
  aggregated_calls.push_back(col_sums);

  // per-device metrics
  for (auto p : device_metrics) {
    Map<String, ObjectRef> metrics = p.second;
    metrics.Set("Name", String("Total"));
    aggregated_calls.push_back(metrics);
  }

  // Table formatting
  std::unordered_set<std::string> unique_headers;

  for (auto row : aggregated_calls) {
    for (auto p : row) {
      unique_headers.insert(p.first);
    }
  }

  std::vector<std::string> headers = {"Name", "Duration (us)",
                                      "Percent"};  // always include these headers
  for (auto header : unique_headers) {
    if (header != "Name" && header != "Duration (us)" && header != "Percent") {
      headers.push_back(header);
    }
  }

  // Switch layout from row major to column major so we can easily compute column widths.
  std::vector<std::vector<std::string>> cols;
  for (auto header : headers) {
    cols.push_back({header});
  }
  for (auto row : aggregated_calls) {
    for (size_t i = 0; i < headers.size(); i++) {
      auto it = row.find(headers[i]);
      if (it == row.end()) {
        // fill empty data with empty strings
        cols[i].push_back("");
      } else {
        std::string val;
        if ((*it).second.as<CountNode>()) {
          std::stringstream s;
          s.imbue(std::locale(""));  // for 1000s seperators
          s << std::fixed << (*it).second.as<CountNode>()->value;
          val = s.str();
        } else if ((*it).second.as<DurationNode>()) {
          std::stringstream s;
          s.imbue(std::locale(""));  // for 1000s seperators
          s << std::fixed << std::setprecision(2) << (*it).second.as<DurationNode>()->microseconds;
          val = s.str();
        } else if ((*it).second.as<PercentNode>()) {
          std::stringstream s;
          s << std::fixed << std::setprecision(2) << (*it).second.as<PercentNode>()->percent;
          val = s.str();
        } else if ((*it).second.as<StringObj>()) {
          val = Downcast<String>((*it).second);
        }
        cols[i].push_back(val);
      }
    }
  }

  std::vector<size_t> widths;
  for (auto v : cols) {
    size_t width = 0;
    for (auto x : v) {
      width = std::max(width, x.size());
    }
    widths.push_back(width);
  }
  size_t length = 0;
  for (auto v : cols) {
    length = std::max(length, v.size());
  }

  std::stringstream s;
  for (size_t row = 0; row < length; row++) {
    for (size_t col = 0; col < cols.size(); col++) {
      // left align first column
      if (col == 0) {
        s << std::left;
      } else {
        s << std::right;
      }
      if (row < cols[col].size()) {
        s << std::setw(widths[col]) << cols[col][row] << "  ";
      } else {
        s << std::setw(widths[col]) << ""
          << "  ";
      }
    }
    s << std::endl;
  }
  return s.str();
}

std::string DeviceString(Device dev) {
  return DeviceName(dev.device_type) + std::to_string(dev.device_id);
}

Report Profiler::Report(bool aggregate, bool sort) {
  std::vector<std::pair<Device, double>> global_times;
  for (auto p : global_timers_) {
    global_times.emplace_back(p.first, p.second->SyncAndGetElapsedNanos() / 1e3);
  }

  double overall_time = 0;
  for (auto p : global_times) {
    overall_time = std::max(overall_time, p.second);
  }

  std::unordered_map<String, Map<String, ObjectRef>> device_metrics;
  for (auto p : global_times) {
    std::unordered_map<String, ObjectRef> row;
    row["Name"] = String("Total");
    row["Duration (us)"] = ObjectRef(make_object<DurationNode>(p.second));
    row["Percent"] = ObjectRef(make_object<PercentNode>(p.second / overall_time * 100));
    row["Device"] = String(DeviceString(p.first));
    device_metrics[DeviceString(p.first)] = row;
  }

  std::vector<Map<String, ObjectRef>> rows;
  for (auto& cf : calls_) {
    std::unordered_map<String, ObjectRef> row;
    double us = cf.timer->SyncAndGetElapsedNanos() / 1e3;
    row["Percent"] = ObjectRef(make_object<PercentNode>(us / overall_time * 100));
    row["Duration (us)"] = ObjectRef(make_object<DurationNode>(us));
    row["Count"] = ObjectRef(make_object<CountNode>(1));
    row["Name"] = cf.name;
    row["Device"] = String(DeviceString(cf.dev));
    for (auto p : cf.extra_metrics) {
      row[p.first] = p.second;
    }
    rows.push_back(row);
  }

  return profiling::Report(rows, device_metrics);
}

Report::Report(Array<Map<String, ObjectRef>> calls,
               Map<String, Map<String, ObjectRef>> device_metrics) {
  auto node = make_object<ReportNode>();
  node->calls = std::move(calls);
  node->device_metrics = std::move(device_metrics);
  data_ = std::move(node);
}

TVM_REGISTER_OBJECT_TYPE(DurationNode);
TVM_REGISTER_OBJECT_TYPE(PercentNode);
TVM_REGISTER_OBJECT_TYPE(CountNode);
TVM_REGISTER_OBJECT_TYPE(ReportNode);

TVM_REGISTER_GLOBAL("runtime.profiling.AsCSV").set_body_typed([](Report n) { return n->AsCSV(); });
}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
