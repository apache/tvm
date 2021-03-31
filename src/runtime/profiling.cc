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

std::string FormatTable(const std::vector<std::unordered_map<std::string, ObjectRef>>& rows,
                        std::unordered_set<std::string> hidden_cols = {"Argument Shapes",
                                                                       "Device"}) {
  std::unordered_set<std::string> unique_headers;

  for (auto row : rows) {
    for (auto p : row) {
      unique_headers.insert(p.first);
    }
  }

  std::vector<std::string> headers = {"Name", "Duration (us)", "Percent"};
  for (auto header : unique_headers) {
    if (header != "Name" && header != "Duration (us)" && header != "Percent" &&
        hidden_cols.find(header) == hidden_cols.end()) {
      headers.push_back(header);
    }
  }

  std::vector<std::vector<std::string>> cols;
  for (auto header : headers) {
    cols.push_back({header});
  }
  for (auto row : rows) {
    for (size_t i = 0; i < headers.size(); i++) {
      auto it = row.find(headers[i]);
      if (it == row.end()) {
        cols[i].push_back("");
      } else {
        std::string val;
        if (it->second.as<CountNode>()) {
          std::stringstream s;
          s.imbue(std::locale(""));  // for 1000s seperators
          s << std::fixed << it->second.as<CountNode>()->value;
          val = s.str();
        } else if (it->second.as<DurationNode>()) {
          std::stringstream s;
          s.imbue(std::locale(""));  // for 1000s seperators
          s << std::fixed << std::setprecision(2) << it->second.as<DurationNode>()->microseconds;
          val = s.str();
        } else if (it->second.as<PercentNode>()) {
          std::stringstream s;
          s << std::fixed << std::setprecision(2) << it->second.as<PercentNode>()->percent;
          val = s.str();
        } else if (it->second.as<StringObj>()) {
          val = Downcast<String>(it->second);
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

String Profiler::Report(bool aggregate, bool sort) {
  std::vector<std::pair<Device, double>> global_times;
  for (auto p : global_timers_) {
    global_times.emplace_back(p.first, p.second->SyncAndGetElapsedNanos() / 1e3);
  }
  double overall_time = 0.;
  for (auto p : global_times) {
    overall_time = std::max(overall_time, p.second);
  }

  // aggregate times by op name
  std::vector<std::pair<std::string, std::vector<size_t>>> aggregate_rows;
  if (aggregate) {
    std::unordered_map<std::string, std::vector<size_t>> aggregates;
    for (size_t i = 0; i < calls_.size(); i++) {
      CallFrame& cf = calls_[i];
      std::string name = cf.name;
      // don't aggregate dynamic ops with different shapes
      auto it = cf.extra_metrics.find("Argument Shapes");
      if (it != cf.extra_metrics.end()) {
        name = name + Downcast<String>(it->second);
      }

      if (aggregates.find(name) == aggregates.end()) {
        aggregates[name] = {i};
      } else {
        aggregates[name].push_back(i);
      }
    }
    for (const auto& p : aggregates) {
      aggregate_rows.push_back(p);
    }
  } else {
    for (size_t i = 0; i < calls_.size(); i++) {
      aggregate_rows.push_back({calls_[i].name, {i}});
    }
  }

  // aggregated rows (poor man's dataframe)
  std::vector<std::unordered_map<std::string, ObjectRef>> rows;

  // form aggregates and compute aggregate statistics (sum).
  for (auto p : aggregate_rows) {
    std::unordered_map<std::string, ObjectRef> row;
    double time_sum = 0;
    size_t count = 0;
    for (auto i : p.second) {
      double us = calls_[i].timer->SyncAndGetElapsedNanos() / 1e3;
      time_sum += us;
      count += 1;
    }
    row["Percent"] = ObjectRef(make_object<PercentNode>(time_sum / overall_time * 100));
    row["Duration (us)"] = ObjectRef(make_object<DurationNode>(time_sum));
    row["Count"] = ObjectRef(make_object<CountNode>(count));
    row["Name"] = calls_[p.second[0]].name;
    Device dev = calls_[p.second[0]].dev;
    row["Device"] = String(DeviceName(dev.device_type) + std::to_string(dev.device_id));

    // assume all rows in the aggregate have the same metrics
    for (auto metric : calls_[p.second[0]].extra_metrics) {
      if (metric.second.as<CountNode>()) {
        int64_t sum = 0;
        for (auto i : p.second) {
          sum += calls_[i].extra_metrics[metric.first].as<CountNode>()->value;
        }
        row[metric.first] = ObjectRef(make_object<CountNode>(sum));
      } else if (metric.second.as<DurationNode>()) {
        double sum = 0;
        for (auto i : p.second) {
          sum += calls_[i].extra_metrics[metric.first].as<DurationNode>()->microseconds;
        }
        row[metric.first] = ObjectRef(make_object<DurationNode>(sum));
      } else if (metric.second.as<PercentNode>()) {
        double sum = 0;
        for (auto i : p.second) {
          sum += calls_[i].extra_metrics[metric.first].as<PercentNode>()->percent;
        }
        row[metric.first] = ObjectRef(make_object<PercentNode>(sum));
      } else if (metric.second.as<StringObj>()) {
        // assume all rows contain the same value for this metric
        row[metric.first] = Downcast<String>(metric.second);
      }
    }

    rows.push_back(row);
  }

  // sort rows by duration
  if (sort) {
    std::sort(rows.begin(), rows.end(),
              [&](const std::unordered_map<std::string, ObjectRef>& a,
                  const std::unordered_map<std::string, ObjectRef>& b) {
                return a.at("Duration (us)").as<DurationNode>()->microseconds >
                       b.at("Duration (us)").as<DurationNode>()->microseconds;
              });
  }

  double op_sum = 0;
  int64_t total_count = 0;
  double per = 0;
  for (auto row : rows) {
    op_sum += row["Duration (us)"].as<DurationNode>()->microseconds;
    total_count += row["Count"].as<CountNode>()->value;
    per += row["Percent"].as<PercentNode>()->percent;
  }

  rows.push_back({{"Name", String("------------------")}});
  rows.push_back({{"Name", String("Total")},
                  {"Duration (us)", ObjectRef(make_object<DurationNode>(op_sum))},
                  {"Count", ObjectRef(make_object<CountNode>(total_count))},
                  {"Percent", ObjectRef(make_object<PercentNode>(per))}});

  std::stringstream s;
  s.imbue(std::locale(""));
  s << FormatTable(rows);
  s << std::fixed << std::setprecision(2);
  for (auto p : global_times) {
    s << "Total time " << DeviceName(p.first.device_type) << p.first.device_id << ": " << p.second
      << "us" << std::endl;
  }
  s << "Overhead: " << overall_time - op_sum << "us  "
    << (overall_time - op_sum) / overall_time * 100 << "%  (Time not spent in operators)";

  return s.str();
}

TVM_REGISTER_OBJECT_TYPE(DurationNode);
TVM_REGISTER_OBJECT_TYPE(PercentNode);
TVM_REGISTER_OBJECT_TYPE(CountNode);
}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
