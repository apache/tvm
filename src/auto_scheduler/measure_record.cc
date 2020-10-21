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
 * \file auto_scheduler/measure_record.cc
 * \brief Json serialization format for dumping and loading tuning records.
 */

#include <dmlc/json.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

// Json serialization handler for MeasureInput, MeasureResult
// (and recursively for SearchTask, State, Step, ...)
namespace dmlc {
namespace json {

template <>
struct Handler<::tvm::Array<::tvm::auto_scheduler::Stage>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::Array<::tvm::auto_scheduler::Stage>& data) {
    writer->BeginArray(false);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::Array<::tvm::auto_scheduler::Stage>* data) {
    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

template <>
struct Handler<::tvm::Array<::tvm::auto_scheduler::Step>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::Array<::tvm::auto_scheduler::Step>& data) {
    writer->BeginArray(false);
    for (const auto& step : data) {
      writer->WriteArraySeperator();
      writer->BeginArray(false);
      step->WriteToRecord(writer);
      writer->EndArray();
    }
    writer->EndArray();
  }

  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::Array<::tvm::auto_scheduler::Step>* data) {
    bool s;
    reader->BeginArray();
    data->clear();
    while (reader->NextArrayItem()) {
      reader->BeginArray();
      data->push_back(::tvm::auto_scheduler::StepReadFromRecord(reader));
      s = reader->NextArrayItem();
      CHECK(!s);
    }
  }
};

template <>
struct Handler<::tvm::auto_scheduler::StateNode> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::auto_scheduler::StateNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(data.stages);
    writer->WriteArrayItem(data.transform_steps);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::auto_scheduler::StateNode* data) {
    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&data->stages);
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&data->transform_steps);
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

template <>
struct Handler<::tvm::auto_scheduler::SearchTaskNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::auto_scheduler::SearchTaskNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(std::string(data.workload_key));
    writer->WriteArrayItem(data.target->str());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::auto_scheduler::SearchTaskNode* data) {
    bool s;
    std::string str_value;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&str_value);
    data->workload_key = std::move(str_value);
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&str_value);
    data->target = ::tvm::Target(str_value);
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

template <>
struct Handler<::tvm::auto_scheduler::MeasureInputNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::auto_scheduler::MeasureInputNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(*data.task.operator->());
    writer->WriteArrayItem(*data.state.operator->());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::auto_scheduler::MeasureInputNode* data) {
    auto task_node = ::tvm::make_object<::tvm::auto_scheduler::SearchTaskNode>();
    auto state_node = ::tvm::make_object<::tvm::auto_scheduler::StateNode>();
    state_node->concrete = true;

    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(task_node.get());
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(state_node.get());
    s = reader->NextArrayItem();
    CHECK(!s);

    data->task = ::tvm::auto_scheduler::SearchTask(task_node);
    data->state = ::tvm::auto_scheduler::State(state_node);
  }
};

template <>
struct Handler<::tvm::auto_scheduler::MeasureResultNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::auto_scheduler::MeasureResultNode& data) {
    writer->BeginArray(false);
    writer->WriteArraySeperator();
    writer->BeginArray(false);
    for (const auto& x : data.costs) {
      auto pf = x.as<::tvm::tir::FloatImmNode>();
      CHECK(pf != nullptr) << "Cost can only contain float values";
      writer->WriteArrayItem(pf->value);
    }
    writer->EndArray();
    writer->WriteArrayItem(data.error_no);
    writer->WriteArrayItem(data.all_cost);
    writer->WriteArrayItem(static_cast<int>((data.timestamp)));
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::auto_scheduler::MeasureResultNode* data) {
    std::vector<double> double_list;
    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&double_list);
    data->costs.clear();
    for (const auto& i : double_list) {
      data->costs.push_back(::tvm::FloatImm(::tvm::DataType::Float(64), i));
    }
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&data->error_no);
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&data->all_cost);
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&data->timestamp);
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

}  // namespace json
}  // namespace dmlc

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(RecordToFileNode);
TVM_REGISTER_OBJECT_TYPE(RecordReaderNode);

const std::string AUTO_SCHEDULER_LOG_VERSION = "v0.2";  // NOLINT(*)

RecordToFile::RecordToFile(String filename) {
  auto node = make_object<RecordToFileNode>();
  node->filename = std::move(filename);
  data_ = std::move(node);
}

void WriteMeasureRecords(std::ostream* os, const Array<MeasureInput>& inputs,
                         const Array<MeasureResult>& results) {
  dmlc::JSONWriter writer(os);
  for (size_t i = 0; i < inputs.size(); ++i) {
    writer.BeginObject(false);
    writer.WriteObjectKeyValue("i", *inputs[i].operator->());
    writer.WriteObjectKeyValue("r", *results[i].operator->());
    writer.WriteObjectKeyValue("v", AUTO_SCHEDULER_LOG_VERSION);
    writer.EndObject();
    *os << "\n";
  }
}

void ReadMeasureRecord(const std::string& str, MeasureInputNode* inp, MeasureResultNode* res,
                       std::string* log_version) {
  std::istringstream ss(str);
  dmlc::JSONReader reader(&ss);
  std::string key;

  reader.BeginObject();
  while (reader.NextObjectItem(&key)) {
    if (key == "i") {
      reader.Read(inp);
    } else if (key == "r") {
      reader.Read(res);
    } else if (key == "v") {
      reader.Read(log_version);
    } else {
      LOG(FATAL) << "Invalid key in json log: " << key;
    }
  }
}

void RecordToFileNode::Callback(const SearchPolicy& policy, const Array<MeasureInput>& inputs,
                                const Array<MeasureResult>& results) {
  std::ofstream ofs(filename, std::ofstream::app);
  WriteMeasureRecords(&ofs, inputs, results);
}

RecordReader::RecordReader(String filename) {
  auto node = make_object<RecordReaderNode>();
  node->filename = filename;
  node->infile.open(filename, std::ifstream::in);
  data_ = std::move(node);
}

RecordReaderNode::~RecordReaderNode() { infile.close(); }

bool RecordReaderNode::ReadNext(MeasureInputNode* inp, MeasureResultNode* res) {
  std::string log_version;

  while (std::getline(infile, cur_line_)) {
    if (cur_line_[0] == '#' || cur_line_[0] == ' ') {
      // skip comment lines begin with '#' or ' '
      continue;
    }
    ReadMeasureRecord(cur_line_, inp, res, &log_version);
    return true;
  }

  return false;
}

std::pair<Array<MeasureInput>, Array<MeasureResult>> RecordReaderNode::ReadLines(int max_size,
                                                                                 int skip_size) {
  auto inp = make_object<MeasureInputNode>();
  auto res = make_object<MeasureResultNode>();
  Array<MeasureInput> inputs;
  Array<MeasureResult> results;

  while (ReadNext(inp.get(), res.get())) {
    if (skip_size > 0) {
      skip_size--;
      continue;
    }

    inputs.push_back(inp->copy());
    results.push_back(res->copy());

    if (max_size > 0 && static_cast<int>(inputs.size()) >= max_size) {
      break;
    }
  }

  return std::make_pair(inputs, results);
}

TVM_REGISTER_GLOBAL("auto_scheduler.RecordToFile").set_body_typed([](const String& filename) {
  return RecordToFile(filename);
});

TVM_REGISTER_GLOBAL("auto_scheduler.RecordReader").set_body_typed([](const String& filename) {
  return RecordReader(filename);
});

TVM_REGISTER_GLOBAL("auto_scheduler.RecordReaderReadLines")
    .set_body_typed([](RecordReader reader, int size, int skip_size) {
      const auto& res = reader->ReadLines(size, skip_size);
      return Array<ObjectRef>{res.first, res.second};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.RecordReaderReadNext").set_body_typed([](RecordReader reader) {
  auto inp = make_object<MeasureInputNode>();
  auto res = make_object<MeasureResultNode>();
  if (reader->ReadNext(inp.get(), res.get())) {
    return Array<ObjectRef>{ObjectRef(inp), ObjectRef(res)};
  } else {
    return Array<ObjectRef>();
  }
});

TVM_REGISTER_GLOBAL("auto_scheduler.SaveRecords")
    .set_body_typed([](String filename, Array<MeasureInput> in, Array<MeasureResult> res) {
      std::ofstream ofs(filename, std::ofstream::app);
      WriteMeasureRecords(&ofs, in, res);
    });
}  // namespace auto_scheduler
}  // namespace tvm
