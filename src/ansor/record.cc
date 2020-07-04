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
 * \file ansor/record.cc
 * \brief Json serialization format for dumping and loading tuning records.
 */

#include "record.h"

#include <dmlc/json.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "loop_state.h"
#include "transform_step.h"
#include "utils.h"

// Json serialization handler for MeasureInput, MeasureResult
// (and recursively for SearchTask, State, Step, ...)
namespace dmlc {
namespace json {

inline std::vector<int>& IntArrayToVector(std::vector<int>* out,
                                          const ::tvm::Array<::tvm::Integer>& data) {
  out->clear();
  for (const auto& x : data) {
    CHECK(x.defined());
    out->push_back(x);
  }
  return *out;
}

template <>
struct Handler<::tvm::Array<::tvm::ansor::Stage>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::Array<::tvm::ansor::Stage>& data) {
    writer->BeginArray(false);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::Array<::tvm::ansor::Stage>* data) {
    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

template <>
struct Handler<::tvm::Array<::tvm::ansor::Step>> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::Array<::tvm::ansor::Step>& data) {
    std::vector<int> tmp;
    writer->BeginArray(false);
    for (size_t i = 0; i < data.size(); ++i) {
      writer->WriteArraySeperator();
      writer->BeginArray(false);
      if (auto ps = data[i].as<::tvm::ansor::ReorderStepNode>()) {
        writer->WriteArrayItem(std::string("RE"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(IntArrayToVector(&tmp, ps->after_ids));
      } else if (auto ps = data[i].as<::tvm::ansor::SplitStepNode>()) {
        writer->WriteArrayItem(std::string("SP"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(ps->extent.defined() ? ::tvm::ansor::GetIntImm(ps->extent) : 0);
        writer->WriteArrayItem(IntArrayToVector(&tmp, ps->lengths));
        writer->WriteArrayItem(static_cast<int>(ps->inner_to_outer));
      } else if (auto ps = data[i].as<::tvm::ansor::FuseStepNode>()) {
        writer->WriteArrayItem(std::string("FU"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(IntArrayToVector(&tmp, ps->fused_ids));
      } else {
        LOG(FATAL) << "Invalid step: " << data[i];
      }
      writer->EndArray();
    }
    writer->EndArray();
  }

  inline static void Read(dmlc::JSONReader* reader, ::tvm::Array<::tvm::ansor::Step>* data) {
    std::vector<int> int_list;
    bool s, inner_to_outer;
    std::string name, scope_name, pragma_type, ti_func_name;
    int stage_id, iter_id, extent;

    reader->BeginArray();
    data->clear();
    while (reader->NextArrayItem()) {
      reader->BeginArray();
      s = reader->NextArrayItem();
      CHECK(s);
      reader->Read(&name);
      if (name == "RE") {
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&int_list);
        ::tvm::Array<::tvm::Integer> after_ids;
        for (const auto& i : int_list) {
          after_ids.push_back(i);
        }
        data->push_back(::tvm::ansor::ReorderStep(stage_id, after_ids));
      } else if (name == "SP") {
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&extent);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&int_list);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&inner_to_outer);
        ::tvm::Array<::tvm::Integer> lengths;
        for (const auto& i : int_list) {
          lengths.push_back(i);
        }
        data->push_back(::tvm::ansor::SplitStep(
            stage_id, iter_id, extent == 0 ? ::tvm::PrimExpr() : extent, lengths, inner_to_outer));
      } else if (name == "FU") {
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem();
        CHECK(s);
        reader->Read(&int_list);
        ::tvm::Array<::tvm::Integer> fused_ids;
        for (const auto& i : int_list) {
          fused_ids.push_back(i);
        }
        data->push_back(::tvm::ansor::FuseStep(stage_id, fused_ids));
      } else {
        LOG(FATAL) << "Invalid step format";
      }
      s = reader->NextArrayItem();
      CHECK(!s);
    }
  }
};

template <>
struct Handler<::tvm::ansor::StateNode> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::ansor::StateNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(data.stages);
    writer->WriteArrayItem(data.transform_steps);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::ansor::StateNode* data) {
    reader->BeginArray();
    bool s;
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
struct Handler<::tvm::ansor::SearchTaskNode> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::ansor::SearchTaskNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(std::string(data.workload_key));
    writer->WriteArrayItem(data.target->str());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::ansor::SearchTaskNode* data) {
    std::string target_str;
    bool s;

    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&target_str);
    data->workload_key = std::move(target_str);
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&target_str);
    data->target = ::tvm::Target::Create(target_str);
    s = reader->NextArrayItem();
    CHECK(!s);
  }
};

template <>
struct Handler<::tvm::ansor::MeasureInputNode> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::ansor::MeasureInputNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(*data.task.operator->());
    writer->WriteArrayItem(*data.state.operator->());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::ansor::MeasureInputNode* data) {
    bool s;
    auto task_node = ::tvm::make_object<::tvm::ansor::SearchTaskNode>();
    auto state_node = ::tvm::make_object<::tvm::ansor::StateNode>();
    state_node->complete = true;

    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(task_node.get());
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(state_node.get());
    s = reader->NextArrayItem();
    CHECK(!s);

    data->task = ::tvm::ansor::SearchTask(task_node);
    data->state = ::tvm::ansor::State(state_node);
  }
};

template <>
struct Handler<::tvm::ansor::MeasureResultNode> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::ansor::MeasureResultNode& data) {
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
  inline static void Read(dmlc::JSONReader* reader, ::tvm::ansor::MeasureResultNode* data) {
    bool s;
    std::vector<double> tmp;

    reader->BeginArray();
    s = reader->NextArrayItem();
    CHECK(s);
    reader->Read(&tmp);
    data->costs.clear();
    for (const auto& i : tmp) {
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
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(LogToFileNode);
TVM_REGISTER_OBJECT_TYPE(LogReaderNode);

const std::string ANSOR_LOG_VERSION = "v0.2";  // NOLINT(*)

LogToFile::LogToFile(String filename) {
  auto node = make_object<LogToFileNode>();
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
    writer.WriteObjectKeyValue("v", ANSOR_LOG_VERSION);
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

void LogToFileNode::Callback(const SearchPolicy& policy, const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {
  std::ofstream ofs(filename, std::ofstream::app);
  WriteMeasureRecords(&ofs, inputs, results);
}

LogReader::LogReader(String filename) {
  auto node = make_object<LogReaderNode>();
  node->filename = filename;
  node->infile.open(filename, std::ifstream::in);
  data_ = std::move(node);
}

LogReaderNode::~LogReaderNode() { infile.close(); }

bool LogReaderNode::ReadNext(MeasureInputNode* inp, MeasureResultNode* res) {
  std::string log_version;

  while (std::getline(infile, cur_line)) {
    if (cur_line[0] == '#' || cur_line[0] == ' ') {
      // skip comment lines begin with '#' or ' '
      continue;
    }
    ReadMeasureRecord(cur_line, inp, res, &log_version);
    return true;
  }

  return false;
}

std::pair<Array<MeasureInput>, Array<MeasureResult>> LogReaderNode::ReadLines(int max_size,
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

TVM_REGISTER_GLOBAL("ansor.LogToFile").set_body_typed([](const String& filename) {
  return LogToFile(filename);
});

TVM_REGISTER_GLOBAL("ansor.LogReader").set_body_typed([](const String& filename) {
  return LogReader(filename);
});

TVM_REGISTER_GLOBAL("ansor.LogReaderReadLines")
    .set_body_typed([](LogReader reader, int size, int skip_size) {
      const auto& res = reader->ReadLines(size, skip_size);
      return Array<ObjectRef>{res.first, res.second};
    });

TVM_REGISTER_GLOBAL("ansor.LogReaderReadNext").set_body_typed([](LogReader reader) {
  auto inp = make_object<MeasureInputNode>();
  auto res = make_object<MeasureResultNode>();
  if (reader->ReadNext(inp.get(), res.get())) {
    return Array<ObjectRef>{ObjectRef(inp), ObjectRef(res)};
  } else {
    return Array<ObjectRef>();
  }
});

TVM_REGISTER_GLOBAL("ansor.AppendMeasureRecordsToFile")
    .set_body_typed([](String filename, Array<MeasureInput> in, Array<MeasureResult> res) {
      std::ofstream ofs(filename, std::ofstream::app);
      WriteMeasureRecords(&ofs, in, res);
    });
}  // namespace ansor
}  // namespace tvm
