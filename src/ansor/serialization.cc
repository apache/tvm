/*!
 *  Copyright (c) 2020 by Contributors
 */
#include <dmlc/json.h>
// #include <tvm/build_module.h>
#include <tvm/runtime/registry.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include "serialization.h"
#include "loop_state.h"
#include "utils.h"

// Json serialization handler for MeasureInput, MeasureResult
// (and recursively SearchTask, State, Step, ...
namespace dmlc {
namespace json {

inline std::vector<double>& FloatArrayToVector(std::vector<double>* out,
                                               const ::tvm::Array<::tvm::PrimExpr>& data) {
  out->clear();
  for (const auto&x : data) {
    auto pf = x.as<::tvm::tir::FloatImmNode>();
    CHECK(pf != nullptr) << "Cost can only contain float values";
    out->push_back(pf->value);
  }
  return *out;
}

inline std::vector<int>& IntArrayToVector(std::vector<int>* out,
                                          const ::tvm::Array<::tvm::PrimExpr>& data) {
  out->clear();
  for (const auto&x : data) {
    auto pi = x.as<::tvm::tir::IntImmNode>();
    CHECK(pi != nullptr) << "Cost can only contain int values";
    out->push_back(pi->value);
  }
  return *out;
}

template <>
struct Handler<std::vector<::tvm::ansor::Stage> > {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::vector<::tvm::ansor::Stage> & data) {
    // todo(lmzheng): support serialization of Stage
    writer->BeginArray(false);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::vector<::tvm::ansor::Stage> * data) {
    bool s;
    reader->BeginArray();
    s = reader->NextArrayItem(); CHECK(!s);
  }
};

template <>
struct Handler<std::vector<::tvm::ansor::Step> > {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::vector<::tvm::ansor::Step> & data) {
    std::vector<int> tmp;
    writer->BeginArray(false);
    for (size_t i = 0; i < data.size(); ++i) {
      writer->WriteArraySeperator();
      writer->BeginArray(false);
      if (auto ps = data[i].as<::tvm::ansor::ReorderStepNode>()) {
        writer->WriteArrayItem(std::string("RS"));
        writer->WriteArrayItem(ps->stage_id);

        writer->WriteArraySeperator();
        writer->BeginArray(false);
        for (int x : ps->after_ids) {
          writer->WriteArrayItem(x);
        }
        writer->EndArray();
      } else if (auto ps = data[i].as<::tvm::ansor::SplitStepNode>()) {
        writer->WriteArrayItem(std::string("SS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        if (ps->extent.defined()) {
          writer->WriteArrayItem(::tvm::ansor::GetIntImm(ps->extent));
        } else {
          writer->WriteArrayItem(0);
        }
        writer->WriteArrayItem(IntArrayToVector(&tmp, ps->lengths));
        writer->WriteArrayItem(static_cast<int>(ps->inner_to_outer));
      } else if (auto ps = data[i].as<::tvm::ansor::FollowSplitStepNode>()) {
        writer->WriteArrayItem(std::string("FSS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(ps->src_step_id);
        writer->WriteArrayItem(ps->n_split);
      } else if (auto ps = data[i].as<::tvm::ansor::FollowFusedSplitStepNode>()) {
        writer->WriteArrayItem(std::string("FFSS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);

        writer->WriteArraySeperator();
        writer->BeginArray(false);
        for (int x : ps->src_step_ids) {
          writer->WriteArrayItem(x);
        }
        writer->EndArray();

        writer->WriteArrayItem(ps->level);
        writer->WriteArrayItem(static_cast<int>(ps->factor_or_nparts));
      } else if (auto ps = data[i].as<::tvm::ansor::FuseStepNode>()) {
        writer->WriteArrayItem(std::string("FS"));
        writer->WriteArrayItem(ps->stage_id);

        writer->WriteArraySeperator();
        writer->BeginArray(false);
        for (int x : ps->fused_ids) {
          writer->WriteArrayItem(x);
        }
        writer->EndArray();
      } else if (auto ps = data[i].as<::tvm::ansor::AnnotationStepNode>()) {
        writer->WriteArrayItem(std::string("AS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(static_cast<int>(ps->annotation));
      } else if (auto ps = data[i].as<::tvm::ansor::ComputeAtStepNode>()) {
        writer->WriteArrayItem(std::string("CA"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->target_stage_id);
        writer->WriteArrayItem(ps->target_iter_id);
      } else if (auto ps = data[i].as<::tvm::ansor::ComputeRootStepNode>()) {
        writer->WriteArrayItem(std::string("CR"));
        writer->WriteArrayItem(ps->stage_id);
      } else if (auto ps = data[i].as<::tvm::ansor::ComputeInlineStepNode>()) {
        writer->WriteArrayItem(std::string("CI"));
        writer->WriteArrayItem(ps->stage_id);
      } else if (auto ps = data[i].as<::tvm::ansor::CacheReadStepNode>()) {
        writer->WriteArrayItem(std::string("CHR"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->scope_name);
        writer->WriteArrayItem(ps->reader_stage_ids);
      } else if (auto ps = data[i].as<::tvm::ansor::CacheWriteStepNode>()) {
        writer->WriteArrayItem(std::string("CHW"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->scope_name);
      } else if (auto ps = data[i].as<::tvm::ansor::PragmaStepNode>()) {
        writer->WriteArrayItem(std::string("PS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(ps->pragma_type);
      } else if (auto ps = data[i].as<::tvm::ansor::RfactorStepNode>()) {
        writer->WriteArrayItem(std::string("RFS"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(ps->factor_iter_id);
      } else if (auto ps = data[i].as<::tvm::ansor::StorageAlignStepNode>()) {
        writer->WriteArrayItem(std::string("SA"));
        writer->WriteArrayItem(ps->stage_id);
        writer->WriteArrayItem(ps->iter_id);
        writer->WriteArrayItem(ps->factor);
        writer->WriteArrayItem(ps->offset);
      } else {
        LOG(FATAL) << "Invalid step: " << data[i];
      }
      writer->EndArray();
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::vector<::tvm::ansor::Step> * data) {
    std::vector<int> int_list;
    bool s, inner_to_outer, factor_or_nparts;
    std::string name, scope_name, pragma_type;
    int stage_id, target_stage_id, iter_id, src_step_id, n_split, ann, extent;
    int level, factor_iter_id, factor, offset;

    reader->BeginArray();
    data->clear();
    while (reader->NextArrayItem()) {
      reader->BeginArray();
      s = reader->NextArrayItem(); CHECK(s);
      reader->Read(&name);
      if (name == "RS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&int_list);
        data->push_back(::tvm::ansor::ReorderStepNode::make(stage_id, int_list));
      } else if (name == "SS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&extent);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&int_list);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&inner_to_outer);
        data->push_back(::tvm::ansor::SplitStepNode::make(
            stage_id, iter_id, extent,
            std::vector<::tvm::PrimExpr>(int_list.begin(), int_list.end()),
            inner_to_outer));
      } else if (name == "FSS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&src_step_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&n_split);
        data->push_back(::tvm::ansor::FollowSplitStepNode::make(
            stage_id, iter_id, src_step_id, n_split));
      } else if (name == "FFSS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&int_list);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&level);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&factor_or_nparts);
        data->push_back(::tvm::ansor::FollowFusedSplitStepNode::make(
            stage_id, iter_id, int_list, level, factor_or_nparts));
      } else if (name == "FS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&int_list);
        data->push_back(::tvm::ansor::FuseStepNode::make(stage_id, int_list));
      } else if (name == "AS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&ann);
        data->push_back(::tvm::ansor::AnnotationStepNode::make(stage_id,
            iter_id, ::tvm::ansor::IteratorAnnotation(ann)));
      } else if (name == "CA") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&target_stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        data->push_back(::tvm::ansor::ComputeAtStepNode::make(
            stage_id, target_stage_id, iter_id));
      } else if (name == "CR") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        data->push_back(::tvm::ansor::ComputeRootStepNode::make(stage_id));
      } else if (name == "CI") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        data->push_back(::tvm::ansor::ComputeInlineStepNode::make(stage_id));
      } else if (name == "CHR") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&scope_name);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&int_list);
        data->push_back(::tvm::ansor::CacheReadStepNode::make(
            stage_id, scope_name, int_list));
      } else if (name == "CHW") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&scope_name);
        data->push_back(::tvm::ansor::CacheWriteStepNode::make(
            stage_id, scope_name));
      } else if (name == "PS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&pragma_type);
        data->push_back(::tvm::ansor::PragmaStepNode::make(
            stage_id, iter_id, pragma_type));
      } else if (name == "RFS") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&factor_iter_id);
        data->push_back(::tvm::ansor::RfactorStepNode::make(
            stage_id, iter_id, factor_iter_id));
      } else if (name == "SA") {
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&stage_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&iter_id);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&factor);
        s = reader->NextArrayItem(); CHECK(s);
        reader->Read(&offset);
        data->push_back(::tvm::ansor::StorageAlignStepNode::make(
            stage_id, iter_id, factor, offset));
      } else {
        LOG(FATAL) << "Invalid step format";
      }
      s = reader->NextArrayItem(); CHECK(!s);
    }
  }
};

template <>
struct Handler<::tvm::ansor::StateNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::ansor::StateNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(data.stages);
    writer->WriteArrayItem(data.transform_steps);
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::ansor::StateNode* data) {
    reader->BeginArray();
    bool s;
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->stages);
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->transform_steps);
    s = reader->NextArrayItem(); CHECK(!s);
  }
};

template <>
struct Handler<::tvm::ansor::SearchTaskNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::ansor::SearchTaskNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(data.workload_key);
    writer->WriteArrayItem(data.target->str());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::ansor::SearchTaskNode* data) {
    std::string target_str;
    bool s;

    reader->BeginArray();
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->workload_key);
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&target_str);
    data->target = ::tvm::Target::Create(target_str);
    s = reader->NextArrayItem(); CHECK(!s);
  }
};

template <>
struct Handler<::tvm::ansor::MeasureInputNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::ansor::MeasureInputNode& data) {
    writer->BeginArray(false);
    writer->WriteArrayItem(*data.task.operator->());
    writer->WriteArrayItem(*data.state.operator->());
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::ansor::MeasureInputNode* data) {
    bool s;
    auto task_node = ::tvm::make_object<::tvm::ansor::SearchTaskNode>();
    auto state_node = ::tvm::make_object<::tvm::ansor::StateNode>();
    state_node->complete = true;

    reader->BeginArray();
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(task_node.get());
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(state_node.get());
    s = reader->NextArrayItem(); CHECK(!s);

    data->task = ::tvm::ansor::SearchTask(task_node);
    data->state = ::tvm::ansor::State(state_node);
  }
};

template <>
struct Handler<::tvm::ansor::MeasureResultNode> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::ansor::MeasureResultNode& data) {
    writer->BeginArray(false);
    writer->WriteArraySeperator();
    writer->BeginArray(false);
    for (const auto&x : data.costs) {
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
                          ::tvm::ansor::MeasureResultNode* data) {
    bool s;
    std::vector<double> tmp;

    reader->BeginArray();
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&tmp);
    data->costs.clear();
    for (const auto& i : tmp) {
      data->costs.push_back(i);
    }
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->error_no);
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->all_cost);
    s = reader->NextArrayItem(); CHECK(s);
    reader->Read(&data->timestamp);
    s = reader->NextArrayItem(); CHECK(!s);
  }
};

}  // namespace json
}  // namespace dmlc

namespace tvm {
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(LogToFileNode);
TVM_REGISTER_OBJECT_TYPE(LogReaderNode);

const std::string ansor_LOG_VERSION = "v0.1";    // NOLINT(*)

MeasureCallback LogToFileNode::make(std::string filename) {
  auto node = make_object<LogToFileNode>();
  node->filename = std::move(filename);
  return MeasureCallback(node);
}

void WriteMeasureRecords(std::ostream* os,
                         const Array<MeasureInput>& inputs,
                         const Array<MeasureResult>& results) {
  dmlc::JSONWriter writer(os);
  for (size_t i = 0; i < inputs.size(); ++i) {
    writer.BeginObject(false);
    writer.WriteObjectKeyValue("i", *inputs[i].operator->());
    writer.WriteObjectKeyValue("r", *results[i].operator->());
    writer.WriteObjectKeyValue("v", ansor_LOG_VERSION);
    writer.EndObject();
    *os << "\n";
  }
}

void ReadMeasureRecords(std::string str,
                        MeasureInputNode* inp,
                        MeasureResultNode* res,
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

TVM_REGISTER_GLOBAL("ansor.write_measure_records_to_file")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  std::string filename = args[0];
  Array<MeasureInput> in = args[1];
  Array<MeasureResult> res = args[2];
  std::ofstream ofs(filename, std::ofstream::app);
  WriteMeasureRecords(&ofs, in, res);
});

void LogToFileNode::callback(const SearchPolicy& policy,
                             const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {
  std::ofstream ofs(filename, std::ofstream::app);
  WriteMeasureRecords(&ofs, inputs, results);
}

LogReader LogReaderNode::make(std::string filename) {
  auto node = make_object<LogReaderNode>();
  node->filename = filename;
  node->infile.open(filename, std::ifstream::in);
  return LogReader(node);
}

bool LogReaderNode::ReadNext(MeasureInputNode* inp, MeasureResultNode* res) {
  std::string log_version;

  while (std::getline(infile, cur_line)) {
    if (cur_line[0] == '#' || cur_line[0] == ' ') {
      // skip comment lines begin with '#' or ' '
      continue;
    }

    try {
      ReadMeasureRecords(cur_line, inp, res, &log_version);
    } catch (...) {
      return false;
    }

    return true;
  }

  return false;
}

std::pair<Array<MeasureInput>, Array<MeasureResult> > LogReaderNode::ReadLines(
        int max_size, int skip_size) {
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

std::pair<MeasureInput, MeasureResult> BestMeasurePairInFile(const std::string& filename,
                                                             const std::string& workload_key,
                                                             const Target& target) {
  std::pair<MeasureInput, MeasureResult> best_pair;
  double best_cost = 1e30;

  auto inp = make_object<MeasureInputNode>();
  auto res = make_object<MeasureResultNode>();
  LogReader reader = LogReaderNode::make(filename);

  while (reader->ReadNext(inp.get(), res.get())) {
    if (res->error_no != kNoError || inp->task->workload_key != workload_key
       || inp->task->target->target_name != target->target_name) {
      continue;
    }

    double cost = FloatArrayMean(res->costs);

    if (cost < best_cost) {
      best_cost = cost;
      best_pair = std::make_pair(inp->copy(), res->copy());
    }
  }

  return best_pair;
}

}  // namespace ansor
}  // namespace tvm