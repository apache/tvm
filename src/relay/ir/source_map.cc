/*!
 *  Copyright (c) 2018 by Contributors
 * \file source_map.cc
 * \brief Source maps for Relay.
 */

#include <tvm/relay/source_map.h>
#include <tvm/relay/logging.h>
#include <iostream>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

SourceFragment::SourceFragment(const SourceName& name, const std::string& source)
    : name(name), source_lines({}) {
  RELAY_LOG(INFO)<< "SourceFragment::SourceFragment source=" << source << std::endl;
  std::stringstream source_stream;
  source_stream.str(source.c_str());
  std::string line;

  while (std::getline(source_stream, line)) {
    RELAY_LOG(INFO) << "SourceFragment::SourceFragment: line=" << line << std::endl;
    std::string copy(line);
    source_lines.push_back(copy);
  }
}

std::vector<std::string> SourceFragment::LinesAt(Span sp, int max_lines) {
  // We need to move from 1 based indexing to zero based indexing.
  int starting_line = sp->lineno;

  if (starting_line >= static_cast<int>(this->source_lines.size())) {
    throw dmlc::Error("SourceFragment: index out of bounds");
  }

  auto num_of_lines =
    std::max(static_cast<size_t>(max_lines),
      source_lines.size() - starting_line);

  std::vector<std::string> lines;
  for (size_t i = 0; i < num_of_lines; i++) {
    lines.push_back(this->source_lines.at(starting_line + i));
  }

  // RELAY_LOG(INFO) << "SourceFragment::SourceAt: source_slice=" << source_slice << std::endl;
  return lines;
}

SourceName SourceMap::AddSource(const SourceName& source_name, const std::string& source) {
  SourceFragment sfile(source_name, source);
  this->map_.insert({source_name, sfile});
  return source_name;
}

SourceName SourceMap::AddSource(const std::string& file_name, const std::string& source) {
  auto source_name = SourceName::Get(file_name);
  return this->AddSource(source_name, source);
}

const SourceFragment& SourceMap::GetSource(SourceName id) const {
  auto item = map_.find(id);
  if (item != map_.end()) {
    return (*item).second;
  } else {
      LOG(FATAL) << "could not find requested source fragment" << id;
  }
}

}  // namespace relay
}  // namespace tvm
