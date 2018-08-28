/*!
 *  Copyright (c) 2018 by Contributors
 * \file source_map.cc
 * \brief Source maps for Relay.
 */

#include <tvm/relay/logging.h>
#include <tvm/relay/source_map.h>
#include <iostream>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

SourceFragment::SourceFragment(std::string file_name, std::string source)
    : file_name(file_name), source_lines({}) {
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

std::string SourceFragment::SourceAt(Span sp, int max_lines) {
  std::stringstream out;

  // We need to move from 1 based indexing to zero based indexing.
  int starting_line = sp->lineno;

  if (starting_line >= static_cast<int>(this->source_lines.size())) {
    throw dmlc::Error("SourceFragment: index out of bounds");
  }

  auto lines = std::max(static_cast<size_t>(max_lines), source_lines.size() - starting_line);

  for (size_t i = 0; i < lines; i++) {
    out << std::endl << this->source_lines.at(starting_line + i);
  }

  auto source_slice = out.str();

  RELAY_LOG(INFO) << "SourceFragment::SourceAt: source_slice=" << source_slice << std::endl;
  return source_slice;
}

SourceName SourceMap::AddSource(std::string file_name, std::string source) {
  auto new_id = SourceNameNode::make(file_name);
  SourceFragment sfile(file_name, source);
  this->map_.insert({new_id, sfile});
  return new_id;
}

const SourceFragment& SourceMap::GetSource(SourceName id) const {
  auto item = map_.find(id);
  if (item != map_.end()) {
    return (*item).second;
  } else {
      throw dmlc::Error("could not find requested source fragment");
  }
}

}  // namespace relay
}  // namespace tvm