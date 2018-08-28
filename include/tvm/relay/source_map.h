/*!
 *  Copyright (c) 2018 by Contributors
 * \file source_map.h
 * \brief A representation of source files and a data structure for 
 * storing them.
 */
#ifndef TVM_RELAY_SOURCE_MAP_H_
#define TVM_RELAY_SOURCE_MAP_H_

#include <tvm/relay/expr.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

struct SourceFragment {
  std::string file_name;
  std::vector<std::string> source_lines;

  SourceFragment(std::string file_name, std::string source);

  SourceFragment(const SourceFragment& sf) {
    this->file_name = sf.file_name;
    this->source_lines = sf.source_lines;
  }

  std::string SourceAt(Span sp, int lines);
};

/*! \brief Maps from FileId's to a SourceFragment.
 */
class SourceMap {
  /*! \brief Map from unique token to a fragment of a source file. */
  std::unordered_map<SourceName, SourceFragment, NodeHash> map_;
 public:
  SourceMap() : map_() {}
  SourceName AddSource(std::string file_name, std::string source);
  const SourceFragment & GetSource(SourceName id) const;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_SOURCE_MAP_H_