/*!
 *  Copyright (c) 2018 by Contributors
 * \file source_map.h
 * \brief A representation of source files and a data structure for
 * storing them.
 */
#ifndef TVM_RELAY_SOURCE_MAP_H_
#define TVM_RELAY_SOURCE_MAP_H_

#include <tvm/relay/base.h>
#include <tvm/relay/expr.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

struct SourceFragment {
  SourceName name;
  std::vector<std::string> source_lines;

  SourceFragment(const SourceName& file_name, const std::string& source);

  SourceFragment(const SourceFragment& sf) {
    this->name = sf.name;
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
  SourceName AddSource(const std::string& file_name, const std::string& source);
  SourceName AddSource(const SourceName& source_name, const std::string& source);
  const SourceFragment & GetSource(SourceName id) const;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_SOURCE_MAP_H_
