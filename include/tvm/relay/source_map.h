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

/*! \brief A fragment of a source file used for error reporting.
 *
 * These can be registered by the frontends and are used for
 * displaying errors.
 */
struct SourceFragment {
  /*! \brief The file name which the source fragment originates from. */
  std::string file_name;
  /*! \brief The lines of source corresponding to the fragment. */
  std::vector<std::string> source_lines;

  SourceFragment(const std::string& file_name, const std::string& source);

  SourceFragment(const SourceFragment& sf) {
    this->file_name = sf.file_name;
    this->source_lines = sf.source_lines;
  }

  /*! \brief The lines of source code originate at lines. */
  std::string SourceAt(Span sp, int lines);
};

/*! \brief Maps from FileId's to a SourceFragment.
 */
class SourceMap {
  /*! \brief Map from unique token to a fragment of a source file. */
  std::unordered_map<SourceName, SourceFragment, NodeHash> map_;

 public:
  SourceMap() : map_() {}
  /*! \brief Add a source fragment with the file name and source. */
  SourceName AddSource(const std::string& file_name, const std::string& source);
  /*! \brief Retrieve a source fragment by source name. */
  const SourceFragment& GetSource(SourceName id) const;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_SOURCE_MAP_H_
