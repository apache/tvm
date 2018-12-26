/*!
 *  Copyright (c) 2018 by Contributors
 * \file error.cc
 * \brief Relay type inference and checking.
 *
 */

#include <tvm/relay/error.h>
#include "../util/rang.h"

namespace tvm {
namespace relay {

dmlc::Error ErrorReporter::Render() {
    for (auto err : this->errors) {
      auto sp = err.sp;
      CHECK(sp.defined()) << "while attempting to report an error its span was null";
      auto source_file = this->src_map.GetSource(err.sp->source);
      auto file_name = source_file.name->name;
      auto lines = source_file.LinesAt(err.sp, 1);
      std::string error_marker = "error:";
      auto line_info =
          std::to_string(sp->lineno) + ":" + std::to_string(sp->col_offset);

      std::cout << rang::style::bold << rang::fg::red << error_marker
                << rang::fg::reset << file_name << ":" << line_info
                << rang::style::reset << " " << lines[0] << std::endl;

      // Build the cursor.

      // Fix this code, hardwired to compute alignment of pointer.
      size_t spaces = error_marker.size() + line_info.size() + file_name.size() +
                      sp->col_offset - 3;

      std::string cursor = "~~~~^~~~~";
      for (size_t i = 0; i < spaces; i++) {
        std::cout << " ";
      }

      std::cout << rang::fg::red << cursor << " " << err.what() << rang::style::reset
                << std::endl;
    }
    return dmlc::Error("print me");
  }


} // relay
} // tvm
