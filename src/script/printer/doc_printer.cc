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
#include "doc_printer.h"

#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/str_escape.h"
#include "../../support/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace {

/*! \brief Range of byte offsets in a string */
using ByteSpan = std::pair<size_t, size_t>;

/*!
 * \brief DocPrinter is responsible for printing Doc tree into text format
 * \details This is the base class for translating Doc into string.
 *          Each target language needs to have its subclass of DocPrinter
 *          to define the actual logic of printing Doc.
 *
 * \sa Doc
 */
class DocPrinter {
 public:
  /*!
   * \brief The constructor of DocPrinter
   *
   * \param options the option for printer
   */
  explicit DocPrinter(const PrinterConfig& options);

  virtual ~DocPrinter() = default;

  /*! \brief Append an ordered entry-owned header with line/span accounting.
   * Elements are fixed String source chunks or ordinary CommentDocs, in the
   * exact order to emit. Comments use normal Doc spans and newline handling.
   * Any other element type raises TypeError. The printer retains no header.
   */
  void AppendHeader(const ffi::Array<ffi::Any>& header);

  /*!
   * \brief Append a doc to the final content
   *
   * \param doc  Doc to be printed
   * \param paths Source paths requested for underlining
   *
   * \sa GetString
   */
  void Append(const Doc& doc, const ffi::Array<AccessPath>& paths);

  /*!
   * \brief Get the printed string of all Doc appended
   *
   * The content of each Doc in the returned string will
   * appear in the same order as they are appended.
   *
   * \sa Append
   */
  ffi::String GetString() const;

  /*!
   * \brief Return the visible source path selected for each requested
   *        underline path.
   */
  ffi::Array<ffi::Optional<AccessPath>> GetVisiblePaths() const;

 protected:
  /*!
   * \brief Get the printed string
   *
   * It will dispatch to the PrintTypedDoc method based on
   * the actual type of Doc.
   *
   * \sa PrintTypedDoc
   */
  void PrintDoc(const Doc& doc);

  /*!
   * \brief Virtual method to print a LiteralDoc
   */
  virtual void PrintTypedDoc(const LiteralDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an ExprStringDoc
   */
  virtual void PrintTypedDoc(const ExprStringDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IdDoc
   */
  virtual void PrintTypedDoc(const IdDoc& doc) = 0;

  /*! \brief Print a canonical namespace using this invocation's alias. */
  virtual void PrintTypedDoc(const NamespaceDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AttrAccessDoc
   */
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IndexDoc
   */
  virtual void PrintTypedDoc(const IndexDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an OperationDoc
   */
  virtual void PrintTypedDoc(const OperationDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a CallDoc
   */
  virtual void PrintTypedDoc(const CallDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a LambdaDoc
   */
  virtual void PrintTypedDoc(const LambdaDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ListDoc
   */
  virtual void PrintTypedDoc(const ListDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a TupleDoc
   */
  virtual void PrintTypedDoc(const TupleDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a DictDoc
   */
  virtual void PrintTypedDoc(const DictDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a SliceDoc
   */
  virtual void PrintTypedDoc(const SliceDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a StmtBlockDoc
   */
  virtual void PrintTypedDoc(const StmtBlockDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AssignDoc
   */
  virtual void PrintTypedDoc(const AssignDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IfDoc
   */
  virtual void PrintTypedDoc(const IfDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a WhileDoc
   */
  virtual void PrintTypedDoc(const WhileDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a BreakDoc
   */
  virtual void PrintTypedDoc(const BreakDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ContinueDoc
   */
  virtual void PrintTypedDoc(const ContinueDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ForDoc
   */
  virtual void PrintTypedDoc(const ForDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ScopeDoc
   */
  virtual void PrintTypedDoc(const ScopeDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an ExprStmtDoc
   */
  virtual void PrintTypedDoc(const ExprStmtDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AssertDoc
   */
  virtual void PrintTypedDoc(const AssertDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ReturnDoc
   */
  virtual void PrintTypedDoc(const ReturnDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a FunctionDoc
   */
  virtual void PrintTypedDoc(const FunctionDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ClassDoc
   */
  virtual void PrintTypedDoc(const ClassDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a CommentDoc
   */
  virtual void PrintTypedDoc(const CommentDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a DocStringDoc
   */
  virtual void PrintTypedDoc(const DocStringDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a OpCallDoc
   */
  virtual void PrintTypedDoc(const OpCallDoc& doc) = 0;

  /*!
   * \brief Increase the indent level of any content to be
   *        printed after this call
   */
  void IncreaseIndent() { indent_ += options_->indent_spaces; }

  /*!
   * \brief Decrease the indent level of any content to be
   *        printed after this call
   */
  void DecreaseIndent() { indent_ -= options_->indent_spaces; }

  /*!
   * \brief Add a new line into the output stream
   *
   * \sa output_
   */
  std::ostream& NewLine() {
    size_t start_pos = output_.tellp();
    output_ << "\n";
    line_starts_.push_back(output_.tellp());
    output_ << std::string(indent_, ' ');
    size_t end_pos = output_.tellp();
    underlines_exempted_.push_back({start_pos, end_pos});
    return output_;
  }

  /*!
   * \brief The output stream of printer
   *
   * All printed content will be stored in this stream and returned
   * when GetString is called.
   *
   * \sa GetString
   */
  std::ostringstream output_;

  /*! \brief Configuration is read only while a Doc is rendered. */
  const PrinterConfig& config() const { return options_; }

  /*! \brief Spans that we have already committed to underline exemption. */
  std::vector<ByteSpan> underlines_exempted_;

 private:
  void MarkSpan(const ByteSpan& span, const AccessPath& path);

  /*! \brief Options to customize certain aspects of the output */
  PrinterConfig options_;

  /*! \brief the current level of indent */
  int indent_ = 0;

  /*! \brief For each line in the output_, byte offset of its first character */
  std::vector<size_t> line_starts_;

  /*! \brief Path of the object that we would like to underline */
  ffi::Array<AccessPath> path_to_underline_;

  /*!
   * \brief Candidate spans to be underlined, until we find a better match.
   * (A better match is an object with a longer path that is still a prefix of path_to_underline_.)
   */
  std::vector<std::vector<ByteSpan>> current_underline_candidates_;

  /*! \brief Path length of the objects that are current candidates for underlining. */
  std::vector<int> current_max_path_depth_;

  /*! \brief Visible path selected by the current best underline candidate. */
  std::vector<ffi::Optional<AccessPath>> current_visible_paths_;

  /*! \brief Spans that we have already committed to underline. */
  std::vector<ByteSpan> underlines_;
};

namespace {

std::vector<ByteSpan> MergeAndExemptSpans(const std::vector<ByteSpan>& spans,
                                          const std::vector<ByteSpan>& spans_exempted) {
  // use prefix sum to merge and exempt spans
  std::vector<ByteSpan> res;
  std::vector<std::pair<size_t, int>> prefix_stamp;
  for (ByteSpan span : spans) {
    prefix_stamp.push_back({span.first, 1});
    prefix_stamp.push_back({span.second, -1});
  }
  // at most spans.size() spans accumulated in prefix sum
  // use spans.size() + 1 as stamp unit to exempt all positive spans
  // with only one negative span
  int max_n = spans.size() + 1;
  for (ByteSpan span : spans_exempted) {
    prefix_stamp.push_back({span.first, -max_n});
    prefix_stamp.push_back({span.second, max_n});
  }
  std::sort(prefix_stamp.begin(), prefix_stamp.end());
  int prefix_sum = 0;
  int n = prefix_stamp.size();
  for (int i = 0; i < n - 1; ++i) {
    prefix_sum += prefix_stamp[i].second;
    // positive prefix sum leads to spans without exemption
    // different stamp positions guarantee the stamps in same position accumulated
    if (prefix_sum > 0 && prefix_stamp[i].first < prefix_stamp[i + 1].first) {
      if (res.size() && res.back().second == prefix_stamp[i].first) {
        // merge to the last spans if it is successive
        res.back().second = prefix_stamp[i + 1].first;
      } else {
        // add a new independent span
        res.push_back({prefix_stamp[i].first, prefix_stamp[i + 1].first});
      }
    }
  }
  return res;
}

size_t GetTextWidth(const std::string& text, const ByteSpan& span) {
  // FIXME: this only works for ASCII characters.
  // To do this "correctly", we need to parse UTF-8 into codepoints
  // and call wcwidth() or equivalent for every codepoint.
  size_t ret = 0;
  for (size_t i = span.first; i != span.second; ++i) {
    if (isprint(text[i])) {
      ret += 1;
    }
  }
  return ret;
}

size_t MoveBack(size_t pos, size_t distance) { return distance > pos ? 0 : pos - distance; }

size_t MoveForward(size_t pos, size_t distance, size_t max) {
  return distance > max - pos ? max : pos + distance;
}

size_t GetLineIndex(size_t byte_pos, const std::vector<size_t>& line_starts) {
  auto it = std::upper_bound(line_starts.begin(), line_starts.end(), byte_pos);
  return (it - line_starts.begin()) - 1;
}

using UnderlineIter = typename std::vector<ByteSpan>::const_iterator;

ByteSpan PopNextUnderline(UnderlineIter* next_underline, UnderlineIter end_underline) {
  if (*next_underline == end_underline) {
    return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
  } else {
    return *(*next_underline)++;
  }
}

void PrintChunk(const std::pair<size_t, size_t>& lines_range,
                const std::pair<UnderlineIter, UnderlineIter>& underlines, const std::string& text,
                const std::vector<size_t>& line_starts, const PrinterConfig& options,
                size_t line_number_width, std::string* out) {
  UnderlineIter next_underline = underlines.first;
  ByteSpan current_underline = PopNextUnderline(&next_underline, underlines.second);

  for (size_t line_idx = lines_range.first; line_idx < lines_range.second; ++line_idx) {
    if (options->print_line_numbers) {
      std::string line_num_str = std::to_string(line_idx + 1);
      line_num_str.push_back(' ');
      for (size_t i = line_num_str.size(); i < line_number_width; ++i) {
        out->push_back(' ');
      }
      *out += line_num_str;
    }

    size_t line_start = line_starts.at(line_idx);
    size_t line_end =
        line_idx + 1 == line_starts.size() ? text.size() : line_starts.at(line_idx + 1);
    out->append(text.begin() + line_start, text.begin() + line_end);

    bool printed_underline = false;
    size_t line_pos = line_start;
    bool printed_extra_caret = 0;
    while (current_underline.first < line_end) {
      if (!printed_underline) {
        *out += std::string(line_number_width, ' ');
        printed_underline = true;
      }

      size_t underline_end_for_line = std::min(line_end, current_underline.second);
      size_t num_spaces = GetTextWidth(text, {line_pos, current_underline.first});
      if (num_spaces > 0 && printed_extra_caret) {
        num_spaces -= 1;
        printed_extra_caret = false;
      }
      *out += std::string(num_spaces, ' ');

      size_t num_carets = GetTextWidth(text, {current_underline.first, underline_end_for_line});
      if (num_carets == 0 && !printed_extra_caret) {
        // Special case: when underlineing an empty or unprintable string, make sure to print
        // at least one caret still.
        num_carets = 1;
        printed_extra_caret = true;
      } else if (num_carets > 0 && printed_extra_caret) {
        num_carets -= 1;
        printed_extra_caret = false;
      }
      *out += std::string(num_carets, '^');

      line_pos = current_underline.first = underline_end_for_line;
      if (current_underline.first == current_underline.second) {
        current_underline = PopNextUnderline(&next_underline, underlines.second);
      }
    }

    if (printed_underline) {
      out->push_back('\n');
    }
  }
}

void PrintCut(size_t num_lines_skipped, std::string* out) {
  if (num_lines_skipped != 0) {
    std::ostringstream s;
    s << "(... " << num_lines_skipped << " lines skipped ...)\n";
    *out += s.str();
  }
}

std::pair<size_t, size_t> GetLinesForUnderline(const ByteSpan& underline,
                                               const std::vector<size_t>& line_starts,
                                               size_t num_lines, const PrinterConfig& options) {
  int context_lines = options->num_context_lines < 0 ? std::numeric_limits<int32_t>::max()
                                                     : options->num_context_lines;
  size_t first_line_of_underline = GetLineIndex(underline.first, line_starts);
  size_t first_line_of_chunk = MoveBack(first_line_of_underline, context_lines);
  size_t end_line_of_underline = GetLineIndex(underline.second - 1, line_starts) + 1;
  size_t end_line_of_chunk = MoveForward(end_line_of_underline, context_lines, num_lines);

  return {first_line_of_chunk, end_line_of_chunk};
}

// If there is only one line between the chunks, it is better to print it as is,
// rather than something like "(... 1 line skipped ...)".
constexpr const size_t kMinLinesToCutOut = 2;

bool TryMergeChunks(std::pair<size_t, size_t>* cur_chunk,
                    const std::pair<size_t, size_t>& new_chunk) {
  if (new_chunk.first < cur_chunk->second + kMinLinesToCutOut) {
    cur_chunk->second = new_chunk.second;
    return true;
  } else {
    return false;
  }
}

size_t GetNumLines(const std::string& text, const std::vector<size_t>& line_starts) {
  if (line_starts.back() == text.size()) {
    // Final empty line doesn't count as a line
    return line_starts.size() - 1;
  } else {
    return line_starts.size();
  }
}

size_t GetLineNumberWidth(size_t num_lines, const PrinterConfig& options) {
  if (options->print_line_numbers) {
    return std::to_string(num_lines).size() + 1;
  } else {
    return 0;
  }
}

std::string DecorateText(const std::string& text, const std::vector<size_t>& line_starts,
                         const PrinterConfig& options, const std::vector<ByteSpan>& underlines) {
  size_t num_lines = GetNumLines(text, line_starts);
  size_t line_number_width = GetLineNumberWidth(num_lines, options);

  std::string ret;
  if (underlines.empty()) {
    PrintChunk({0, num_lines}, {underlines.begin(), underlines.begin()}, text, line_starts, options,
               line_number_width, &ret);
    return ret;
  }

  size_t last_end_line = 0;
  std::pair<size_t, size_t> cur_chunk =
      GetLinesForUnderline(underlines[0], line_starts, num_lines, options);
  if (cur_chunk.first < kMinLinesToCutOut) {
    cur_chunk.first = 0;
  }

  auto first_underline_in_cur_chunk = underlines.begin();
  for (auto underline_it = underlines.begin() + 1; underline_it != underlines.end();
       ++underline_it) {
    std::pair<size_t, size_t> new_chunk =
        GetLinesForUnderline(*underline_it, line_starts, num_lines, options);

    if (!TryMergeChunks(&cur_chunk, new_chunk)) {
      PrintCut(cur_chunk.first - last_end_line, &ret);
      PrintChunk(cur_chunk, {first_underline_in_cur_chunk, underline_it}, text, line_starts,
                 options, line_number_width, &ret);
      last_end_line = cur_chunk.second;
      cur_chunk = new_chunk;
      first_underline_in_cur_chunk = underline_it;
    }
  }

  PrintCut(cur_chunk.first - last_end_line, &ret);
  if (num_lines - cur_chunk.second < kMinLinesToCutOut) {
    cur_chunk.second = num_lines;
  }
  PrintChunk(cur_chunk, {first_underline_in_cur_chunk, underlines.end()}, text, line_starts,
             options, line_number_width, &ret);
  PrintCut(num_lines - cur_chunk.second, &ret);
  return ret;
}

}  // namespace

DocPrinter::DocPrinter(const PrinterConfig& options) : options_(options) {
  line_starts_.push_back(0);
}

void DocPrinter::AppendHeader(const ffi::Array<ffi::Any>& header) {
  for (const ffi::Any& item : header) {
    if (auto comment = item.as<CommentDoc>()) {
      PrintDoc(comment.value());
      NewLine();
    } else {
      // Executable imports and separators are fixed entry-owned text. Comment
      // headers use ordinary Docs and their normal span/exemption machinery.
      ffi::String text = item.as_or_throw<ffi::String>();
      for (size_t i = 0; i < text.size(); ++i) {
        if (text.data()[i] == '\n')
          NewLine();
        else
          output_ << text.data()[i];
      }
    }
  }
}

void DocPrinter::Append(const Doc& doc, const ffi::Array<AccessPath>& paths) {
  for (const AccessPath& p : paths) {
    path_to_underline_.push_back(p);
    current_max_path_depth_.push_back(0);
    current_visible_paths_.push_back(std::nullopt);
    current_underline_candidates_.push_back(std::vector<ByteSpan>());
  }
  PrintDoc(doc);
  for (const auto& c : current_underline_candidates_) {
    underlines_.insert(underlines_.end(), c.begin(), c.end());
  }
}

ffi::String DocPrinter::GetString() const {
  std::string text = output_.str();

  // Remove any trailing indentation
  while (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }

  if (!text.empty() && text.back() != '\n') {
    text.push_back('\n');
  }

  return DecorateText(text, line_starts_, options_,
                      MergeAndExemptSpans(underlines_, underlines_exempted_));
}

ffi::Array<ffi::Optional<AccessPath>> DocPrinter::GetVisiblePaths() const {
  return ffi::Array<ffi::Optional<AccessPath>>(current_visible_paths_);
}

void DocPrinter::PrintDoc(const Doc& doc) {
  size_t start_pos = output_.tellp();

  static const auto dispatch = []() {
    ObjectFunctor<void(const ffi::ObjectRef&, DocPrinter*)> table;
#define TVM_SCRIPT_PRINT_DISPATCH(DocType)                                                   \
  table.SetDispatch<DocType##Node>([](const ffi::ObjectRef& obj, DocPrinter* self) {         \
    self->PrintTypedDoc(ffi::GetRef<DocType>(static_cast<const DocType##Node*>(obj.get()))); \
  });
    TVM_SCRIPT_PRINT_DISPATCH(LiteralDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ExprStringDoc);
    TVM_SCRIPT_PRINT_DISPATCH(IdDoc);
    TVM_SCRIPT_PRINT_DISPATCH(NamespaceDoc);
    TVM_SCRIPT_PRINT_DISPATCH(AttrAccessDoc);
    TVM_SCRIPT_PRINT_DISPATCH(IndexDoc);
    TVM_SCRIPT_PRINT_DISPATCH(OperationDoc);
    TVM_SCRIPT_PRINT_DISPATCH(CallDoc);
    TVM_SCRIPT_PRINT_DISPATCH(LambdaDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ListDoc);
    TVM_SCRIPT_PRINT_DISPATCH(TupleDoc);
    TVM_SCRIPT_PRINT_DISPATCH(DictDoc);
    TVM_SCRIPT_PRINT_DISPATCH(SliceDoc);
    TVM_SCRIPT_PRINT_DISPATCH(StmtBlockDoc);
    TVM_SCRIPT_PRINT_DISPATCH(AssignDoc);
    TVM_SCRIPT_PRINT_DISPATCH(IfDoc);
    TVM_SCRIPT_PRINT_DISPATCH(WhileDoc);
    TVM_SCRIPT_PRINT_DISPATCH(BreakDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ContinueDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ForDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ScopeDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ExprStmtDoc);
    TVM_SCRIPT_PRINT_DISPATCH(AssertDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ReturnDoc);
    TVM_SCRIPT_PRINT_DISPATCH(FunctionDoc);
    TVM_SCRIPT_PRINT_DISPATCH(ClassDoc);
    TVM_SCRIPT_PRINT_DISPATCH(CommentDoc);
    TVM_SCRIPT_PRINT_DISPATCH(DocStringDoc);
    TVM_SCRIPT_PRINT_DISPATCH(OpCallDoc);
#undef TVM_SCRIPT_PRINT_DISPATCH
    table.Finalize();
    return table;
  }();
  dispatch(doc, this);

  size_t end_pos = output_.tellp();
  for (const AccessPath& path : doc->source_paths) {
    MarkSpan({start_pos, end_pos}, path);
  }
}

void DocPrinter::MarkSpan(const ByteSpan& span, const AccessPath& path) {
  int n = path_to_underline_.size();
  for (int i = 0; i < n; ++i) {
    AccessPath p = path_to_underline_[i];
    if (path->depth >= current_max_path_depth_[i] && path->IsPrefixOf(p)) {
      if (path->depth > current_max_path_depth_[i] || !current_visible_paths_[i].has_value()) {
        current_max_path_depth_[i] = path->depth;
        current_visible_paths_[i] = path;
        current_underline_candidates_[i].clear();
      }
      current_underline_candidates_[i].push_back(span);
    }
  }
}

namespace {

/*!
 * \brief Escape an ExprStringDoc child while preserving final output positions.
 *
 * ExprStringDoc must print its child through the active PythonDocPrinter so that the child's
 * precedence, printer configuration, and source-path bookkeeping stay in the current traversal.
 * However, escaping after printing into a temporary buffer would make PrintDoc record positions
 * in that unescaped buffer, rather than the final Python source. This scoped adapter instead
 * replaces the output stream's buffer only while the child is printed. It escapes each write and
 * forwards it immediately to the original final buffer; the caller emits the surrounding quotes
 * outside the scope so that only the child is transformed.
 *
 * The adapter owns neither the output stream nor its original buffer. Both outlive this stack
 * object. The constructor saves the original buffer in destination_ and installs this adapter;
 * the destructor restores the original buffer before the caller continues writing to the stream.
 *
 * PrintDoc uses tellp() before and after every child to record source spans. tellp() reaches
 * seekoff(0, cur, out), which is the only seek this adapter accepts and delegates directly to the
 * original buffer. Because that buffer already contains the preceding output and advances by the
 * escaped byte count, child spans use absolute positions in the final source, including any
 * expansion introduced by StrEscape.
 *
 * A raw newline violates the one-line expression-string contract. xsputn records its presence
 * while still escaping and forwarding the write, and the caller checks saw_newline() before
 * completing the literal. On every exit path, including exception unwinding, the noexcept
 * destructor restores the original buffer.
 */
class ScopedExprStringEscapeBuf : public std::streambuf {
 public:
  explicit ScopedExprStringEscapeBuf(std::ostream* output)
      : output_(output), destination_(output->rdbuf()) {
    output_->rdbuf(this);
  }

  ~ScopedExprStringEscapeBuf() noexcept { output_->rdbuf(destination_); }

  ScopedExprStringEscapeBuf(const ScopedExprStringEscapeBuf&) = delete;
  ScopedExprStringEscapeBuf& operator=(const ScopedExprStringEscapeBuf&) = delete;

  bool saw_newline() const { return saw_newline_; }

 protected:
  std::streamsize xsputn(const char* data, std::streamsize count) final {
    if (count <= 0) return 0;
    saw_newline_ = saw_newline_ || std::find(data, data + count, '\n') != data + count;
    // StrEscape is byte-wise, so each write can be transformed independently and forwarded
    // without retaining or copying the complete output. Report consumed input bytes, not the
    // potentially larger number of escaped bytes written to the destination.
    std::string escaped = support::StrEscape(data, static_cast<size_t>(count));
    destination_->sputn(escaped.data(), escaped.size());
    return count;
  }

  int_type overflow(int_type ch) final {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    char value = traits_type::to_char_type(ch);
    return xsputn(&value, 1) == 1 ? ch : traits_type::eof();
  }

  int sync() final { return destination_->pubsync(); }

  pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                   std::ios_base::openmode mode) final {
    // Support the current-position query used by tellp(), preserving the destination's absolute
    // escaped offset. ExprStringDoc rendering never needs to reposition the output sequence.
    if (offset == 0 && direction == std::ios_base::cur && (mode & std::ios_base::out)) {
      return destination_->pubseekoff(offset, direction, mode);
    }
    return pos_type(off_type(-1));
  }

 private:
  std::ostream* output_;
  std::streambuf* destination_;
  bool saw_newline_{false};
};

ffi::String RenderInvisiblePathInfo(const ffi::String& script,
                                    const ffi::Array<AccessPath>& requested_paths,
                                    const ffi::Array<ffi::Optional<AccessPath>>& visible_paths) {
  if (requested_paths.empty()) return script;

  std::ostringstream os;
  for (size_t i = 0; i < requested_paths.size(); ++i) {
    if (i != 0) os << "\n";
    const AccessPath& requested_path = requested_paths[i];
    os << "Access path: " << requested_path;

    ffi::Optional<AccessPath> visible_path = std::nullopt;
    if (i < visible_paths.size()) visible_path = visible_paths[i];
    if (!visible_path.has_value()) {
      os << "\nNote: No visible object for this path is rendered in TVMScript.";
    } else if (!visible_path.value()->PathEqual(requested_path) &&
               visible_path.value()->IsPrefixOf(requested_path)) {
      os << "\nNote: The underlined object is the nearest visible parent of this path.";
    }
  }
  os << "\n\n" << script;
  return ffi::String(os.str());
}

}  // namespace

/*!
 * \brief Operator precedence
 *
 * This is based on
 * https://docs.python.org/3/reference/expressions.html#operator-precedence
 */
enum class ExprPrecedence : int32_t {
  /*! \brief Unknown precedence */
  kUnkown = 0,
  /*! \brief Lambda Expression */
  kLambda = 1,
  /*! \brief Conditional Expression */
  kIfThenElse = 2,
  /*! \brief Boolean OR */
  kBooleanOr = 3,
  /*! \brief Boolean AND */
  kBooleanAnd = 4,
  /*! \brief Boolean NOT */
  kBooleanNot = 5,
  /*! \brief Comparisons */
  kComparison = 6,
  /*! \brief Bitwise OR */
  kBitwiseOr = 7,
  /*! \brief Bitwise XOR */
  kBitwiseXor = 8,
  /*! \brief Bitwise AND */
  kBitwiseAnd = 9,
  /*! \brief Shift Operators */
  kShift = 10,
  /*! \brief Addition and subtraction */
  kAdd = 11,
  /*! \brief Multiplication, division, floor division, remainder */
  kMult = 12,
  /*! \brief Positive negative and bitwise NOT */
  kUnary = 13,
  /*! \brief Exponentiation */
  kExp = 14,
  /*! \brief Index access, attribute access, call and atom expression */
  kIdentity = 15,
};

ExprPrecedence GetExprPrecedence(const ExprDoc& doc) {
  // Key is the value of OperationDocNode::Kind
  static const std::vector<ExprPrecedence> op_kind_precedence = []() {
    using OpKind = OperationDocNode::Kind;
    std::map<OpKind, ExprPrecedence> raw_table = {
        {OpKind::kUSub, ExprPrecedence::kUnary},
        {OpKind::kInvert, ExprPrecedence::kUnary},
        {OpKind::kNot, ExprPrecedence::kBooleanNot},
        {OpKind::kAdd, ExprPrecedence::kAdd},
        {OpKind::kSub, ExprPrecedence::kAdd},
        {OpKind::kMult, ExprPrecedence::kMult},
        {OpKind::kDiv, ExprPrecedence::kMult},
        {OpKind::kFloorDiv, ExprPrecedence::kMult},
        {OpKind::kMod, ExprPrecedence::kMult},
        {OpKind::kPow, ExprPrecedence::kExp},
        {OpKind::kLShift, ExprPrecedence::kShift},
        {OpKind::kRShift, ExprPrecedence::kShift},
        {OpKind::kBitAnd, ExprPrecedence::kBitwiseAnd},
        {OpKind::kBitOr, ExprPrecedence::kBitwiseOr},
        {OpKind::kBitXor, ExprPrecedence::kBitwiseXor},
        {OpKind::kLt, ExprPrecedence::kComparison},
        {OpKind::kLtE, ExprPrecedence::kComparison},
        {OpKind::kEq, ExprPrecedence::kComparison},
        {OpKind::kNotEq, ExprPrecedence::kComparison},
        {OpKind::kGt, ExprPrecedence::kComparison},
        {OpKind::kGtE, ExprPrecedence::kComparison},
        {OpKind::kAnd, ExprPrecedence::kBooleanAnd},
        {OpKind::kOr, ExprPrecedence::kBooleanOr},
        {OpKind::kMatMul, ExprPrecedence::kMult},
        {OpKind::kIfThenElse, ExprPrecedence::kIfThenElse},
    };
    int n = static_cast<int>(OpKind::kSpecialEnd);
    std::vector<ExprPrecedence> table(n + 1, ExprPrecedence::kUnkown);
    for (const auto& kv : raw_table) {
      table[static_cast<int>(kv.first)] = kv.second;
    }
    return table;
  }();

  // Key is the type index of Doc
  static const std::unordered_map<uint32_t, ExprPrecedence> doc_type_precedence = {
      {LiteralDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {ExprStringDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {IdDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {NamespaceDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {AttrAccessDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {IndexDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {CallDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {LambdaDocNode::RuntimeTypeIndex(), ExprPrecedence::kLambda},
      {TupleDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {ListDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
      {DictDocNode::RuntimeTypeIndex(), ExprPrecedence::kIdentity},
  };

  if (const auto* op_doc = doc.as<OperationDocNode>()) {
    size_t kind = static_cast<int>(op_doc->kind);
    TVM_FFI_CHECK_LT(kind, op_kind_precedence.size(), ValueError) << "Invalid operation: " << kind;
    ExprPrecedence precedence = op_kind_precedence[kind];
    TVM_FFI_ICHECK(precedence != ExprPrecedence::kUnkown)
        << "Precedence for operator " << static_cast<int>(op_doc->kind) << " is unknown";
    return precedence;
  }
  auto it = doc_type_precedence.find(doc->type_index());
  if (it != doc_type_precedence.end()) {
    return it->second;
  }
  TVM_FFI_ICHECK(false) << "Precedence for doc type " << doc->GetTypeKey() << " is unknown";
  throw;
}

class PythonDocPrinter : public DocPrinter {
 public:
  explicit PythonDocPrinter(const PrinterConfig& options) : DocPrinter(options) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintTypedDoc(const LiteralDoc& doc) final;
  void PrintTypedDoc(const ExprStringDoc& doc) final;
  void PrintTypedDoc(const IdDoc& doc) final;
  void PrintTypedDoc(const NamespaceDoc& doc) final;
  void PrintTypedDoc(const AttrAccessDoc& doc) final;
  void PrintTypedDoc(const IndexDoc& doc) final;
  void PrintTypedDoc(const OperationDoc& doc) final;
  void PrintTypedDoc(const CallDoc& doc) final;
  void PrintTypedDoc(const LambdaDoc& doc) final;
  void PrintTypedDoc(const ListDoc& doc) final;
  void PrintTypedDoc(const DictDoc& doc) final;
  void PrintTypedDoc(const TupleDoc& doc) final;
  void PrintTypedDoc(const SliceDoc& doc) final;
  void PrintTypedDoc(const StmtBlockDoc& doc) final;
  void PrintTypedDoc(const AssignDoc& doc) final;
  void PrintTypedDoc(const IfDoc& doc) final;
  void PrintTypedDoc(const WhileDoc& doc) final;
  void PrintTypedDoc(const BreakDoc& doc) final;
  void PrintTypedDoc(const ContinueDoc& doc) final;
  void PrintTypedDoc(const ForDoc& doc) final;
  void PrintTypedDoc(const ExprStmtDoc& doc) final;
  void PrintTypedDoc(const AssertDoc& doc) final;
  void PrintTypedDoc(const ReturnDoc& doc) final;
  void PrintTypedDoc(const ScopeDoc& doc) final;
  void PrintTypedDoc(const FunctionDoc& doc) final;
  void PrintTypedDoc(const ClassDoc& doc) final;
  void PrintTypedDoc(const CommentDoc& doc) final;
  void PrintTypedDoc(const DocStringDoc& doc) final;
  void PrintTypedDoc(const OpCallDoc& doc) final;

  void PrintAssignment(const ExprDoc& lhs, const ffi::Optional<ExprDoc>& rhs,
                       const ffi::Optional<ExprDoc>& annotation);

 private:
  /*! \brief Borrowed tuple whose parentheses are omitted during the active print call. */
  const TupleDocNode* unparenthesized_tuple_{nullptr};
  /*! \brief Whether that tuple's singleton comma is supplied by the surrounding syntax. */
  bool omit_tuple_singleton_comma_{false};

  void NewLineWithoutIndent() {
    size_t start_pos = output_.tellp();
    output_ << "\n";
    size_t end_pos = output_.tellp();
    underlines_exempted_.push_back({start_pos, end_pos});
  }

  template <typename DocType>
  void PrintJoinedDocs(const ffi::Array<DocType>& docs, const std::string& separator) {
    bool is_first = true;
    for (auto& doc : docs) {
      if (is_first) {
        is_first = false;
      } else {
        output_ << separator;
      }
      PrintDoc(doc);
    }
  }

  void PrintIndentedBlock(const ffi::Array<StmtDoc>& docs) {
    IncreaseIndent();
    bool outer_is_last_stmt = is_last_stmt_;
    for (size_t i = 0; i < docs.size(); ++i) {
      NewLine();
      is_last_stmt_ = i + 1 == docs.size();
      PrintDoc(docs[i]);
    }
    is_last_stmt_ = outer_is_last_stmt;
    if (docs.empty()) {
      NewLine();
      output_ << "pass";
    }
    DecreaseIndent();
  }

  void PrintDecorators(const ffi::Array<ExprDoc>& decorators) {
    for (const ExprDoc& decorator : decorators) {
      output_ << "@";
      PrintDoc(decorator);
      NewLine();
    }
  }

  /*!
   * \brief Print expression and add parenthesis if needed.
   */
  void PrintChildExpr(const ExprDoc& doc, ExprPrecedence parent_precedence,
                      bool parenthesis_for_same_precedence = false) {
    ExprPrecedence doc_precedence = GetExprPrecedence(doc);
    if (doc_precedence < parent_precedence ||
        (parenthesis_for_same_precedence && doc_precedence == parent_precedence)) {
      output_ << "(";
      PrintDoc(doc);
      output_ << ")";
    } else {
      PrintDoc(doc);
    }
  }

  /*!
   * \brief Print expression and add parenthesis if doc has lower precedence than parent.
   */
  void PrintChildExpr(const ExprDoc& doc, const ExprDoc& parent,
                      bool parenthesis_for_same_precedence = false) {
    ExprPrecedence parent_precedence = GetExprPrecedence(parent);
    return PrintChildExpr(doc, parent_precedence, parenthesis_for_same_precedence);
  }

  /*!
   * \brief Print expression and add parenthesis if doc doesn't have higher precedence than parent.
   *
   * This function should be used to print an child expression that needs to be wrapped
   * by parenthesis even if it has the same precedence as its parent, e.g., the `b` in `a + b`
   * and the `b` and `c` in `a if b else c`.
   */
  void PrintChildExprConservatively(const ExprDoc& doc, const ExprDoc& parent) {
    PrintChildExpr(doc, parent, /*parenthesis_for_same_precedence=*/true);
  }

  void MaybePrintCommentInline(const StmtDoc& stmt) {
    if (stmt->comment.has_value()) {
      const std::string& comment = stmt->comment.value();
      bool has_newline = std::find(comment.begin(), comment.end(), '\n') != comment.end();
      TVM_FFI_CHECK(!has_newline, ValueError)
          << "the comment string of " << stmt->GetTypeKey() << " cannot have newline.";
      size_t start_pos = output_.tellp();
      output_ << "  # " << comment;
      size_t end_pos = output_.tellp();
      underlines_exempted_.push_back({start_pos, end_pos});
    }
  }

  void MaybePrintCommenMultiLines(const StmtDoc& stmt, bool new_line = false) {
    if (stmt->comment.has_value()) {
      std::vector<std::string> comment_lines = support::Split(stmt->comment.value(), '\n');
      bool first_line = true;
      size_t start_pos = output_.tellp();
      for (const std::string& line : comment_lines) {
        if (first_line) {
          output_ << "# " << line;
          first_line = false;
        } else {
          NewLine() << "# " << line;
        }
      }
      size_t end_pos = output_.tellp();
      underlines_exempted_.push_back({start_pos, end_pos});
      if (new_line) {
        NewLine();
      }
    }
  }

  void PrintDocString(const ffi::String& comment) {
    size_t start_pos = output_.tellp();
    output_ << "\"\"\"";

    std::vector<std::string> comment_lines = support::Split(comment, '\n');
    for (const std::string& line : comment_lines) {
      if (line.empty()) {
        // No indentation on empty line
        output_ << "\n";
      } else {
        NewLine() << line;
      }
    }

    NewLine() << "\"\"\"";
    size_t end_pos = output_.tellp();
    underlines_exempted_.push_back({start_pos, end_pos});
  }

  void PrintBlockComment(const ffi::String& comment) {
    IncreaseIndent();
    NewLine();
    PrintDocString(comment);
    DecreaseIndent();
  }

  // A ScopeDoc may use concise spelling only with enclosing-list context.
  bool is_last_stmt_{false};
};

void PythonDocPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ffi::Any& value = doc->value;
  if (value == nullptr) {
    output_ << "None";
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    PrimType int_ty = int_imm->ty.as_or_throw<PrimType>();
    if (int_ty.MatchesCode(DLDataTypeCode::kDLBool)) {
      output_ << (int_imm->value ? "True" : "False");
    } else {
      output_ << int_imm->value;
    }
  } else if (const auto* float_imm = value.as<FloatImmNode>()) {
    // TODO(yelite): Make float number printing roundtrippable
    if (std::isinf(float_imm->value) || std::isnan(float_imm->value)) {
      output_ << '"' << float_imm->value << '"';
    } else if (std::nearbyint(float_imm->value) == float_imm->value) {
      // Special case for floating-point values which would be
      // formatted using %g, are not displayed in scientific
      // notation, and whose fractional part is zero.
      //
      // By default, using `operator<<(std::ostream&, double)`
      // delegates to the %g printf formatter.  This strips off any
      // trailing zeros, and also strips the decimal point if no
      // trailing zeros are found.  When parsed in python, due to the
      // missing decimal point, this would incorrectly convert a float
      // to an integer.  Providing the `std::showpoint` modifier
      // instead delegates to the %#g printf formatter.  On its own,
      // this resolves the round-trip errors, but also prevents the
      // trailing zeros from being stripped off.
      std::showpoint(output_);
      std::fixed(output_);
      output_.precision(1);
      output_ << float_imm->value;
    } else {
      std::defaultfloat(output_);
      std::noshowpoint(output_);
      output_.precision(17);
      output_ << float_imm->value;
    }

  } else if (const auto opt_str = value.as<ffi::String>()) {
    output_ << "\"" << support::StrEscape((*opt_str).data(), (*opt_str).size()) << "\"";
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported literal value type: " << value.GetTypeKey();
  }
}

void PythonDocPrinter::PrintTypedDoc(const ExprStringDoc& doc) {
  this->output_ << '"';
  {
    ScopedExprStringEscapeBuf escaping_scope(&this->output_);
    this->PrintDoc(doc->value);
    TVM_FFI_ICHECK(!escaping_scope.saw_newline())
        << "An expression rendered inside a Python string literal must be one line";
  }
  this->output_ << '"';
}

void PythonDocPrinter::PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

void PythonDocPrinter::PrintTypedDoc(const NamespaceDoc& doc) {
  const ffi::String& name = doc->canonical_name;
  ffi::String key = std::string(name) + ".prefix";
  ffi::String fallback = GetNamespaceAliases().Get(key).value_or(name);
  output_ << config()->GetExtraConfig<ffi::String>(key,
                                                   name == "ir" ? config()->ir_prefix : fallback);
}

void PythonDocPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintChildExpr(doc->value, doc);
  output_ << "." << doc->name;
}

void PythonDocPrinter::PrintTypedDoc(const IndexDoc& doc) {
  PrintChildExpr(doc->value, doc);
  if (doc->indices.size() == 0) {
    output_ << "[()]";
  } else {
    output_ << "[";
    PrintJoinedDocs(doc->indices, ", ");
    output_ << "]";
  }
}

const std::string OperatorToString(OperationDocNode::Kind operation_kind) {
  static const std::vector<std::string> op_kind2str = []() {
    using OpKind = OperationDocNode::Kind;
    std::map<OpKind, std::string> raw_table = {
        {OpKind::kUSub, "-"},       //
        {OpKind::kInvert, "~"},     //
        {OpKind::kNot, "not "},     //
        {OpKind::kAdd, "+"},        //
        {OpKind::kSub, "-"},        //
        {OpKind::kMult, "*"},       //
        {OpKind::kDiv, "/"},        //
        {OpKind::kFloorDiv, "//"},  //
        {OpKind::kMod, "%"},        //
        {OpKind::kPow, "**"},       //
        {OpKind::kLShift, "<<"},    //
        {OpKind::kRShift, ">>"},    //
        {OpKind::kBitAnd, "&"},     //
        {OpKind::kBitOr, "|"},      //
        {OpKind::kBitXor, "^"},     //
        {OpKind::kLt, "<"},         //
        {OpKind::kLtE, "<="},       //
        {OpKind::kEq, "=="},        //
        {OpKind::kNotEq, "!="},     //
        {OpKind::kGt, ">"},         //
        {OpKind::kGtE, ">="},       //
        {OpKind::kAnd, "and"},      //
        {OpKind::kOr, "or"},        //
        {OpKind::kMatMul, "@"},     //
    };

    std::vector<std::string> table;
    table.resize(static_cast<int>(OperationDocNode::Kind::kSpecialEnd) + 1);

    for (const auto& kv : raw_table) {
      table[static_cast<int>(kv.first)] = kv.second;
    }

    return table;
  }();

  auto op_index = static_cast<int>(operation_kind);
  TVM_FFI_ICHECK_LT(op_index, op_kind2str.size());
  const std::string str = op_kind2str[op_index];
  TVM_FFI_ICHECK(!str.empty()) << "OperationDocNode::Kind " << static_cast<int>(operation_kind)
                               << " cannot be converted to operator token in Python directly.";
  return str;
}

void PythonDocPrinter::PrintTypedDoc(const OperationDoc& doc) {
  using OpKind = OperationDocNode::Kind;
  if (doc->kind < OpKind::kUnaryEnd) {
    // Unary Operators
    TVM_FFI_ICHECK_EQ(doc->operands.size(), 1);
    output_ << OperatorToString(doc->kind);
    PrintChildExpr(doc->operands[0], doc);
  } else if (doc->kind == OpKind::kPow) {
    // Power operator is different than other binary operators
    // It's right-associative and binds less tightly than unary operator on its right.
    // https://docs.python.org/3/reference/expressions.html#the-power-operator
    // https://docs.python.org/3/reference/expressions.html#operator-precedence
    TVM_FFI_ICHECK_EQ(doc->operands.size(), 2);
    PrintChildExprConservatively(doc->operands[0], doc);
    output_ << " ** ";
    PrintChildExpr(doc->operands[1], ExprPrecedence::kUnary);
  } else if (doc->kind < OpKind::kBinaryEnd) {
    // Binary Operator
    TVM_FFI_ICHECK_EQ(doc->operands.size(), 2);
    PrintChildExpr(doc->operands[0], doc);
    output_ << " " << OperatorToString(doc->kind) << " ";
    PrintChildExprConservatively(doc->operands[1], doc);
  } else if (doc->kind == OpKind::kIfThenElse) {
    TVM_FFI_CHECK_EQ(doc->operands.size(), 3, ValueError)
        << "IfThenElse requires 3 operands, but got " << doc->operands.size();
    PrintChildExpr(doc->operands[1], doc);
    output_ << " if ";
    PrintChildExprConservatively(doc->operands[0], doc);
    output_ << " else ";
    PrintChildExprConservatively(doc->operands[2], doc);
  } else {
    TVM_FFI_THROW(InternalError) << "Unknown OperationDocNode::Kind "
                                 << static_cast<int>(doc->kind);
    throw;
  }
}

void PythonDocPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintChildExpr(doc->callee, doc);

  output_ << "(";

  // Print positional args
  bool is_first = true;
  for (const ExprDoc& arg : doc->args) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    PrintDoc(arg);
  }

  // Print keyword args
  TVM_FFI_ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
      << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
  for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    const ffi::String& keyword = doc->kwargs_keys[i];
    output_ << keyword;
    output_ << "=";
    PrintDoc(doc->kwargs_values[i]);
  }

  output_ << ")";
}

void PythonDocPrinter::PrintTypedDoc(const LambdaDoc& doc) {
  output_ << "lambda ";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ": ";
  PrintChildExpr(doc->body, doc);
}

void PythonDocPrinter::PrintTypedDoc(const ListDoc& doc) {
  output_ << "[";
  PrintJoinedDocs(doc->elements, ", ");
  output_ << "]";
}

void PythonDocPrinter::PrintTypedDoc(const TupleDoc& doc) {
  bool parentheses = doc.get() != unparenthesized_tuple_ || doc->elements.empty();
  if (parentheses) output_ << "(";
  if (doc->elements.size() == 1) {
    PrintDoc(doc->elements[0]);
    if (parentheses || !omit_tuple_singleton_comma_) output_ << ",";
  } else {
    PrintJoinedDocs(doc->elements, ", ");
  }
  if (parentheses) output_ << ")";
}

void PythonDocPrinter::PrintTypedDoc(const DictDoc& doc) {
  TVM_FFI_ICHECK_EQ(doc->keys.size(), doc->values.size())
      << "DictDoc should have equal number of elements in keys and values.";
  output_ << "{";
  size_t idx = 0;
  for (const ExprDoc& key : doc->keys) {
    if (idx > 0) {
      output_ << ", ";
    }
    PrintDoc(key);
    output_ << ": ";
    PrintDoc(doc->values[idx]);
    idx++;
  }
  output_ << "}";
}

void PythonDocPrinter::PrintTypedDoc(const SliceDoc& doc) {
  if (doc->start != nullptr) {
    PrintDoc(doc->start.value());
  }
  output_ << ":";
  if (doc->stop != nullptr) {
    PrintDoc(doc->stop.value());
  }
  if (doc->step != nullptr) {
    output_ << ":";
    PrintDoc(doc->step.value());
  }
}

void PythonDocPrinter::PrintTypedDoc(const StmtBlockDoc& doc) {
  bool outer_is_last_stmt = is_last_stmt_;
  for (size_t i = 0; i < doc->stmts.size(); ++i) {
    is_last_stmt_ = i + 1 == doc->stmts.size() && outer_is_last_stmt;
    PrintDoc(doc->stmts[i]);
    if (i + 1 != doc->stmts.size()) {
      NewLine();
    }
  }
  is_last_stmt_ = outer_is_last_stmt;
}

void PythonDocPrinter::PrintAssignment(const ExprDoc& lhs, const ffi::Optional<ExprDoc>& rhs,
                                       const ffi::Optional<ExprDoc>& annotation) {
  const TupleDocNode* outer_tuple = unparenthesized_tuple_;
  bool outer_comma = omit_tuple_singleton_comma_;
  unparenthesized_tuple_ = lhs.as<TupleDocNode>();
  omit_tuple_singleton_comma_ = true;
  PrintDoc(lhs);
  unparenthesized_tuple_ = outer_tuple;
  omit_tuple_singleton_comma_ = outer_comma;

  if (annotation) {
    output_ << ": ";
    PrintDoc(annotation.value());
  }
  if (rhs) {
    output_ << " = ";
    if (const auto* tuple = rhs.as<TupleDocNode>(); tuple && tuple->elements.size() > 1) {
      unparenthesized_tuple_ = tuple;
    }
    PrintDoc(rhs.value());
    unparenthesized_tuple_ = outer_tuple;
  }
}

void PythonDocPrinter::PrintTypedDoc(const AssignDoc& doc) {
  PrintAssignment(doc->lhs, doc->rhs, doc->annotation);
  MaybePrintCommentInline(doc);
}

void PythonDocPrinter::PrintTypedDoc(const IfDoc& doc) {
  MaybePrintCommenMultiLines(doc, true);
  output_ << "if ";
  PrintDoc(doc->predicate);
  output_ << ":";

  PrintIndentedBlock(doc->then_branch);

  if (!doc->else_branch.empty()) {
    NewLine();
    output_ << "else:";
    PrintIndentedBlock(doc->else_branch);
  }
}

void PythonDocPrinter::PrintTypedDoc(const WhileDoc& doc) {
  MaybePrintCommenMultiLines(doc, true);
  output_ << "while ";
  PrintDoc(doc->predicate);
  output_ << ":";

  PrintIndentedBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const BreakDoc& doc) { output_ << "break"; }

void PythonDocPrinter::PrintTypedDoc(const ContinueDoc& doc) { output_ << "continue"; }

void PythonDocPrinter::PrintTypedDoc(const ForDoc& doc) {
  MaybePrintCommenMultiLines(doc, true);
  output_ << "for ";
  const TupleDocNode* outer_tuple = unparenthesized_tuple_;
  unparenthesized_tuple_ = doc->lhs.as<TupleDocNode>();
  PrintDoc(doc->lhs);
  unparenthesized_tuple_ = outer_tuple;
  output_ << " in ";
  PrintDoc(doc->rhs);
  output_ << ":";

  PrintIndentedBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  MaybePrintCommenMultiLines(doc, true);
  if (doc->allow_concise_scoping && is_last_stmt_ && !doc->comment.has_value()) {
    if (doc->lhs.has_value()) {
      PrintDoc(doc->lhs.value());
      output_ << " = ";
    }
    PrintDoc(doc->rhs);
    for (size_t i = 0; i < doc->body.size(); ++i) {
      NewLine();
      is_last_stmt_ = i + 1 == doc->body.size();
      PrintDoc(doc->body[i]);
    }
    return;
  }
  output_ << "with ";
  PrintDoc(doc->rhs);
  if (doc->lhs != nullptr) {
    output_ << " as ";
    PrintDoc(doc->lhs.value());
  }
  output_ << ":";

  PrintIndentedBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ExprStmtDoc& doc) {
  PrintDoc(doc->expr);
  MaybePrintCommentInline(doc);
}

void PythonDocPrinter::PrintTypedDoc(const AssertDoc& doc) {
  output_ << "assert ";
  PrintDoc(doc->test);
  if (doc->msg.has_value()) {
    output_ << ", ";
    PrintDoc(doc->msg.value());
  }
  MaybePrintCommentInline(doc);
}

void PythonDocPrinter::PrintTypedDoc(const ReturnDoc& doc) {
  output_ << "return ";
  PrintDoc(doc->value);
  MaybePrintCommentInline(doc);
}

void PythonDocPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  for (const AssignDoc& arg_doc : doc->args) {
    TVM_FFI_ICHECK(!arg_doc->comment.has_value())
        << "Function arg cannot have comment attached to them.";
  }

  PrintDecorators(doc->decorators);

  output_ << "def ";
  PrintDoc(doc->name);
  if (!doc->type_params.empty()) {
    output_ << "[";
    PrintJoinedDocs(doc->type_params, ", ");
    output_ << "]";
  }

  output_ << "(";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";

  if (doc->return_type.has_value()) {
    output_ << " -> ";
    PrintDoc(doc->return_type.value());
  }

  output_ << ":";

  if (doc->comment.has_value()) {
    PrintBlockComment(doc->comment.value());
  }
  PrintIndentedBlock(doc->body);
  NewLineWithoutIndent();
}

void PythonDocPrinter::PrintTypedDoc(const ClassDoc& doc) {
  PrintDecorators(doc->decorators);

  output_ << "class ";
  PrintDoc(doc->name);
  output_ << ":";

  if (doc->comment.has_value()) {
    PrintBlockComment(doc->comment.value());
  }
  PrintIndentedBlock(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const CommentDoc& doc) {
  if (doc->comment.has_value()) {
    MaybePrintCommenMultiLines(doc, false);
  }
}

void PythonDocPrinter::PrintTypedDoc(const DocStringDoc& doc) {
  if (doc->comment.has_value() && !doc->comment.value().empty()) {
    PrintDocString(doc->comment.value());
  }
}

void PythonDocPrinter::PrintTypedDoc(const OpCallDoc& doc) {
  PrintDoc(doc->callee);

  output_ << "(";

  // Print positional args
  bool wrote_any = false;
  for (const Doc& arg : doc->args) {
    if (wrote_any) {
      output_ << ", ";
    }
    wrote_any = true;
    PrintDoc(arg);
  }
  // workspace first (if present and non-empty)
  if (doc->workspace.has_value() && !doc->workspace.value()->keys.empty()) {
    if (wrote_any) output_ << ", ";
    wrote_any = true;
    output_ << "workspace=";
    PrintDoc(doc->workspace.value());
  }
  // dispatch next (if present)
  if (doc->dispatch.has_value()) {
    if (wrote_any) output_ << ", ";
    wrote_any = true;
    output_ << "dispatch=";
    PrintDoc(doc->dispatch.value());
  }
  // Flatten config as keyword args: key=value
  if (doc->config.has_value() && !doc->config.value()->keys.empty()) {
    const auto* dict = doc->config.value().as<DictDocNode>();
    // Only flatten if all keys are literal strings; otherwise, fallback to config={...}
    bool all_str_keys = true;
    for (const ExprDoc& k : dict->keys) {
      if (!k.as<LiteralDocNode>()) {
        all_str_keys = false;
        break;
      }
      const auto* lit = k.as<LiteralDocNode>();
      if (!lit->value.as<ffi::String>()) {
        all_str_keys = false;
        break;
      }
    }
    if (all_str_keys) {
      int n = dict->keys.size();
      for (int i = 0; i < n; ++i) {
        const auto* lit = dict->keys[i].as<LiteralDocNode>();
        std::string key = lit->value.as_or_throw<ffi::String>();
        if (wrote_any) output_ << ", ";
        wrote_any = true;
        output_ << key << "=";
        PrintDoc(dict->values[i]);
      }
    } else {
      if (wrote_any) output_ << ", ";
      wrote_any = true;
      output_ << "config=";
      PrintDoc(doc->config.value());
    }
  }
  output_ << ")";
}

// Associate each recovered path with its nearest statement container.  Expression
// children keep their more precise paths while annotations use the enclosing
// statement's ordinary comment rendering. Transparent statement lists introduce
// no separate comment position.
void CollectAnnotationTargets(ffi::AnyView value, ffi::Optional<StmtDoc> enclosing,
                              std::vector<std::pair<AccessPath, StmtDoc>>* targets,
                              std::unordered_set<const ffi::Object*>* active,
                              bool function_parameter = false) {
  const ffi::Object* object = value.as<ffi::Object>();
  if (!object || !active->insert(object).second) return;
  if (auto stmt = value.as<StmtDoc>();
      stmt && !value.as<StmtBlockDocNode>() && !(function_parameter && value.as<AssignDocNode>())) {
    enclosing = stmt.value();
  }
  if (auto doc = value.as<DocNode>(); doc && enclosing.has_value()) {
    for (const AccessPath& path : doc->source_paths) {
      targets->emplace_back(path, enclosing.value());
    }
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto& item : array.value()) {
      CollectAnnotationTargets(item, enclosing, targets, active, function_parameter);
    }
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto& [key, item] : map.value()) {
      CollectAnnotationTargets(item, enclosing, targets, active, function_parameter);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
          if (ffi::String(field->name) == "source_paths") return;
          // Parameters are AssignDocs for signature layout, but cannot carry
          // statement comments. Keep their expressions' precise source paths
          // while placing requested annotations on the enclosing function.
          bool parameter = function_parameter || (value.as<FunctionDoc>().has_value() &&
                                                  ffi::String(field->name) == "args");
          CollectAnnotationTargets(ffi::reflection::FieldGetter(field)(object), enclosing, targets,
                                   active, parameter);
        });
  }
  active->erase(object);
}

void AttachAnnotations(const Doc& root, const ffi::Map<AccessPath, ffi::String>& annotations) {
  if (annotations.empty()) return;
  std::vector<std::pair<AccessPath, StmtDoc>> targets;
  std::unordered_set<const ffi::Object*> active;
  CollectAnnotationTargets(root, std::nullopt, &targets, &active);
  std::unordered_map<const ffi::Object*, std::unordered_set<std::string>> attached;
  for (const auto& [requested, message] : annotations) {
    int deepest = -1;
    for (const auto& [path, stmt] : targets) {
      if (path->IsPrefixOf(requested)) deepest = std::max(deepest, path->depth);
    }
    for (const auto& [path, stmt] : targets) {
      if (path->depth != deepest || !path->IsPrefixOf(requested)) continue;
      if (!attached[stmt.get()].insert(std::string(message)).second) continue;
      bool inline_comment = stmt.as<AssignDocNode>() || stmt.as<ExprStmtDocNode>() ||
                            stmt.as<AssertDocNode>() || stmt.as<ReturnDocNode>();
      std::string text = message;
      if (inline_comment) {
        // Diagnostic requests may converge on one visible statement. Keep
        // every message without constructing an invalid multiline inline Doc.
        size_t newline = 0;
        while ((newline = text.find('\n', newline)) != std::string::npos) {
          text.replace(newline, 1, "; ");
          newline += 2;
        }
      }
      stmt->comment = stmt->comment.has_value() ? ffi::String(std::string(stmt->comment.value()) +
                                                              (inline_comment ? "; " : "\n") + text)
                                                : ffi::String(text);
    }
  }
}

}  // namespace

namespace details {

ffi::String RenderPythonScript(Doc doc, const PrinterConfig& cfg,
                               const ffi::Array<ffi::Any>& header,
                               const ffi::Array<AccessPath>& underline_paths,
                               const ffi::Map<AccessPath, ffi::String>& annotations) {
  AttachAnnotations(doc, annotations);
  PythonDocPrinter printer(cfg);
  printer.AppendHeader(header);
  printer.Append(doc, underline_paths);
  std::string script = printer.GetString();

  // GetString terminates non-empty output with one newline.  Preserve the
  // established rendering result without normalizing any other
  // trailing whitespace.
  if (!script.empty()) {
    TVM_FFI_ICHECK_EQ(script.back(), '\n');
    script.pop_back();
  }

  if (!cfg->render_invisible_path_info) return script;
  return RenderInvisiblePathInfo(script, underline_paths, printer.GetVisiblePaths());
}

}  // namespace details

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.DocToPythonScript", [](Doc doc,
                                                               const PrinterConfig& config) {
    return details::RenderPythonScript(std::move(doc), config, {}, config->path_to_underline, {});
  });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
