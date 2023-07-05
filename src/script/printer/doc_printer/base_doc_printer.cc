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

#include "./base_doc_printer.h"

namespace tvm {
namespace script {
namespace printer {

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
  size_t first_line_of_underline = GetLineIndex(underline.first, line_starts);
  size_t first_line_of_chunk = MoveBack(first_line_of_underline, options->num_context_lines);
  size_t end_line_of_underline = GetLineIndex(underline.second - 1, line_starts) + 1;
  size_t end_line_of_chunk =
      MoveForward(end_line_of_underline, options->num_context_lines, num_lines);

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

void DocPrinter::Append(const Doc& doc) { Append(doc, PrinterConfig()); }

void DocPrinter::Append(const Doc& doc, const PrinterConfig& cfg) {
  for (const ObjectPath& p : cfg->path_to_underline) {
    path_to_underline_.push_back(p);
    current_max_path_length_.push_back(0);
    current_underline_candidates_.push_back(std::vector<ByteSpan>());
  }
  PrintDoc(doc);
  for (const auto& c : current_underline_candidates_) {
    underlines_.insert(underlines_.end(), c.begin(), c.end());
  }
}

String DocPrinter::GetString() const {
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

void DocPrinter::PrintDoc(const Doc& doc) {
  size_t start_pos = output_.tellp();

  if (auto doc_node = doc.as<LiteralDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IdDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AttrAccessDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IndexDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<OperationDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<CallDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<LambdaDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ListDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<TupleDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<DictDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<SliceDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<StmtBlockDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AssignDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IfDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<WhileDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ForDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ScopeDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ExprStmtDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AssertDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ReturnDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<FunctionDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ClassDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<CommentDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<DocStringDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }

  size_t end_pos = output_.tellp();
  for (const ObjectPath& path : doc->source_paths) {
    MarkSpan({start_pos, end_pos}, path);
  }
}

void DocPrinter::MarkSpan(const ByteSpan& span, const ObjectPath& path) {
  int n = path_to_underline_.size();
  for (int i = 0; i < n; ++i) {
    ObjectPath p = path_to_underline_[i];
    if (path->Length() >= current_max_path_length_[i] && path->IsPrefixOf(p)) {
      if (path->Length() > current_max_path_length_[i]) {
        current_max_path_length_[i] = path->Length();
        current_underline_candidates_[i].clear();
      }
      current_underline_candidates_[i].push_back(span);
    }
  }
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
