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
#ifndef TVM_SUPPORT_TABLE_PRINTER_H_
#define TVM_SUPPORT_TABLE_PRINTER_H_

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace support {

/*!
 * \brief TablePrinter is a helper class to print a table.
 *
 * \code
 *
 * TablePrinter p;
 * p.Row() << "ID"
 *         << "Latency (ms)"
 *         << "Speed (GFLOPS)"
 *         << "Trials";
 * p.Separator();
 * p.Row() << 0 << 0.072 << 4208.59 << 6656;
 * p.Row() << 1 << 0.020 << 3804.24 << 7296;
 * p.Row() << 2 << 0.003 << 1368.10 << 320;
 * p.Row() << 3 << 0.010 << 117.75 << 128;
 * p.Row() << 4 << 0.002 << 23.75 << 320;
 * p.Row() << 5 << 0.004 << 1696.18 << 704;
 * p.Row() << 6 << 0.002 << 69.89 << 320;
 * p.Row() << 7 << 0.047 << 6394.42 << 4352;
 * p.Separator();
 * std::cout << tab.AsStr();
 *
 * \endcode
 */
class TablePrinter {
  struct Line;

 public:
  /*! \brief Create a new row */
  inline Line Row();
  /*! \brief Create a row separator */
  inline void Separator();
  /*! \brief Converts TablePrinter to a string */
  inline std::string AsStr() const;

 private:
  std::vector<std::vector<std::string>> tab_;
  friend struct Line;

  /*! \brief A helper class to print a specific row in the table */
  struct Line {
    inline Line& operator<<(int x);
    inline Line& operator<<(int64_t x);
    inline Line& operator<<(double x);
    inline Line& operator<<(const std::string& x);

   private:
    TablePrinter* p;
    friend class TablePrinter;
  };
};

inline TablePrinter::Line& TablePrinter::Line::operator<<(int x) {
  p->tab_.back().push_back(std::to_string(x));
  return *this;
}

inline TablePrinter::Line& TablePrinter::Line::operator<<(int64_t x) {
  p->tab_.back().push_back(std::to_string(x));
  return *this;
}

inline TablePrinter::Line& TablePrinter::Line::operator<<(double x) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4) << x;
  p->tab_.back().push_back(os.str());
  return *this;
}

inline TablePrinter::Line& TablePrinter::Line::operator<<(const std::string& x) {
  p->tab_.back().push_back(x);
  return *this;
}

inline TablePrinter::Line TablePrinter::Row() {
  tab_.emplace_back();
  Line line;
  line.p = this;
  return line;
}

inline void TablePrinter::Separator() { tab_.emplace_back(); }

inline std::string TablePrinter::AsStr() const {
  constexpr char kRowSep = '-';
  constexpr char kColSep = '|';
  if (tab_.empty()) return "";
  std::vector<size_t> column_width;
  for (const std::vector<std::string>& row : tab_) {
    if (row.size() > column_width.size()) {
      column_width.resize(row.size(), 0);
    }
    for (size_t i = 0; i < row.size(); ++i) {
      column_width[i] = std::max(column_width[i], row[i].size());
    }
  }
  ICHECK(!column_width.empty());
  size_t total_width =
      std::accumulate(column_width.begin(), column_width.end(), 0) + 3 * column_width.size() - 1;
  bool is_first = true;
  std::ostringstream os;
  for (const std::vector<std::string>& row : tab_) {
    if (is_first) {
      is_first = false;
    } else {
      os << '\n';
    }
    if (row.empty()) {
      os << std::string(total_width, kRowSep);
      continue;
    }
    for (size_t i = 0; i < column_width.size(); ++i) {
      if (i != 0) {
        os << kColSep;
      }
      std::string s = (i < row.size()) ? row[i] : "";
      os << std::string(column_width[i] + 1 - s.size(), ' ') << s << ' ';
    }
  }
  return os.str();
}

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_TABLE_PRINTER_H_
