// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include "aocl_utils.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace aocl_utils {

Options::Options() {
}

Options::Options(int num, char *argv[]) {
  addFromCommandLine(num, argv);
}

bool Options::has(const std::string &name) const {
  return m_options.find(name) != m_options.end();
}

std::string &Options::get(const std::string &name) {
  return m_options[name];
}

const std::string &Options::get(const std::string &name) const {
  OptionMap::const_iterator it = m_options.find(name);
  if(it == m_options.end()) {
    errorNonExistent(name);
    std::cerr << "Option '" << name << "' does not exist.\n";
    exit(1);
  }
  return it->second;
}

void Options::addFromCommandLine(int num, char *argv[]) {
  for(int i = 1; i < num; ++i) {
    const std::string arg = argv[i];

    // Look for the first '-'.
    if(arg.size() > 1 && arg[0] == '-') {
      size_t eq = arg.find('=');
      size_t name_start = 1;

      // Check if there's a second '-'.
      if(arg.size() > 2 && arg[1] == '-') {
        name_start = 2;
      }

      if(eq == std::string::npos) {
        // No '='; treat as a boolean option.
        set(arg.substr(name_start), true);
      }
      else if(eq == name_start) {
        // No name?!
        errorNameless();
      }
      else {
        set(arg.substr(name_start, eq - name_start), arg.substr(eq + 1));
      }
    }
    else {
      // Not an option.
      m_nonoptions.push_back(arg);
    }
  }
}

void Options::errorNameless() const {
  std::cerr << "No name provided for option.\n";
  exit(1);
}

void Options::errorNonExistent(const std::string &name) const {
  std::cerr << "Option '" << name << "' does not exist.\n";
  exit(1);
}

void Options::errorWrongType(const std::string &name) const {
  std::cerr << "Value for option '" << name << "' is not of the right type (value = '"
            << get(name) << "').\n";
  exit(1);
}

} // ns aocl_utils

