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

// Declares a utility class used to parse command-line options.

#ifndef AOCL_UTILS_OPTIONS_H
#define AOCL_UTILS_OPTIONS_H

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace aocl_utils {

class Options {
public:
  typedef std::vector<std::string> StringVec;

  Options();
  Options(int num, char *argv[]);

  bool has(const std::string &name) const;
  std::string &get(const std::string &name); // will create an empty option if it does not exist
  const std::string &get(const std::string &name) const; // error if option does not exist

  void set(const std::string &name, const std::string &value) { get(name) = value; }

  // Command line options must be of the following form:
  //  [-]-name (indicates option exists)
  //  [-]-name=value
  //
  // This function assumes that the values are from main(int, char *).
  // This means that the argv[0] is skipped.
  void addFromCommandLine(int num, char *argv[]);

  // This templated function converts the option value to the given type.
  // An assert is raised if the conversion fails.
  template<typename T>
  T get(const std::string &name) const;

  template<typename T>
  void set(const std::string &name, const T &value);

  // Non-options are arguments processed in addFromCommandLine
  // that were not recognized as options.
  const StringVec &getNonOptions() const { return m_nonoptions; }
  size_t getNonOptionCount() const { return m_nonoptions.size(); }
  const std::string &getNonOption(size_t i) const { return m_nonoptions[i]; }

private:
  typedef std::map<std::string, std::string> OptionMap;

  // Displays an error message indicating that a nameless option
  // was provided.
  void errorNameless() const;

  // Displays an error message indicating that the given option
  // has the wrong type and then exits with an error code.
  void errorWrongType(const std::string &name) const;

  // Displays an error message indicating that the given option
  // does not exist and then exits with an error code.
  void errorNonExistent(const std::string &name) const;

  OptionMap m_options;
  StringVec m_nonoptions;

  Options(const Options &); // not implemented
  void operator =(const Options &); // not implemented
};

template<typename T>
T Options::get(const std::string &name) const {
  std::stringstream ss;
  ss << get(name);

  T v;
  ss >> v;
  if(ss.fail() || !ss.eof()) {
    // Failed to parse or did not consume the whole string value.
    errorWrongType(name);
  }
  return v;
}

// Specialization for bool. 
template<>
inline bool Options::get<bool>(const std::string &name) const {
  if(has(name)) {
    const std::string &v = get(name);
    if(v == "1") {
      return true;
    }
  }
  return false;
}

// Specialization for std::string. Simply returns the option string.
// Requires specialization because using stringstream to read the string
// will stop at the first whitespace character (which is wrong).
template<>
inline std::string Options::get<std::string>(const std::string &name) const {
  return get(name);
}

// This assumes the type T can be serialized to a string and back (when get
// is called).
template<typename T>
void Options::set(const std::string &name, const T &value) {
  std::stringstream ss;
  ss << value;
  set(name, ss.str());
}

} // ns aocl_utils

#endif

