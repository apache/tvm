/*!
 *  Copyright (c) 2018 by Contributors
 * \file util.h
 * \brief Defines some common utility function..
 */
#ifndef TVM_COMMON_UTIL_H_
#define TVM_COMMON_UTIL_H_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

namespace tvm {
namespace common {

/*!
 * \brief IsNumber check whether string is a number.
 * \param str input string
 * \return result of operation.
 */
inline bool IsNumber(const std::string& str) {
  return !str.empty() && std::find_if(str.begin(),
      str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
}

/*!
 * \brief split Split the string based on delimiter
 * \param str Input string
 * \param delim The delimiter.
 * \return vector of strings which are splitted.
 */
inline std::vector<std::string> Split(const std::string& str, char delim) {
  std::string item;
  std::istringstream is(str);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief EndsWith check whether the strings ends with
 * \param value The full string
 * \param end The end substring
 * \return bool The result.
 */
inline bool EndsWith(std::string const & value, std::string const & end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

}  // namespace common
}  // namespace tvm
#endif  // TVM_COMMON_UTIL_H_

