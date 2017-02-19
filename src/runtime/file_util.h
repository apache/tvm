/*!
 *  Copyright (c) 2017 by Contributors
 * \file file_util.h
 * \brief Minimum file manipulation util for runtime.
 */
#ifndef TVM_RUNTIME_FILE_UTIL_H_
#define TVM_RUNTIME_FILE_UTIL_H_

#include <dmlc/logging.h>
#include <fstream>
#include <string>

namespace tvm {
namespace runtime {
/*!
 * \brief Get file format from given file name or format argument.
 * \param file_name The name of the file.
 * \param format The format of the file.
 */
inline std::string GetFileFormat(const std::string& file_name,
                                 const std::string& format) {
  std::string fmt = format;
  if (fmt.length() == 0) {
    size_t pos = file_name.find_last_of(".");
    if (pos != std::string::npos) {
      return file_name.substr(pos + 1, file_name.length() - pos - 1);
    } else {
      return "";
    }
  } else {
    return format;
  }
}

/*!
 * \brief Load binary file into a in-memory buffer.
 * \param file_name The name of the file.
 */
inline std::string LoadBinaryFile(const std::string& file_name) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  CHECK(!fs.fail())
      << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::string data;
  data.resize(size);
  fs.read(&data[0], size);
  return data;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_FILE_UTIL_H_
