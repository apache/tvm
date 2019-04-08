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

/*!
 *  Copyright (c) 2017 by Contributors
 * \file file_util.h
 * \brief Minimum file manipulation util for runtime.
 */
#ifndef TVM_RUNTIME_FILE_UTIL_H_
#define TVM_RUNTIME_FILE_UTIL_H_

#include <string>
#include <unordered_map>
#include "meta_data.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Get file format from given file name or format argument.
 * \param file_name The name of the file.
 * \param format The format of the file.
 */
std::string GetFileFormat(const std::string& file_name,
                          const std::string& format);

/*!
 * \return the directory in which TVM stores cached files.
 *         May be set using TVM_CACHE_DIR; defaults to system locations.
 */
std::string GetCacheDir();

/*!
 * \brief Get meta file path given file name and format.
 * \param file_name The name of the file.
 */
std::string GetMetaFilePath(const std::string& file_name);

/*!
 * \brief Get file basename (i.e. without leading directories)
 * \param file_name The name of the file.
 * \return the base name
 */
std::string GetFileBasename(const std::string& file_name);

/*!
 * \brief Load binary file into a in-memory buffer.
 * \param file_name The name of the file.
 * \param data The data to be loaded.
 */
void LoadBinaryFromFile(const std::string& file_name,
                        std::string* data);

/*!
 * \brief Load binary file into a in-memory buffer.
 * \param file_name The name of the file.
 * \param data The binary data to be saved.
 */
void SaveBinaryToFile(const std::string& file_name,
                      const std::string& data);

/*!
 * \brief Save meta data to file.
 * \param file_name The name of the file.
 * \param fmap The function info map.
 */
void SaveMetaDataToFile(
    const std::string& file_name,
    const std::unordered_map<std::string, FunctionInfo>& fmap);

/*!
 * \brief Load meta data to file.
 * \param file_name The name of the file.
 * \param fmap The function info map.
 */
void LoadMetaDataFromFile(
    const std::string& file_name,
    std::unordered_map<std::string, FunctionInfo>* fmap);

/*!
 * \brief Remove (unlink) a file.
 * \param file_name The file name.
 */
void RemoveFile(const std::string& file_name);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_FILE_UTIL_H_
