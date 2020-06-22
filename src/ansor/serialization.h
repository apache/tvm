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
 * \file ansor/serialization.h
 * \brief Json serialization format for dumping and loading tuning records
 */

#ifndef TVM_ANSOR_SERIALIZATION_H_
#define TVM_ANSOR_SERIALIZATION_H_

#include <string>
#include <fstream>
#include <utility>
#include "measure.h"

namespace tvm {
namespace ansor {

/*! \brief Callback for logging the input and results of measurements to file */
class LogToFileNode : public MeasureCallbackNode {
 public:
  std::string filename;

  /*! \brief Log measure pairs to file. This is called by the search policy */
  void callback(const SearchPolicy& policy,
                const Array<MeasureInput>& inputs,
                const Array<MeasureResult>& results) final;

  static constexpr const char *_type_key = "ansor.LogToFile";
  TVM_DECLARE_FINAL_OBJECT_INFO(LogToFileNode, MeasureCallbackNode);
};

/*!
 * \brief Managed reference to LogToFileNode.
 * \sa LogToFileNode
 */
class LogToFile : public MeasureCallback {
 public:
  explicit LogToFile(std::string filename);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LogToFile, MeasureCallback, LogToFileNode);
};

/*! \brief Log reader to load step logs from a target file.*/
class LogReaderNode : public Object {
 public:
  std::string filename;
  std::ifstream infile;

  ~LogReaderNode();

  /*! \brief Read next line in the log file
   *  \return Whether the read is successful */
  bool ReadNext(MeasureInputNode* inp, MeasureResultNode* res);

  /*! \brief Read multiple lines from the log file
   *  \param max_size The maximum number of lines. -1 means read all lines
   *  \param skip_size Skip the first n lines */
  std::pair<Array<MeasureInput>, Array<MeasureResult> > ReadLines(
          int max_size = -1, int skip_size = 0);

  static constexpr const char* _type_key = "ansor.LogReader";
  TVM_DECLARE_FINAL_OBJECT_INFO(LogReaderNode, Object);

 private:
  std::string cur_line;
};

/*!
 * \brief Managed reference to LogReaderNode.
 * \sa LogReaderNode
 */
class LogReader : public ObjectRef {
 public:
  explicit LogReader(std::string filename);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LogReader, ObjectRef, LogReaderNode);
};

/*! \brief Write measure records to an output stream */
void WriteMeasureRecords(std::ostream* os,
                         const Array<MeasureInput>& inputs,
                         const Array<MeasureResult>& results);

/*! \brief Read one measure record from a string */
void ReadMeasureRecord(const std::string& str,
                       MeasureInputNode* inp,
                       MeasureResultNode* res,
                       std::string* log_version);

/*! \brief Return the best measure pair with lowest cost in a file */
std::pair<MeasureInput, MeasureResult> BestMeasurePairInFile(const std::string& filename,
                                                             const std::string& workload_key,
                                                             const Target& target);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SERIALIZATION_H_
