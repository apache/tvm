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
 * \file tvm/auto_scheduler/measure_record.h
 * \brief Json serialization format for dumping and loading measurement records.
 */

#ifndef TVM_AUTO_SCHEDULER_MEASURE_RECORD_H_
#define TVM_AUTO_SCHEDULER_MEASURE_RECORD_H_

#include <tvm/auto_scheduler/measure.h>

#include <fstream>
#include <string>
#include <utility>

namespace tvm {
namespace auto_scheduler {

const std::string AUTO_SCHEDULER_LOG_VERSION = "v0.6";  // NOLINT(*)

/*! \brief Callback for logging the input and results of measurements to file */
class RecordToFileNode : public MeasureCallbackNode {
 public:
  /*! \brief The name of output file. */
  String filename;

  void Callback(const SearchPolicy& policy, const Array<MeasureInput>& inputs,
                const Array<MeasureResult>& results) final;

  static constexpr const char* _type_key = "auto_scheduler.RecordToFile";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecordToFileNode, MeasureCallbackNode);
};

/*!
 * \brief Managed reference to RecordToFileNode.
 * \sa RecordToFileNode
 */
class RecordToFile : public MeasureCallback {
 public:
  /*!
   * \brief The constructor.
   * \param filename The name of output file
   */
  explicit RecordToFile(String filename);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RecordToFile, MeasureCallback, RecordToFileNode);
};

/*! \brief Log reader to load step logs from a file.*/
class RecordReaderNode : public Object {
 public:
  /*! \brief The name of input file. */
  String filename;
  /*! \brief The reading file stream. */
  std::ifstream infile;

  ~RecordReaderNode();

  /*!
   * \brief Read next line in the log file.
   * \param inp A pointer to a MeasureInputNode, this is used as output.
   * \param res A pointer to a MeasureResultNode, this is used as output.
   * \return Whether the read is successful. */
  bool ReadNext(MeasureInputNode* inp, MeasureResultNode* res);

  /*!
   * \brief Read multiple lines from the log file.
   * \param max_size The maximum number of lines. -1 means read all lines.
   * \param skip_size Skip the first n lines.
   * \return The MeasureInputs and MeasureResults loaded from the log file.
   */
  std::pair<Array<MeasureInput>, Array<MeasureResult>> ReadLines(int max_size = -1,
                                                                 int skip_size = 0);

  static constexpr const char* _type_key = "auto_scheduler.RecordReader";
  TVM_DECLARE_FINAL_OBJECT_INFO(RecordReaderNode, Object);

 private:
  /*! \brief A string storing the current line. */
  std::string cur_line_;
};

/*!
 * \brief Managed reference to RecordReaderNode.
 * \sa RecordReaderNode
 */
class RecordReader : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param filename The name of input file
   */
  explicit RecordReader(String filename);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RecordReader, ObjectRef, RecordReaderNode);
};

/*!
 * \brief Append measure records to an output stream.
 * \param os A pointer to a output stream.
 * \param inputs The MeasureInputs to be written.
 * \param results The MeasureResults to be written.
 * \param log_version The log version for the given record.
 */
void WriteMeasureRecords(std::ostream* os, const Array<MeasureInput>& inputs,
                         const Array<MeasureResult>& results,
                         const std::string log_version = AUTO_SCHEDULER_LOG_VERSION);

/*!
 * \brief Read one measure record from a string.
 * \param str The record string to be parsed.
 * \param inp A pointer to a MeasureInputNode used to store the return value.
 * \param res A pointer to a MeasureResultNode used to store the return value.
 * \param log_version A pointer to a string used to store the log version.
 */
void ReadMeasureRecord(const std::string& str, MeasureInputNode* inp, MeasureResultNode* res,
                       std::string* log_version);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_MEASURE_RECORD_H_
