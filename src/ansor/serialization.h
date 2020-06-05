/*!
 *  Copyright (c) 2020 by Contributors
 * \file ansor/serialization.h
 * \brief Json serialization format for dumping and loading tuning records
 */

#ifndef TVM_ANSOR_SERIALIZATION_H_
#define TVM_ANSOR_SERIALIZATION_H_

#include <fstream>
#include <string>
#include <utility>

#include "measure.h"

namespace tvm {
namespace ansor {

class LogReader;

/*! \brief Log the input and results of measurments to file */
class LogToFileNode : public MeasureCallbackNode {
 public:
  std::string filename;

  static MeasureCallback make(std::string filename);

  /*! \brief Log measure pairs to file. This is called by the search policy */
  void callback(const SearchPolicy& policy, const Array<MeasureInput>& inputs,
                const Array<MeasureResult>& results) final;

  static constexpr const char* _type_key = "ansor.LogToFile";
  TVM_DECLARE_FINAL_OBJECT_INFO(LogToFileNode, MeasureCallbackNode);
};

/*! \brief Log reader */
class LogReaderNode : public Object {
 public:
  std::string filename;
  std::ifstream infile;

  static LogReader make(std::string filename);

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
TVM_DEFINE_MUTABLE_NODE_REF(LogReader, LogReaderNode);

void WriteMeasureRecords(std::ostream* os, const Array<MeasureInput>& inputs,
                         const Array<MeasureResult>& results);

void ReadMeasureRecords(std::string str, MeasureInputNode* inp,
                        MeasureResultNode* res, std::string* log_version);

std::pair<MeasureInput, MeasureResult> BestMeasurePairInFile(
    const std::string& filename, const std::string& workload_key,
    const Target& target);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SERIALIZATION_H_
