/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_server_env.cc
 * \brief Server environment of the RPC.
 */
#include <tvm/runtime/registry.h>
#if defined(__linux__)
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#endif
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

#include "../file_util.h"
#include "rpc_server_env.h"

namespace tvm {
namespace runtime {

/*!
 * \brief EndsWith check whether the strings ends with
 * \param value The full string
 * \param end The end substring
 * \return bool The result.
 */
bool EndsWith(std::string const & value, std::string const & end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string dirname) {
  #if defined(__linux__) || defined(__ANDROID__)
    std::vector<std::string> vec;
    DIR *dp = opendir(dirname.c_str());
    dirent *d;
    while ((d = readdir(dp)) != NULL) {
      std::string filename = d->d_name;
      if (filename != "." && filename != "..") {
        vec.push_back(dirname + d->d_name);
      }
    }
    return vec;
  #else
    LOG(FATAL) << "Only support RPC in linux environment";
  #endif
}

/*!
 * \brief LinuxShared Creates a linux shared library
 * \param output The output file name
 * \param files The files for building
 * \param options The compiler options
 * \param cc The compiler
 */
void LinuxShared(const std::string output,
                 const std::vector<std::string> &files,
                 std::string options = "",
                 std::string cc = "g++") {
    std::string cmd = cc;
    cmd += " -shared -fPIC ";
    cmd += " -o " + output;
    for (auto f = files.begin(); f != files.end(); ++f) {
     cmd += " " + *f;
    }
    cmd += " " + options;
    CHECK(system(cmd.c_str()) == 0) << "Compilation error.";
}

/*!
 * \brief CreateShared Creates a shared library
 * \param output The output file name
 * \param files The files for building
 */
void CreateShared(const std::string output, const std::vector<std::string> &files) {
  #if defined(__linux__) || defined(__ANDROID__)
    LinuxShared(output, files);
  #else
    LOG(FATAL) << "Do not support creating shared library";
  #endif
}

/*!
 * \brief Load Load module from file
          This function will automatically call
          cc.create_shared if the path is in format .o or .tar
          High level handling for .o and .tar file.
          We support this to be consistent with RPC module load.
 * \param fileIn The input file
 * \param file The format of file
 * \return Module The loaded module
 */
Module Load(std::string *fileIn, const std::string fmt) {
  std::string file = *fileIn;
  if (EndsWith(file, ".so")) {
      return Module::LoadFromFile(file, fmt);
  }

  #if defined(__linux__) || defined(__ANDROID__)
    std::string file_name = file + ".so";
    if (EndsWith(file, ".o")) {
      std::vector<std::string> files;
      files.push_back(file);
      CreateShared(file_name, files);
    } else if (EndsWith(file, ".tar")) {
      std::string tmp_dir = "rpc/tmp/";
      mkdir(&tmp_dir[0], 0777);
      std::string cmd = "tar -C " + tmp_dir + " -zxf " + file;
      CHECK(system(cmd.c_str()) == 0) << "Untar library error.";
      CreateShared(file_name, ListDir(tmp_dir));
      CleanDir(tmp_dir);
      rmdir(&tmp_dir[0]);
    }
    *fileIn = file_name;
    return Module::LoadFromFile(file_name, fmt);
  #else
    LOG(FATAL) << "Donot support creating shared library";
  #endif
}

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname THe name of the directory
 */
void CleanDir(const std::string dirname) {
  #if defined(__linux__) || defined(__ANDROID__)
    DIR *dp = opendir(dirname.c_str());
    dirent *d;
    while ((d = readdir(dp)) != NULL) {
      std::string filename = d->d_name;
      if (filename != "." && filename != "..") {
        filename = dirname + d->d_name;
        std::remove(&filename[0]);
      }
    }
  #else
    LOG(FATAL) << "Only support RPC in linux environment";
  #endif
}

/*!
 * \brief RPCGetPath To get the workpath from packed function
 * \param name The file name
 * \return The full path of file.
 */
std::string RPCGetPath(const std::string& name) {
  static const PackedFunc* f =
      runtime::Registry::Get("tvm.rpc.server.workpath");
  CHECK(f != nullptr) << "require tvm.rpc.server.workpath";
  return (*f)(name);
}

TVM_REGISTER_GLOBAL("tvm.rpc.server.upload").
set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string file_name = RPCGetPath(args[0]);
    std::string data = args[1];
    SaveBinaryToFile(file_name, data);
  });

TVM_REGISTER_GLOBAL("tvm.rpc.server.download")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string file_name = RPCGetPath(args[0]);
    std::string data;
    LoadBinaryFromFile(file_name, &data);
    TVMByteArray arr;
    arr.data = data.c_str();
    arr.size = data.length();
    LOG(INFO) << "Download " << file_name << "... nbytes=" << arr.size;
    *rv = arr;
  });

}  // namespace runtime
}  // namespace tvm
