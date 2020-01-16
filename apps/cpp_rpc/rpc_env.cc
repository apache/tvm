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
 * \file rpc_env.cc
 * \brief Server environment of the RPC.
 */
#include <tvm/runtime/registry.h>
#include <errno.h>
#ifndef _MSC_VER
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cstring>

#include "rpc_env.h"
#include "../../src/support/util.h"
#include "../../src/runtime/file_util.h"

namespace tvm {
namespace runtime {

RPCEnv::RPCEnv() {
  #if defined(__linux__) || defined(__ANDROID__)
    base_ = "./rpc";
    mkdir(&base_[0], 0777);

    TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
        static RPCEnv env;
        *rv = env.GetPath(args[0]);
      });

    TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
        static RPCEnv env;
        std::string file_name = env.GetPath(args[0]);
        *rv = Load(&file_name, "");
        LOG(INFO) << "Load module from " << file_name << " ...";
      });
  #else
    LOG(FATAL) << "Only support RPC in linux environment";
  #endif
}
/*!
 * \brief GetPath To get the workpath from packed function
 * \param name The file name
 * \return The full path of file.
 */
std::string RPCEnv::GetPath(std::string file_name) {
  // we assume file_name has "/" means file_name is the exact path
  // and does not create /.rpc/
  if (file_name.find("/") != std::string::npos) {
    return file_name;
  } else {
    return base_ + "/" + file_name;
  }
}
/*!
 * \brief Remove The RPC Environment cleanup function
 */
void RPCEnv::CleanUp() {
  #if defined(__linux__) || defined(__ANDROID__)
    CleanDir(&base_[0]);
    int ret = rmdir(&base_[0]);
    if (ret != 0) {
      LOG(WARNING) << "Remove directory " << base_ << " failed";
    }
  #else
    LOG(FATAL) << "Only support RPC in linux environment";
  #endif
}

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string &dirname) {
  std::vector<std::string> vec;
  #ifndef _MSC_VER
    DIR *dp = opendir(dirname.c_str());
    if (dp == nullptr) {
      int errsv = errno;
      LOG(FATAL) << "ListDir " << dirname <<" error: " << strerror(errsv);
    }
    dirent *d;
    while ((d = readdir(dp)) != nullptr) {
      std::string filename = d->d_name;
      if (filename != "." && filename != "..") {
        std::string f = dirname;
        if (f[f.length() - 1] != '/') {
          f += '/';
        }
        f += d->d_name;
        vec.push_back(f);
      }
    }
    closedir(dp);
  #else
    WIN32_FIND_DATA fd;
    std::string pattern = dirname + "/*";
    HANDLE handle = FindFirstFile(pattern.c_str(), &fd);
    if (handle == INVALID_HANDLE_VALUE) {
      int errsv = GetLastError();
      LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
    }
    do {
      if (fd.cFileName != "." && fd.cFileName != "..") {
        std::string  f = dirname;
        char clast = f[f.length() - 1];
        if (f == ".") {
          f = fd.cFileName;
        } else if (clast != '/' && clast != '\\') {
          f += '/';
          f += fd.cFileName;
        }
        vec.push_back(f);
      }
    }  while (FindNextFile(handle, &fd));
    FindClose(handle);
  #endif
  return vec;
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
    std::string err_msg;
    auto executed_status = support::Execute(cmd, &err_msg);
    if (executed_status) {
      LOG(FATAL) << err_msg;
    }
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
 * \param fileIn The input file, file name will be updated
 * \param fmt The format of file
 * \return Module The loaded module
 */
Module Load(std::string *fileIn, const std::string fmt) {
  std::string file = *fileIn;
  if (support::EndsWith(file, ".so")) {
      return Module::LoadFromFile(file, fmt);
  }

  #if defined(__linux__) || defined(__ANDROID__)
    std::string file_name = file + ".so";
    if (support::EndsWith(file, ".o")) {
      std::vector<std::string> files;
      files.push_back(file);
      CreateShared(file_name, files);
    } else if (support::EndsWith(file, ".tar")) {
      std::string tmp_dir = "./rpc/tmp/";
      mkdir(&tmp_dir[0], 0777);
      std::string cmd = "tar -C " + tmp_dir + " -zxf " + file;
      std::string err_msg;
      int executed_status = support::Execute(cmd, &err_msg);
      if (executed_status) {
        LOG(FATAL) << err_msg;
      }
      CreateShared(file_name, ListDir(tmp_dir));
      CleanDir(tmp_dir);
      rmdir(&tmp_dir[0]);
    } else {
      file_name = file;
    }
    *fileIn = file_name;
    return Module::LoadFromFile(file_name, fmt);
  #else
    LOG(FATAL) << "Do not support creating shared library";
  #endif
}

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname The name of the directory
 */
void CleanDir(const std::string &dirname) {
  #if defined(__linux__) || defined(__ANDROID__)
    DIR *dp = opendir(dirname.c_str());
    dirent *d;
    while ((d = readdir(dp)) != nullptr) {
      std::string filename = d->d_name;
      if (filename != "." && filename != "..") {
        filename = dirname + "/" + d->d_name;
        int ret = std::remove(&filename[0]);
        if (ret != 0) {
          LOG(WARNING) << "Remove file " << filename << " failed";
        }
      }
    }
  #else
    LOG(FATAL) << "Only support RPC in linux environment";
  #endif
}

}  // namespace runtime
}  // namespace tvm
