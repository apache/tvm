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
#include <cerrno>
#include <tvm/runtime/registry.h>
#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <Windows.h>
#include <direct.h>
namespace {
  int mkdir(const char* path, int /* ignored */) { return _mkdir(path); }
}
#endif
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string>

#include "../../src/support/util.h"
#include "../../src/runtime/file_util.h"
#include "rpc_env.h"

namespace {
  std::string GenerateUntarCommand(const std::string& tar_file, const std::string& output_dir) {
    std::string untar_cmd;
    untar_cmd.reserve(512);
#if defined(__linux__) || defined(__ANDROID__)
    untar_cmd += "tar -C ";
    untar_cmd += output_dir;
    untar_cmd += " -zxf ";
    untar_cmd += tar_file;
#elif defined(_WIN32)
    untar_cmd += "python -m tarfile -e ";
    untar_cmd += tar_file;
    untar_cmd += " ";
    untar_cmd += output_dir;
#endif
    return untar_cmd;
  }

}// Anonymous namespace

namespace tvm {
namespace runtime {
RPCEnv::RPCEnv() {
  base_ = "./rpc";
  mkdir(base_.c_str(), 0777);
  TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([](TVMArgs args, TVMRetValue* rv) {
    static RPCEnv env;
    *rv = env.GetPath(args[0]);
  });

  TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module").set_body([](TVMArgs args, TVMRetValue* rv) {
    static RPCEnv env;
    std::string file_name = env.GetPath(args[0]);
    *rv = Load(&file_name, "");
    LOG(INFO) << "Load module from " << file_name << " ...";
  });
}
/*!
 * \brief GetPath To get the work path from packed function
 * \param file_name The file name
 * \return The full path of file.
 */
std::string RPCEnv::GetPath(const std::string& file_name) const {
  // we assume file_name has "/" means file_name is the exact path
  // and does not create /.rpc/
  return file_name.find('/') != std::string::npos ? file_name : base_ + "/" + file_name;
}
/*!
 * \brief Remove The RPC Environment cleanup function
 */
void RPCEnv::CleanUp() const {
  CleanDir(base_);
  const int ret = rmdir(base_.c_str());
  if (ret != 0) {
    LOG(WARNING) << "Remove directory " << base_ << " failed";
  }
}

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string& dirname) {
  std::vector<std::string> vec;
#ifndef _WIN32
  DIR* dp = opendir(dirname.c_str());
  if (dp == nullptr) {
    int errsv = errno;
    LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
  }
  dirent* d;
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
#elif defined(_WIN32)
  WIN32_FIND_DATAA fd;
  const std::string pattern = dirname + "/*";
  HANDLE handle = FindFirstFileA(pattern.c_str(), &fd);
  if (handle == INVALID_HANDLE_VALUE) {
    const int errsv = GetLastError();
    LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
  }
  do {
    std::string filename = fd.cFileName;
    if (filename != "." && filename != "..") {
      std::string f = dirname;
      if (f[f.length() - 1] != '/') {
        f += '/';
      }
      f += filename;
      vec.push_back(f);
    }
  } while (FindNextFileA(handle, &fd));
  FindClose(handle);
#else
  LOG(FATAL) << "Operating system not supported";
#endif
  return vec;
}

#if defined(__linux__) || defined(__ANDROID__)
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
#endif

#ifdef _WIN32
/*!
 * \brief WindowsShared Creates a Windows shared library
 * \param output The output file name
 * \param files The files for building
 * \param options The compiler options
 * \param cc The compiler
 */
void WindowsShared(const std::string& output, 
                   const std::vector<std::string>& files,
                   const std::string& options = "", 
                   const std::string& cc = "clang") {
  std::string cmd = cc;
  cmd += " -O2 -flto=full -fuse-ld=lld-link -Wl,/EXPORT:__tvm_main__ -shared ";
  cmd += " -o " + output;
  for (const auto& file : files) {
    cmd += " " + file;
  }
  cmd += " " + options;
  std::string err_msg;
  const auto executed_status = support::Execute(cmd, &err_msg);
  if (executed_status) {
    LOG(FATAL) << err_msg;
  }
}
#endif

/*!
 * \brief CreateShared Creates a shared library
 * \param output The output file name
 * \param files The files for building
 */
void CreateShared(const std::string& output, const std::vector<std::string>& files) {
#if defined(__linux__) || defined(__ANDROID__)
  LinuxShared(output, files);
#elif defined(_WIN32)
  WindowsShared(output, files);
#else
  LOG(FATAL) << "Operating system not supported";
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
Module Load(std::string *fileIn, const std::string& fmt) {
  const std::string& file = *fileIn;
  if (support::EndsWith(file, ".so") || support::EndsWith(file, ".dll")) {
    return Module::LoadFromFile(file, fmt);
  }

  std::string file_name = file + ".so";
  if (support::EndsWith(file, ".o")) {
    std::vector<std::string> files;
    files.push_back(file);
    CreateShared(file_name, files);
  } else if (support::EndsWith(file, ".tar")) {
    const std::string tmp_dir = "./rpc/tmp/";
    mkdir(tmp_dir.c_str(), 0777);

    const std::string cmd = GenerateUntarCommand(file, tmp_dir);

    std::string err_msg;
    const int executed_status = support::Execute(cmd, &err_msg);
    if (executed_status) {
      LOG(FATAL) << err_msg;
    }
    CreateShared(file_name, ListDir(tmp_dir));
    CleanDir(tmp_dir);
    (void)rmdir(tmp_dir.c_str());
  } else {
    file_name = file;
  }
  *fileIn = file_name;
  return Module::LoadFromFile(file_name, fmt);
}

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname The name of the directory
 */
void CleanDir(const std::string& dirname) {
  auto files = ListDir(dirname);
  for (const auto& filename : files) {
    std::string file_path = dirname + "/";
    file_path += filename;
    const int ret = std::remove(filename.c_str());
    if (ret != 0) {
      LOG(WARNING) << "Remove file " << filename << " failed";
    }
  }
}

}  // namespace runtime
}  // namespace tvm
