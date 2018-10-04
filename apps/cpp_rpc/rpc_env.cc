/*!
 *  Copyright (c) 2017 by Contributors
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
#include "../../src/common/util.h"
#include "../../src/runtime/file_util.h"

namespace tvm {
namespace runtime {

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string &dirname) {
  std::vector<std::string> vec;
  #ifndef _MSC_VER
    DIR *dp = opendir(dirname.c_str());
    if (dp == NULL) {
      int errsv = errno;
      LOG(FATAL) << "ListDir " << dirname <<" error: " << strerror(errsv);
    }
    dirent *d;
    while ((d = readdir(dp)) != NULL) {
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
  if (common::EndsWith(file, ".so")) {
      return Module::LoadFromFile(file, fmt);
  }

  #if defined(__linux__) || defined(__ANDROID__)
    std::string file_name = file + ".so";
    if (common::EndsWith(file, ".o")) {
      std::vector<std::string> files;
      files.push_back(file);
      CreateShared(file_name, files);
    } else if (common::EndsWith(file, ".tar")) {
      std::string tmp_dir = "rpc/tmp/";
      mkdir(&tmp_dir[0], 0777);
      std::string cmd = "tar -C " + tmp_dir + " -zxf " + file;
      CHECK(system(cmd.c_str()) == 0) << "Untar library error.";
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

}  // namespace runtime
}  // namespace tvm
