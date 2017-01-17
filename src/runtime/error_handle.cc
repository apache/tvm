/*!
 *  Copyright (c) 2016 by Contributors
 * Implementation of error handling API
 * \file error_handle.cc
 */
#include <dmlc/thread_local.h>
#include <string>
#include "./runtime_base.h"

struct TVMErrorEntry {
  std::string last_error;
};

typedef dmlc::ThreadLocalStore<TVMErrorEntry> TVMAPIErrorStore;

const char *TVMGetLastError() {
  return TVMAPIErrorStore::Get()->last_error.c_str();
}

void TVMAPISetLastError(const char* msg) {
  TVMAPIErrorStore::Get()->last_error = msg;
}
