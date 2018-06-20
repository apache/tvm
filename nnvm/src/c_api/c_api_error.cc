/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include <dmlc/thread_local.h>
#include "./c_api_common.h"

struct ErrorEntry {
  std::string last_error;
};

typedef dmlc::ThreadLocalStore<ErrorEntry> NNAPIErrorStore;

const char *NNGetLastError() {
  return NNAPIErrorStore::Get()->last_error.c_str();
}

void NNAPISetLastError(const char* msg) {
  NNAPIErrorStore::Get()->last_error = msg;
}
