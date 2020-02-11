#ifndef COMMON_H_
#define COMMON_H_

// #include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#ifndef __cplusplus

/* typedef unsigned int size_t; */
typedef unsigned char bool;
static const unsigned char false = 0;
static const unsigned char true = 1;

#define nullptr ((void*)0)

#endif // __cplusplus

#define TVM_STATUS_SUCCESS (0)
#define TVM_STATUS_FAILURE (-1)

#define API_BEGIN() int status; do { status = TVM_STATUS_SUCCESS; } while(false)
#define API_END() return status

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#ifndef __func__
#define __func__ __FUNCTION__
#endif
#define LOGE(fmt,...)                                               \
  do {                                                              \
    char msgbuf[1024];sprintf(msgbuf,"ERROR: " fmt,##__VA_ARGS__);  \
    MessageBoxA(NULL,msgbuf,"ERROR",MB_ICONERROR|MB_OK);            \
  }while(0)
#define LOGW(fmt,...)                                                   \
  do {                                                                  \
    char msgbuf[1024];sprintf(msgbuf,"WARNING: " fmt,##__VA_ARGS__);    \
    MessageBoxA(NULL,msgbuf,"WARNING",MB_ICONWARNING|MB_OK);            \
  }while(0)
#define LOGI(fmt,...)                                               \
  do {                                                              \
    char msgbuf[1024];sprintf(msgbuf,"INFO: " fmt,##__VA_ARGS__);   \
    MessageBoxA(NULL,msgbuf,"INFO",MB_ICONINFORMATION|MB_OK);       \
  }while(0)
#elif defined(ANDROID)
#define LOG_TAG "TVMCRT"
#define LOGE(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,      \
                        "ERROR: " fmt,##__VA_ARGS__);   \
  }while(0)
#define LOGW(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_WARN,LOG_TAG,       \
                        "WARNING: " fmt,##__VA_ARGS__); \
  }while(0)
#define LOGI(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_INFO,LOG_TAG,       \
                        "INFO: " fmt,##__VA_ARGS__);    \
  }while(0)
#elif defined(__linux__) || defined(__APPLE__)
#define LOGE(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"%s:%d: error: " fmt "\n",__FILE__,__LINE__,##__VA_ARGS__); \
    exit(-1);                                                                  \
  }while(0)
#define LOGW(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"%s:%d: warning: " fmt "\n",__FILE__,__LINE__,##__VA_ARGS__); \
  }while(0)
#define LOGI(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"%s:%d: info: " fmt "\n",__FILE__,__LINE__,##__VA_ARGS__); \
  }while(0)
#else 
#endif

#define Printf printf

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif // MAX

static inline void NoOperation() {}
typedef void (*Function)();

static inline void Shape_Print(char * str, int64_t * shape, uint32_t ndim) {
  uint32_t idx;
  char tmp[10];
  for (idx = 0; idx < ndim; idx++) {
    if (idx != (ndim-1)) {
      sprintf(tmp, "%dx", (int)shape[idx]);
    } else {
      sprintf(tmp, "%d", (int)shape[idx]);
    }
    strcat(str, tmp);
  }
}

static inline uint32_t Shape_CountNonZero(int64_t * shape){
  uint32_t ndim;
  for (ndim = 0; ndim < TVM_CRT_MAX_NDIM; ndim++) {
    if (shape[ndim] == 0) {
      break;
    }
  }
  return ndim;
}

static inline uint32_t Shape_Accumulate(int64_t * shape){
  int64_t accum = 1;
  uint32_t ndim;
  for (ndim = 0; ndim < TVM_CRT_MAX_NDIM; ndim++) {
    if (shape[ndim] == 0) { break; }
    accum *= shape[ndim];
  }
  return accum;
}

//// seq class

typedef struct seq_t {
  uint32_t data[500];
  uint32_t size;
  void (*push_back)(struct seq_t * seq, uint32_t src);
  uint32_t * (*back)(struct seq_t * seq);
  void (*pop_back)(struct seq_t * seq);
} Seq;

static inline void SeqPush(Seq * seq, uint32_t src) {
  if (seq->size >= 500) {
    LOGE("seq too large.");
  }
  seq->data[seq->size] = src;
  seq->size += 1;
}

static inline uint32_t * SeqBack(Seq * seq) {
  if (seq->size >= 500) {
    LOGE("seq too large.");
  }
  return seq->data + (seq->size-1);
}

static inline void SeqPop(Seq * seq) {
  if (seq->size >= 500) {
    Printf("seq size is too large.\n");
  }
  if (seq->size == 0) {
    Printf("seq size is too small.\n");
  }
  seq->size -= 1;
}

static inline Seq * SeqCreate() {
  Seq * seq = (Seq*)malloc(sizeof(Seq));
  memset(seq, 0, sizeof(Seq));
  seq->push_back = SeqPush;
  seq->back = SeqBack;
  seq->pop_back = SeqPop;
  return seq;
}

#endif // COMMON_H_
