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

// LINT_C_FILE

/*!
 * \file load_json.c
 * \brief Load graph from JSON file.
 */
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/crt/internal/graph_executor/load_json.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/platform.h>

// the node entry structure in serialized format
typedef struct JSONNodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  void (*Load)(struct JSONNodeEntry* entry, JSONReader* reader);
} JSONNodeEntry;

void JSONNodeEntryLoad(JSONNodeEntry* entry, JSONReader* reader) {
  reader->BeginArray(reader);
  if (reader->NextArrayItem(reader)) {
    fprintf(stderr, "invalid json format\n");
  }
  reader->ReadUnsignedInteger(reader, &(entry->node_id));
  if (reader->NextArrayItem(reader)) {
    fprintf(stderr, "invalid json format\n");
  }
  reader->ReadUnsignedInteger(reader, &(entry->index));
  if (reader->NextArrayItem(reader)) {
    reader->ReadUnsignedInteger(reader, &(entry->version));
    if (!reader->NextArrayItem(reader)) {
      fprintf(stderr, "invalid json format\n");
    }
  } else {
    entry->version = 0;
  }
}

// implementation of Seq class

void SeqPush(Seq* seq, uint32_t src) {
  if (seq->size >= seq->allocated) {
    printf("seq too large.\n");
  }
  seq->data[seq->size] = src;
  seq->size += 1;
}

uint32_t* SeqBack(Seq* seq) {
  if (seq->size >= seq->allocated) {
    printf("seq too large.\n");
  }
  return seq->data + (seq->size - 1);
}

void SeqPop(Seq* seq) {
  if (seq->size >= seq->allocated) {
    printf("seq size is too large.\n");
  }
  if (seq->size == 0) {
    printf("seq size is too small.\n");
  }
  seq->size -= 1;
}

tvm_crt_error_t SeqCreate(uint64_t len, Seq** seq) {
  DLDevice dev = {kDLCPU, 0};
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(Seq), dev, (void**)seq);
  if (err != kTvmErrorNoError) {
    return err;
  }
  memset(*seq, 0, sizeof(Seq));
  (*seq)->allocated = len;

  err = TVMPlatformMemoryAllocate(sizeof(uint32_t) * len, dev, (void**)&(*seq)->data);
  if (err != kTvmErrorNoError) {
    return err;
  }
  (*seq)->push_back = SeqPush;
  (*seq)->back = SeqBack;
  (*seq)->pop_back = SeqPop;
  return err;
}

tvm_crt_error_t SeqRelease(Seq* seq) {
  DLDevice dev = {kDLCPU, 0};
  tvm_crt_error_t err = TVMPlatformMemoryFree(seq->data, dev);
  if (err != kTvmErrorNoError) {
    return err;
  }
  return TVMPlatformMemoryFree(seq, dev);
}

// implementations of JSONReader

/*!
 * \brief Takes the next char from the input source.
 * \return the next character.
 */
char JSONReader_NextChar(JSONReader* reader) {
  char ch = reader->isptr[0];
  reader->isptr += 1;
  return ch;
}

/*!
 * \brief Returns the next char from the input source.
 * \return the next character.
 */
char JSONReader_PeekNextChar(JSONReader* reader) { return reader->isptr[0]; }

/*!
 * \brief Read next nonspace character.
 * \return the next nonspace character.
 */
char JSONReader_NextNonSpace(JSONReader* reader) {
  int ch;
  do {
    ch = reader->NextChar(reader);
    if (ch == '\n') {
      ++(reader->line_count_n_);
    }
    if (ch == '\r') {
      ++(reader->line_count_r_);
    }
  } while (isspace(ch));
  return ch;
}

/*!
 * \brief Read just before next nonspace but not read that.
 * \return the next nonspace character.
 */
char JSONReader_PeekNextNonSpace(JSONReader* reader) {
  int ch;
  while (1) {
    ch = reader->PeekNextChar(reader);
    if (ch == '\n') {
      ++(reader->line_count_n_);
    }
    if (ch == '\r') {
      ++(reader->line_count_r_);
    }
    if (!isspace(ch)) break;
    reader->NextChar(reader);
  }
  return ch;
}

/*!
 * \brief Parse next JSON string.
 * \param out_str the output string. NULL to merely consume input and discard it.
 * \param out_str_size Number of bytes available to write starting from out_str. Includes
 *      terminating \0.
 * \throw tvm::Error when next token is not string
 */
int JSONReader_ReadString(JSONReader* reader, char* out_str, size_t out_str_size) {
  int status = 0;
  int ch = reader->NextNonSpace(reader);
  size_t output_counter = 0;
  while (output_counter < out_str_size || out_str == NULL) {
    ch = reader->NextChar(reader);
    if (ch == '\\') {
      char sch = reader->NextChar(reader);
      switch (sch) {
        case 'r':
          out_str[output_counter++] = '\r';
          break;
        case 'n':
          out_str[output_counter++] = '\n';
          break;
        case '\\':
          out_str[output_counter++] = '\\';
          break;
        case 't':
          out_str[output_counter++] = '\t';
          break;
        case '\"':
          out_str[output_counter++] = '\"';
          break;
        default:
          fprintf(stderr, "unknown string escape %c\n", sch);
          break;
      }
    } else {
      if (ch == '\"') {
        break;
      }
      if (out_str != NULL) {
        out_str[output_counter++] = ch;
      }
    }
    if (output_counter == out_str_size - 1) {
      fprintf(stderr, "Error: string size greater than buffer size (%zu).\n", out_str_size);
      break;
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      fprintf(stderr, "Error at line %zu, Expect \'\"\' but reach end of line\n",
              reader->line_count_n_);
      break;
    }
  }

  if (out_str != NULL) {
    out_str[output_counter] = 0;
  }
  return status;
}

int JSONReader_ReadUnsignedInteger(JSONReader* reader, unsigned int* out_value) {
  int status = 0;
  char* endptr;
  const char* icstr = reader->isptr;
  unsigned int number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}

int JSONReader_ReadInteger(JSONReader* reader, int64_t* out_value) {
  int status = 0;
  char* endptr;
  const char* icstr = reader->isptr;
  int64_t number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}

/*!
 * \brief Begin parsing an object.
 * \code
 *  string key;
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginObject();
 *  while (reader->NextObjectItem(&key)) {
 *    // do somthing to key value
 *    reader->Read(&value);
 *  }
 * \endcode
 */
void JSONReader_BeginObject(JSONReader* reader) {
  int ch = reader->NextNonSpace(reader);
  if (!(ch == '{')) {
    fprintf(stderr, "Error at line %zu, Expect \'{\' but got \'%c\'\n", reader->line_count_n_, ch);
  }
  Seq* scope_counter_ = reader->scope_counter_;
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to move to next object item.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \param out_key the key to the next object.
 * \param out_key_size number of bytes available to write at out_key, including terminating \0.
 * \return true if the read is successful, false if we are at end of the object.
 */
uint8_t JSONReader_NextObjectItem(JSONReader* reader, char* out_key, size_t out_key_size) {
  uint8_t next = 1;
  Seq* scope_counter_ = reader->scope_counter_;
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = 0;
    } else if (ch == '}') {
      next = 0;
    } else {
      if (ch != ',') {
        fprintf(stderr, "Error at line %zu, JSON object expect \'}\' or \',\' but got \'%c\'\n",
                reader->line_count_n_, ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == '}') {
      reader->NextChar(reader);
      next = 0;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return 0;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    int err = reader->ReadString(reader, out_key, out_key_size);
    if (err != 0) {
      fprintf(stderr, "error reading key");
      return 0;
    }
    int ch = reader->NextNonSpace(reader);
    if (ch != ':') {
      fprintf(stderr, "Error at line %zu, Expect \':\' but get \'%c\'\n", reader->line_count_n_,
              ch);
    }
    return 1;
  }
}

/*!
 * \brief Begin parsing an array.
 * \code
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginArray();
 *  while (reader->NextArrayItem(&value)) {
 *    // do somthing to value
 *  }
 * \endcode
 */
void JSONReader_BeginArray(JSONReader* reader) {
  int ch = reader->NextNonSpace(reader);
  if (ch != '[') {
    fprintf(stderr, "Error at line %zu, Expect \'[\' but get \'%c\'\n", reader->line_count_n_, ch);
  }
  Seq* scope_counter_ = reader->scope_counter_;
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to read the next element in the array.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \return true if the read is successful, false if we are at end of the array.
 */
uint8_t JSONReader_NextArrayItem(JSONReader* reader) {
  uint8_t next = 1;
  Seq* scope_counter_ = reader->scope_counter_;
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = 0;
    } else if (ch == ']') {
      next = 0;
    } else {
      if (ch != ',') {
        fprintf(stderr, "Error at line %zu, JSON object expect \']\' or \',\' but got \'%c\'\n",
                reader->line_count_n_, ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == ']') {
      reader->NextChar(reader);
      next = 0;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return 0;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    return 1;
  }
}

/*!
 * \brief Determine the remaining length of the array to read.
 * \param num_elements Pointer which receives the length.
 * \return 0 if successful
 */
int JSONReader_ArrayLength(JSONReader* reader, size_t* num_elements) {
  int status = 0;
  char* old_isptr = reader->isptr;
  size_t old_line_count_r_ = reader->line_count_r_;
  size_t old_line_count_n_ = reader->line_count_n_;
  int old_scope_counter_back = *reader->scope_counter_->back(reader->scope_counter_);

  typedef enum { kObject, kArray } item_type_t;
  Seq* scopes;
  tvm_crt_error_t err = SeqCreate(10, &scopes);
  if (err != kTvmErrorNoError) {
    return -1;
  }
  item_type_t json_item_type = kArray;
  *num_elements = 0;
  for (;;) {
    int has_item = 0;
    if (json_item_type == kArray) {
      has_item = reader->NextArrayItem(reader);
      if (scopes->size == 0 && has_item != 0) {
        (*num_elements)++;
      }
    } else if (json_item_type == kObject) {
      has_item = reader->NextObjectItem(reader, NULL, 0);
    } else {
      status = -1;
      break;
    }

    if (has_item) {
      char c = reader->PeekNextNonSpace(reader);
      if (c == '"') {
        reader->ReadString(reader, NULL, 1024);
      } else if (c == '[') {
        reader->BeginArray(reader);
        scopes->push_back(scopes, json_item_type);
        json_item_type = kArray;
      } else if (c == '{') {
        reader->BeginObject(reader);
        scopes->push_back(scopes, json_item_type);
        json_item_type = kObject;
      } else {
        int64_t val;
        reader->ReadInteger(reader, &val);
      }
    } else {
      if (scopes->size > 0) {
        json_item_type = *scopes->back(scopes);
        scopes->pop_back(scopes);
      } else {
        break;
      }
    }
  }

  reader->isptr = old_isptr;
  reader->line_count_r_ = old_line_count_r_;
  reader->line_count_n_ = old_line_count_n_;
  reader->scope_counter_->push_back(reader->scope_counter_, old_scope_counter_back);

  err = SeqRelease(scopes);
  if (err != kTvmErrorNoError) {
    return -1;
  }

  return status;
}

/*!
 * \brief Constructor.
 * \param is the input source.
 */
tvm_crt_error_t JSONReader_Create(const char* is, JSONReader* reader) {
  memset(reader, 0, sizeof(JSONReader));
  tvm_crt_error_t err = SeqCreate(200, &reader->scope_counter_);
  if (err != kTvmErrorNoError) {
    return err;
  }
  reader->NextChar = JSONReader_NextChar;
  reader->PeekNextChar = JSONReader_PeekNextChar;
  reader->NextNonSpace = JSONReader_NextNonSpace;
  reader->PeekNextNonSpace = JSONReader_PeekNextNonSpace;
  reader->ReadString = JSONReader_ReadString;
  reader->ReadUnsignedInteger = JSONReader_ReadUnsignedInteger;
  reader->ReadInteger = JSONReader_ReadInteger;
  reader->BeginArray = JSONReader_BeginArray;
  reader->BeginObject = JSONReader_BeginObject;
  reader->NextArrayItem = JSONReader_NextArrayItem;
  reader->NextObjectItem = JSONReader_NextObjectItem;
  reader->ArrayLength = JSONReader_ArrayLength;

  DLDevice dev = {kDLCPU, 0};
  err = TVMPlatformMemoryAllocate(strlen(is) + 1, dev, (void**)&reader->is_);
  if (err != kTvmErrorNoError) {
    return err;
  }

  memset(reader->is_, 0, strlen(is) + 1);
  snprintf(reader->is_, strlen(is) + 1, "%s", is);
  reader->isptr = reader->is_;
  return err;
}

tvm_crt_error_t JSONReader_Release(JSONReader* reader) {
  tvm_crt_error_t err = SeqRelease(reader->scope_counter_);
  if (err != kTvmErrorNoError) {
    return err;
  }

  DLDevice dev = {kDLCPU, 0};
  return TVMPlatformMemoryFree(reader->is_, dev);
}
