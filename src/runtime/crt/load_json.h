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
 * \file load_json.h
 * \brief Lightweight JSON Reader that read save into C++ data structs.
 */
#ifndef TVM_RUNTIME_CRT_LOAD_JSON_H_
#define TVM_RUNTIME_CRT_LOAD_JSON_H_

#include <ctype.h>
#include <stdio.h>

enum {
  JSON_READ_TYPE_U8 = 1,
  JSON_READ_TYPE_S8 = 2,
  JSON_READ_TYPE_U16 = 3,
  JSON_READ_TYPE_S16 = 4,
  JSON_READ_TYPE_U32 = 5,
  JSON_READ_TYPE_S32 = 6,
  JSON_READ_TYPE_F32 = 7,
  JSON_READ_TYPE_F64 = 8,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE = 9,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE_ENTRY = 10,
  JSON_READ_TYPE_GRAPH_RUNTIME_GRAPH_ATTR = 11
};

typedef struct Seq {
  uint32_t* data;
  uint64_t allocated;
  uint32_t size;
  void (*push_back)(struct Seq* seq, uint32_t src);
  uint32_t* (*back)(struct Seq* seq);
  void (*pop_back)(struct Seq* seq);
} Seq;

/*!
 * \brief Lightweight JSON Reader to read any STL compositions and structs.
 *  The user need to know the schema of the
 */
typedef struct JSONReader {
  /*! \brief internal reader string */
  char* is_;
  char* isptr;
  /*! \brief "\\r" counter */
  size_t line_count_r_;
  /*! \brief "\\n" counter */
  size_t line_count_n_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  Seq* scope_counter_;

  char (*NextChar)(struct JSONReader* reader);
  char (*NextNonSpace)(struct JSONReader* reader);
  char (*PeekNextChar)(struct JSONReader* reader);
  char (*PeekNextNonSpace)(struct JSONReader* reader);
  int (*ReadUnsignedInteger)(struct JSONReader* reader, unsigned int* out_value);
  int (*ReadInteger)(struct JSONReader* reader, int64_t* out_value);
  int (*ReadString)(struct JSONReader* reader, char* out_value);
  void (*BeginArray)(struct JSONReader* reader);
  void (*BeginObject)(struct JSONReader* reader);
  uint8_t (*NextObjectItem)(struct JSONReader* reader, char* out_key);
  uint8_t (*NextArrayItem)(struct JSONReader* reader);
} JSONReader;

/*!
 * \brief Constructor of JSONReader class
 * \param is the input source.
 */
JSONReader JSONReader_Create(const char* is);

void JSONReader_Release(JSONReader* reader);

#endif  // TVM_RUNTIME_CRT_LOAD_JSON_H_
