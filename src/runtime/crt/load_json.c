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
 * \file saveload_json.cc
 * \brief Save and load graph to/from JSON file.
 */
#include "load_json.h"

// the node entry structure in serialized format
typedef struct _JSONNodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  void (*Load)(struct _JSONNodeEntry * entry, JSONReader *reader);
} JSONNodeEntry;

void JSONNodeEntryLoad(JSONNodeEntry * entry, JSONReader *reader) {
  reader->BeginArray(reader);
  if (reader->NextArrayItem(reader)) { fprintf(stderr, "invalid json format\n"); }
  reader->ReadUnsignedInteger(reader, &(entry->node_id));
  if (reader->NextArrayItem(reader)) { fprintf(stderr, "invalid json format\n"); }
  reader->ReadUnsignedInteger(reader, &(entry->index));
  if (reader->NextArrayItem(reader)) {
    reader->ReadUnsignedInteger(reader, &(entry->version));
    if (!reader->NextArrayItem(reader)) { fprintf(stderr, "invalid json format\n"); }
  } else {
    entry->version = 0;
  }
}
