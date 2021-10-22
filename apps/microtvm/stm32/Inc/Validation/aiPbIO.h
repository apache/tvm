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
 * \file aiPbIO.h
 * \brief Low Level nano PB stack functions
 */

#ifndef _AI_PB_IO_H_
#define _AI_PB_IO_H_

#include <pb.h>

int pb_io_stream_init(void);

void pb_io_flush_ostream(void);
void pb_io_flush_istream(void);

pb_ostream_t pb_io_ostream(int fd);
pb_istream_t pb_io_istream(int fd);

#endif /* _AI_PB_IO_H_ */

