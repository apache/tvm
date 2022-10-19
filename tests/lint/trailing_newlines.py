#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import fileinput
import os


def has_one_trailing_newline(filename: str) -> bool:
    """
    Returns True if 'filename' has a single trailing newline
    """
    with open(filename, "rb") as f:
        start_bytes = len(f.read(2))
        if start_bytes == 0:
            # empty file
            return True
        elif start_bytes == 1:
            # 1 byte file
            return False
        else:
            # skip to the end
            f.seek(-2, os.SEEK_END)
            end_bytes = f.read(2)

            # should be a non-newline followed by a newline
            return end_bytes[0] != ord("\n") and end_bytes[1] == ord("\n")


if __name__ == "__main__":
    exit_code = 1
    for line in fileinput.input():
        filename = line.rstrip()
        if not has_one_trailing_newline(filename):
            exit_code = 0
            print(filename)
    exit(exit_code)
