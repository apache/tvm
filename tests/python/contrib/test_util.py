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
"""Tests for functions in tvm/python/tvm/contrib/util.py."""

import datetime
import os
import shutil
from tvm.contrib import util


def test_tempdir():
  assert util.TempDirectory.DEBUG_MODE == False, "don't submit with DEBUG_MODE == True"

  temp_dir = util.tempdir()
  assert os.path.exists(temp_dir.temp_dir)

  old_debug_mode = util.TempDirectory.DEBUG_MODE
  try:
    util.TempDirectory.DEBUG_MODE = True

    for temp_dir_number in range(0, 3):
      debug_temp_dir = util.tempdir()
      try:
        dirname, basename = os.path.split(debug_temp_dir.temp_dir)
        assert basename == ('0000' + str(temp_dir_number)), 'unexpected basename: %s' % (basename,)

        parent_dir = os.path.basename(dirname)
        create_time = datetime.datetime.strptime(parent_dir.split('___', 1)[0], '%Y-%m-%dT%H-%M-%S')
        assert abs(datetime.datetime.now() - create_time) < datetime.timedelta(seconds=60)

      finally:
        shutil.rmtree(debug_temp_dir.temp_dir)

  finally:
    util.TempDirectory.DEBUG_MODE = old_debug_mode


if __name__ == '__main__':
  test_tempdir()
