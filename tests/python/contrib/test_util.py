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


def validate_debug_dir_path(temp_dir, expected_basename):
    dirname, basename = os.path.split(temp_dir.temp_dir)
    assert basename == expected_basename, "unexpected basename: %s" % (basename,)

    parent_dir = os.path.basename(dirname)
    create_time = datetime.datetime.strptime(parent_dir.split("___", 1)[0], "%Y-%m-%dT%H-%M-%S")
    assert abs(datetime.datetime.now() - create_time) < datetime.timedelta(seconds=60)


def test_tempdir():
    assert util.TempDirectory._KEEP_FOR_DEBUG == False, "don't submit with KEEP_FOR_DEBUG == True"

    temp_dir = util.tempdir()
    assert os.path.exists(temp_dir.temp_dir)

    old_debug_mode = util.TempDirectory._KEEP_FOR_DEBUG
    old_tempdirs = util.TempDirectory.TEMPDIRS
    try:
        for temp_dir_number in range(0, 3):
            with util.TempDirectory.set_keep_for_debug():
                debug_temp_dir = util.tempdir()
                try:
                    validate_debug_dir_path(debug_temp_dir, "0000" + str(temp_dir_number))
                finally:
                    shutil.rmtree(debug_temp_dir.temp_dir)

        with util.TempDirectory.set_keep_for_debug():
            # Create 2 temp_dir within the same session.
            debug_temp_dir = util.tempdir()
            try:
                validate_debug_dir_path(debug_temp_dir, "00003")
            finally:
                shutil.rmtree(debug_temp_dir.temp_dir)

            debug_temp_dir = util.tempdir()
            try:
                validate_debug_dir_path(debug_temp_dir, "00004")
            finally:
                shutil.rmtree(debug_temp_dir.temp_dir)

            with util.TempDirectory.set_keep_for_debug(False):
                debug_temp_dir = util.tempdir()  # This one should get deleted.

                # Simulate atexit hook
                util.TempDirectory.remove_tempdirs()

                # Calling twice should be a no-op.
                util.TempDirectory.remove_tempdirs()

                # Creating a new TempDirectory should fail now
                try:
                    util.tempdir()
                    assert False, "creation should fail"
                except util.DirectoryCreatedPastAtExit:
                    pass

    finally:
        util.TempDirectory.DEBUG_MODE = old_debug_mode
        util.TempDirectory.TEMPDIRS = old_tempdirs


if __name__ == "__main__":
    test_tempdir()
