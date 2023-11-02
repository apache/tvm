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
from tvm.contrib import utils


def validate_debug_dir_path(temp_dir, expected_basename):
    """Validate the dir path of debugging"""
    dirname, basename = os.path.split(temp_dir.temp_dir)
    assert basename == expected_basename, "unexpected basename: %s" % (basename,)

    parent_dir = os.path.basename(dirname)
    create_time = datetime.datetime.strptime(parent_dir.split("___", 1)[0], "%Y-%m-%dT%H-%M-%S")
    assert abs(datetime.datetime.now() - create_time) < datetime.timedelta(seconds=60)


def test_tempdir():
    """Tests for temporary dir"""
    assert utils.TempDirectory._KEEP_FOR_DEBUG is False, "don't submit with KEEP_FOR_DEBUG == True"

    temp_dir = utils.tempdir()
    assert os.path.exists(temp_dir.temp_dir)

    old_debug_mode = utils.TempDirectory._KEEP_FOR_DEBUG
    old_tempdirs = utils.TempDirectory.TEMPDIRS
    try:
        for temp_dir_number in range(0, 3):
            with utils.TempDirectory.set_keep_for_debug():
                debug_temp_dir = utils.tempdir()
                try:
                    validate_debug_dir_path(debug_temp_dir, "0000" + str(temp_dir_number))
                finally:
                    shutil.rmtree(debug_temp_dir.temp_dir)

        with utils.TempDirectory.set_keep_for_debug():
            # Create 2 temp_dir within the same session.
            debug_temp_dir = utils.tempdir()
            try:
                validate_debug_dir_path(debug_temp_dir, "00003")
            finally:
                shutil.rmtree(debug_temp_dir.temp_dir)

            debug_temp_dir = utils.tempdir()
            try:
                validate_debug_dir_path(debug_temp_dir, "00004")
            finally:
                shutil.rmtree(debug_temp_dir.temp_dir)

            with utils.TempDirectory.set_keep_for_debug(False):
                debug_temp_dir = utils.tempdir()  # This one should get deleted.

                # Simulate atexit hook
                utils.TempDirectory.remove_tempdirs()

                # Calling twice should be a no-op.
                utils.TempDirectory.remove_tempdirs()

                # Creating a new TempDirectory should fail now
                try:
                    utils.tempdir()
                    assert False, "creation should fail"
                except utils.DirectoryCreatedPastAtExit:
                    pass

    finally:
        utils.TempDirectory.DEBUG_MODE = old_debug_mode
        utils.TempDirectory.TEMPDIRS = old_tempdirs


if __name__ == "__main__":
    test_tempdir()
