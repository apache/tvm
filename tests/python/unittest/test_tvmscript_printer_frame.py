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
from tvm.script.printer.frame import MetadataFrame


def test_frame_add_callback():
    frame = MetadataFrame()

    flag = 0

    def callback1():
        nonlocal flag
        flag += 1

    def callback2():
        nonlocal flag
        flag += 5

    frame.add_exit_callback(callback1)
    with frame:
        frame.add_exit_callback(callback2)
        assert flag == 0

    assert flag == 6


def test_frame_clear_callbacks_after_exit():
    frame = MetadataFrame()

    flag = 0

    def callback():
        nonlocal flag
        flag += 1

    frame.add_exit_callback(callback)

    with frame:
        pass

    assert flag == 1

    with frame:
        pass

    assert flag == 1
