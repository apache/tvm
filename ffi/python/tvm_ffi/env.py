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
# pylint: disable=invalid-name
"""Env handling."""
from . import core
from ._tensor import device


class DeviceStream:
    def __init__(self, stream):
        try:
            import torch

            if isinstance(stream, torch.cuda.Stream):
                torch_stream_context = torch.cuda.stream(stream)
                dev = device(str(stream.device))
                self.device_type = dev.dlpack_device_type()
                self.device_id = dev.index
                self.stream = stream.cuda_stream
                self.enter_callback = torch_stream_context.__enter__
                self.exit_callback = torch_stream_context.__exit__
            else:
                raise NotImplementedError
        except ImportError:
            pass

    def __enter__(self):
        self.enter_callback()
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.stream
        )

    def __exit__(self, *args):
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.prev_stream
        )
        self.exit_callback(*args)
        return False
