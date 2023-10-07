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
"""Internal DiscoWorker for Disco ProcessSession."""
import os
import sys

from tvm import runtime as _  # pylint: disable=unused-import
from tvm._ffi import get_global_func, register_func
from tvm.runtime import NDArray, ShapeTuple, String
from tvm.runtime.ndarray import array


@register_func("tests.disco.add_one")
def _add_one(x: int) -> int:  # pylint: disable=invalid-name
    return x + 1


@register_func("tests.disco.add_one_float", override=True)
def _add_one_float(x: float):  # pylint: disable=invalid-name
    return x + 0.5


@register_func("tests.disco.add_one_ndarray", override=True)
def _add_one_ndarray(x: NDArray) -> NDArray:  # pylint: disable=invalid-name
    return array(x.numpy() + 1)


@register_func("tests.disco.str", override=True)
def _str_func(x: str):  # pylint: disable=invalid-name
    return x + "_suffix"


@register_func("tests.disco.str_obj", override=True)
def _str_obj_func(x: String):  # pylint: disable=invalid-name
    assert isinstance(x, String)
    return String(x + "_suffix")


@register_func("tests.disco.shape_tuple", override=True)
def _shape_tuple_func(x: ShapeTuple):  # pylint: disable=invalid-name
    assert isinstance(x, ShapeTuple)
    return ShapeTuple(list(x) + [4, 5])


def main():
    """Main worker function"""
    if len(sys.argv) != 5:
        print("Usage: <worker_id> <num_workers> <read_fd> <write_fd>")
        return
    worker_id = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        reader = msvcrt.open_osfhandle(int(sys.argv[3]), os.O_BINARY)
        writer = msvcrt.open_osfhandle(int(sys.argv[4]), os.O_BINARY)
    else:
        reader = int(sys.argv[3])
        writer = int(sys.argv[4])

    worker_func = get_global_func("runtime.disco.WorkerProcess")
    worker_func(worker_id, num_workers, reader, writer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
