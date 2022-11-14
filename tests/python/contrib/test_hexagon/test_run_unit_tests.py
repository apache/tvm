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

""" capture gtest output and return over FFI """

import tvm
from tvm.contrib.hexagon.session import Session


# use pytest -sv to observe gtest output
# use --gtest_args to pass arguments to gtest
# for example to run all "foo" tests twice and observe gtest output run
# pytest -sv <this file> --gtests_args="--gtest_filter=*foo* --gtest_repeat=2"
@tvm.testing.requires_hexagon
def test_run_unit_tests(hexagon_session: Session, gtest_args):
    """Try running gtest unit tests and capture output and error code"""
    try:
        func = hexagon_session._rpc.get_function("hexagon.run_unit_tests")
    except:
        print(
            (
                "This test requires TVM Runtime to be built with a Hexagon gtest"
                "version using Hexagon API cmake flag"
                "-DUSE_HEXAGON_GTEST=/path/to/hexagon/sdk/utils/googletest/gtest"
            )
        )
        raise

    gtest_error_code_and_output = func(gtest_args)
    gtest_error_code = int(gtest_error_code_and_output.splitlines()[0])
    gtest_output = gtest_error_code_and_output.split("\n", 1)[-1]
    print(gtest_output)
    if gtest_error_code != 0:
        raise RuntimeError(
            f"Hexagon gtest retruned non-zero error code = {gtest_error_code}:\n{gtest_output}"
        )


if __name__ == "__main__":
    tvm.testing.main()
