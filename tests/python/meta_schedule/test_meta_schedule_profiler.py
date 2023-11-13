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
""" Test Meta Schedule Profiler """
import time

from tvm import meta_schedule as ms


def test_meta_schedule_profiler_context_manager():
    with ms.Profiler() as profiler:
        time.sleep(1)
        with ms.Profiler.timeit("Level0"):
            time.sleep(1)
            with ms.Profiler.timeit("Level1"):
                time.sleep(2)
    # Note that the results are in seconds

    result = profiler.get()
    assert len(result) == 3
    assert 3.9 <= result["Total"] <= 4.1
    assert 2.9 <= result["Level0"] <= 3.1
    assert 1.9 <= result["Level1"] <= 2.1


def test_meta_schedule_no_context():
    with ms.Profiler.timeit("Level0"):
        assert ms.Profiler.current() is None


if __name__ == "__main__":
    test_meta_schedule_profiler_context_manager()
    test_meta_schedule_no_context()
