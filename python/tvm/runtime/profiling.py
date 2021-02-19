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
"""Runtime profiling functions."""
import tvm._ffi


def start_timer(ctx):
    """
    Start a low-overhead device specific timer.

    Params
    ------
    ctx: TVMContext
        The context to get a timer for.

    Returns
    -------
    timer_start: function
        A function that stops the device specific timer. Calling this function
        stops the timer and returns another function that returns the elapsed
        time in nanoseconds. This third function should be called as late as
        possible to avoid synchronization overhead on GPUs.

    Example
    -------
    .. code-block:: python

        import tvm

        timer_stop = tvm.runtime.start_timer(tvm.cpu())
        x = 0
        for i in range(100000):
          x += 1
        elapsed = timer_stop()
        # do some more stuff
        print(elapsed())  # elapsed time in nanoseconds between timer_start and timer_stop
    """
    raise RuntimeError(
        "Profiling functions not loaded from runtime. Are you sure the runtime was built?"
    )


tvm._ffi._init_api("profiling", "tvm.runtime.profiling")
