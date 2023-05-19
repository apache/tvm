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
# pylint: disable=consider-using-with

"""Define HexagonProfiler class to enable profiling for Hexagon"""

import os
import subprocess
import typing
from tvm.ir.transform import PassContext
from tvm.contrib.hexagon.profiling.process_lwp_data import process_lwp_output
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm.driver.build_module import OperatorModule
from tvm.contrib import utils


class HexagonProfiler:
    """Hexagon Profiler"""

    def __init__(
        self,
        dso_binary: str,
        module: typing.Union[ExecutorFactoryModule, OperatorModule],
        hexagon_server_process,
        enable_debug,
    ):
        """Configure HexagonProfiler"""
        # Save test .so to process profiling data
        self._temp_dir = utils.tempdir(keep_for_debug=enable_debug)
        self._dso_binary_path = self._temp_dir.relpath(dso_binary)
        if isinstance(module, OperatorModule):
            module.save(self._dso_binary_path)
        else:
            module.get_lib().save(self._dso_binary_path)

        self._android_serial_number = os.environ.get("ANDROID_SERIAL_NUMBER")
        self._remote_path = ""
        self._logcat_path = ""

        self._profiling_mode = None
        config = PassContext.current().config
        if self._android_serial_number is None:
            raise RuntimeError("ANDROID_SERIAL_NUMBER must be set for profiling")

        if ("tir.instrument_lwp", True) in config.items():
            # Set profiling mode
            self._profiling_mode = "lwp"

            if self._android_serial_number != "simulator":
                # Clear the logcat buffer and create a child process to redirect logcat output
                # into a file.
                launcher = hexagon_server_process["launcher"]
                subprocess.check_call(launcher._adb_device_sub_cmd + ["logcat", "-c"])
                self._logcat_path = self._temp_dir.relpath("logcat.log")
                self._fo = open(self._logcat_path, "w")
                self._proc = subprocess.Popen(
                    launcher._adb_device_sub_cmd + ["logcat"], stdout=self._fo
                )

                # Get the remote workspace on the device from where the lwp data needs to be copied.
                self._remote_path = launcher._workspace

        if self._profiling_mode is None:
            raise RuntimeError("Profiling mode was not set or was not a valid one.")

    def get_mode(self):
        return self._profiling_mode

    def is_lwp_enabled(self):
        return self._profiling_mode == "lwp"

    def get_temp_dir(self):
        return self._temp_dir

    def get_remote_path(self):
        return self._remote_path

    def get_profile_output(self, hexagon_launcher, hexagon_session):
        """Get runtime profiling data"""
        prof_out = hexagon_launcher.get_profile_output(self, hexagon_session)

        print("lwp json can be found at -- ", prof_out)

        # Process lightweight profiling output into an easily readable csv file
        # The post-processing requires following parameters:
        # 1) Path of the binary file
        # 2) android_serial_number
        # 3) Path of the lwp json file (lwp.json) which gets created in the current directory
        # 4) Path to the run log depending on the environment:
        #    a) For on-device runs:
        #       Use logcat output as the run log
        #    b) For simulator runs:
        #       Use "stdout.txt" as the run log. There is no need to specify the full path to
        #       "stdout.txt" as it will be inferred based on 'prof_out' location.
        # 5) lwp processed output file -  "lwp.csv"
        #
        lwp_csv = self._temp_dir.relpath("lwp.csv")
        if self._android_serial_number == "simulator":
            process_lwp_output(
                self._dso_binary_path, self._android_serial_number, prof_out, "stdout.txt", lwp_csv
            )
        else:
            # For on-device run
            self._proc.kill()  # End the child process for logcat
            self._fo.close()
            if os.path.exists(self._logcat_path):
                process_lwp_output(
                    self._dso_binary_path,
                    self._android_serial_number,
                    prof_out,
                    self._logcat_path,
                    lwp_csv,
                )
            else:
                raise RuntimeError("Error processing lwp output - missing logcat file")
