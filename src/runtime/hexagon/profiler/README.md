<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Hexagon lightweight instrumentation based profiling (LWP)

For Hexagon, LWP can be used to get function and loop level processor cycle count.
This's done by instrumenting the code with profiling builtin calls using a TIR pass.
During codegen, these builtin calls are replaced with the calls to a hexagon specific
handler which records the runtime information into a buffer.
This buffer is written into a JSON file ('lwp.json') which is processed to construct
function and loop level profiling information as a csv file.

**Note:** During codegen, the profiling builtin calls are ignored for other targets.

The TIR pass offers several config flags to control the level of instrumentation
as mentioned below:

1) `lwp_disable_func_prof`: To disable function level profiling. By default, it's
set to 'False', i.e., the function level profiling is enabled.

2) `instr_siblings`: When enabled, only loops with siblings are instrumented and rest are
ignored. The inner-most loops are always excluded from instrumentation unless overwritten
using `lwp_min_height`. This is done to minimize the adverse effect of instrumentation on
actual performance. By default, it's set to 'True'.

3) `lwp_max_depth`: To instrument loops upto a certain depth. This flag is effective
only when `instr_siblings` is disabled. By default, it's set to 0.

4) `lwp_min_height`: To exclude inner loops upto a certain height from instrumentation.
By default, it's set to 1.

For additional usage information on various config flags, please refer to the tests in
`tests/python/unittest/test_tir_transform_profiling_instr.py`


## How to use lightweight profiling with RPC Launcher:

`tests/python/contrib/test_hexagon/test_launcher.py` contains two tests, `test_lwp` and
`test_lwp_multiple_conv2d`, to demonstrate lightweight profiling usage.

The steps involved are as follows:

1) While building a model, set `tir.instrument_lwp` to `True`.
   By default, the builtin calls will only be inserted for the loops with siblings. But, it 
   can be altered using LWP config options as described above.
2) Save the binary file as it'll be needed to process the profiling data (lwp.json) later.
3) Create `HexagonProfiler` object. It's passed to `get_profile_output` to check if the model was
built with profiling enabled before copying the data from the device.

```
with tvm.transform.PassContext(opt_level=3, config={"tir.instrument_lwp": True}):
    lowered = tvm.relay.build(
        relay_mod,
        tvm.target.Target(target_hexagon, host=target_hexagon),
        ...
    )

    # Save binary file to post-process lwp output
    lowered.get_lib().save(dso_binary_path)

    # Create HexagonProfiler object. It sets the profiling mode based on the PassContext config.
    profiler = HexagonProfiler()
```

4) Run the model and get profile data (`lwp.json`) from the device (or the simulator):

**Note:**

- For on-device runs, 'lwp.json' is genrated in the same remote directory where 'tvm_rpc_android'
is copied. This remote path is needed to copy the file from the device and can be found in
'hexagon_server_process["launcher"].workspace'.

- For the simulator runs, the remote path is not needed as the 'lwp.json' file is generated in the
simulator test output directory.

```
    remote_path = ""
    if android_serial_number is not None and android_serial_number != "simulator":
        # Get the workspace on the device to extract lwp output
        remote_path = hexagon_server_process["launcher"]._workspace

    # Get profile data (lwp.json) from the device
    prof_out = hexagon_launcher.get_profile_output(profiler, hexagon_session, remote_path, temp)

```

5) Process `lwp.json` and construct an easy-to-read csv file.

This step requires several parameters as explained below:

- Path of the binary file
- android_serial_number
- Path of the lwp json file (lwp.json) which gets created in the current directory
- Path to the run log depending on the environment:
  - For on-device runs:
     Use logcat output as the run log
     To get the logcat output:
     - Create /vendor/lib/rfsa/adsp/tvm_rpc_android.farf on the device
     - Run logcat command in the background or in a separate terminal while
       running the test:
       adb -s <device-id> logcat -c && adb -s <device-id> logcat 2>&1 | tee /tmp//logcat.log
  - For simulator runs:
     Use "stdout.txt" as the run log. There is no need to specify the full path to
     "stdout.txt" as it will be inferred based on 'prof_out' location.
- lwp processed output file -  "lwp.csv"

**Note:** For on-device run, the logcat output needs to be collected manually and its path
must be passed to 'process_lwp_output' as mentioned above.

```
    lwp_csv = temp.relpath("lwp.csv")
    if android_serial_number == "simulator":
        process_lwp_output(dso_binary_path, android_serial_number, prof_out, "stdout.txt", lwp_csv)
    else:
        # For on-device run
        if os.path.exists("/tmp/logcat.log"):
            process_lwp_output(
                dso_binary_path, android_serial_number, prof_out, "/tmp/logcat.log", lwp_csv
            )
        else:
            print("WARNING: Error processing lwp output - missing logcat file")
```

**Helpful Hints:**

1) The above code snippet generates 'lwp.csv' in the temporary directory which gets deleted when the
test exits. To keep the temp directory, set `keep_for_debug` to `True` while creating it. Alternatively,
you can set `lwp_csv` to "/tmp/lwp.csv".

```
temp = utils.tempdir(keep_for_debug=True)
```

2) To prevent the test directories on the Hexagon device from being deleted, pass `--hexagon-debug` to pytest.

```
python -m pytest --hexagon-debug tests/python/contrib/test_hexagon/test_launcher.py::test_lwp
```
