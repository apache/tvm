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
This is done by instrumenting the code with profiling builtin calls using a TIR pass.
During codegen, these builtin calls are replaced with the calls to a hexagon specific
handler which records the runtime information into a buffer.
This buffer is written into a JSON file ('lwp.json') which is processed to construct
function and loop level profiling information as a csv file.

**Note:** During codegen, the profiling builtin calls are ignored for other targets.

The TIR pass offers several config flags to control the level of instrumentation
as mentioned below:

1) `lwp_disable_func_prof`: To disable function level profiling. By default, it is
set to 'False', i.e., the function level profiling is enabled.

2) `instr_siblings`: When enabled, only loops with siblings are instrumented and rest are
ignored. The inner-most loops are always excluded from instrumentation unless overwritten
using `lwp_min_height`. This is done to minimize the adverse effect of instrumentation on
actual performance. By default, it is set to 'True'.

3) `lwp_max_depth`: To instrument loops up to a certain depth. This flag is effective
only when `instr_siblings` is disabled. By default, it is set to 0.

4) `lwp_min_height`: To exclude inner loops up to a certain height from instrumentation.
By default, it is set to 1.

For additional usage information on various config flags, please refer to the tests in
`tests/python/unittest/test_tir_transform_profiling_instr.py`


## How to use lightweight profiling with RPC Launcher:

`tests/python/contrib/test_hexagon/test_launcher.py` contains two tests, `test_lwp` and
`test_lwp_multiple_conv2d`, to demonstrate lightweight profiling usage.

The steps involved are as follows:

1) While building a model, set `tir.instrument_lwp` to `True`.
   By default, the builtin calls will only be inserted for the loops with siblings. But it
   can be altered using LWP config options as described above.
2) Create `HexagonProfiler` object

```
with tvm.transform.PassContext(opt_level=3, config={"tir.instrument_lwp": True}):
    lowered = tvm.relay.build(
        relay_mod,
        tvm.target.Target(target_hexagon, host=target_hexagon),
        ...
    )

    # Create HexagonProfiler object. It sets the profiling mode based on the PassContext config.
    # '--hexagon-debug' to pytest can be used to retain any temp or test directories to
    # inspect the profiling data.
    profiler = HexagonProfiler(lowered, hexagon_server_process, hexagon_debug)
```

4) Run the model and get the profiling data as a CSV file. It is done by post-processing
   'lwp.json' file generated during runtime.

```
    graph_mod.run(**inputs)

    # Get lightweight profiling output as a CSV file
    profiler.get_profile_output(hexagon_launcher, hexagon_session, hexagon_server_process)
```
**Note:**

- For on-device runs, 'lwp.json' is copied into a temp directory along with the test .so and the processed
  CSV file
- For the simulator runs, the file is generated in the simulator test output directory. Test  .so
  will still be in a separate temp directory. lwp CSV file will also be in the same directory.

**Helpful Hints:**

- To prevent the test directories on the Hexagon device as well as temporary test directory on x86
from being deleted for profiling related runs, pass `--hexagon-debug` to pytest.

```
python -m pytest --hexagon-debug tests/python/contrib/test_hexagon/test_launcher.py::test_lwp
```
