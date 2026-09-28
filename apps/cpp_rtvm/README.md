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


# Native Inference application for CPP Native

Native inference tool ```rtvm``` helps in deploying TVM compiled models from a standalone cpp environment.
Overall process starts from getting a model from a framework all the way up to running on target device using `rtvm` tool.


### Compile the model

Compilation step generates TVM compiler output artifacts which need to be taken to target device for deployment.
The inputs and params are need to be dump as numpy files in a folder, which will be need to run on device.

Below command will generate the same


```bash
python3  scripts/download_models.py
```

### Deployment Run

Now we will verify the deployment run of the compiled model using ```rtvm``` tool on target device.

We need to copy the artifacts and inputs folder under Android temp folder at ```/data/local/tmp/```

Also copy the cross compiled tool ```rtvm```, ```libtvm_runtime.so```, ```libtvm_ffi.so``` and the device
backend library (e.g. ```libtvm_runtime_opencl.so```) to ```data/local/tmp/```

```rtvm``` usage can be quired as below
```bash
Android:/data/local/tmp $ LD_LIBRARY_PATH=./ ./rtvm
Command line usage
--model        - The tvm artifacts(mod.so)
--device       - The target device to use {llvm, opencl, cpu, cuda, metal, rocm, vpi, oneapi}
--input        - Input folder with input Numpy files where file name need to match with TVM mod main() param
--output       - Output folder to dump the model output as numpy
--dump-meta    - Dump model meta information
--profile      - Profile over all execution
--dry-run      - Profile after given dry runs, default 10
--run-count    - Profile for given runs, default 50

  Example
  LD_LIBRARY_PATH=. ./rtvm --model=./opencl_vm_mod.so --input=./inputs --device=opencl --output=./out
```

# Performnace Profiling Options
The tool has added few options to measure wall clock performance of the given model on Target natively.
--profile : Can turn on the profiling
--dry-run : The number of times dry run the model before mearuring the performance. Default value os 10
--run-count : The number times to run the model and take an average. Default value is 50.

Performance profile options dumps information summary as given below.
     Module Load              :27 ms
     Graph Runtime Create     :11 ms
     Params Read              :15 ms
     Params Set               :41 ms
     Pre Compiled Progs Load  :24 ms
Total Load Time     :118 ms
Average ExecTime    :27 ms
Unload Time         :35.9236 ms
