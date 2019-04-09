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

# TVM Runtime Frontend Support

This crate provides an idiomatic Rust API for [TVM](https://github.com/dmlc/tvm) runtime frontend. Currently this requires **Nightly Rust** and tested on `rustc 1.32.0-nightly`

## What Does This Crate Offer?

Here is a major workflow

1. Train your **Deep Learning** model using any major framework such as [PyTorch](https://pytorch.org/), [Apache MXNet](https://mxnet.incubator.apache.org/) or [TensorFlow](https://www.tensorflow.org/)
2. Use **TVM** to build optimized model artifacts on a supported context such as CPU, GPU, OpenCL and specialized accelerators.
3. Deploy your models using **Rust** :heart:

### Example: Deploy Image Classification from Pretrained Resnet18 on ImageNet1k

Please checkout [examples/resnet](examples/resnet) for the complete end-to-end example.

Here's a Python snippet for downloading and building a pretrained Resnet18 via Apache MXNet and TVM

```python
block = get_model('resnet18_v1', pretrained=True)
    
sym, params = nnvm.frontend.from_mxnet(block)
# add the softmax layer for prediction
net = nnvm.sym.softmax(sym)
# compile the model
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)
# same the model artifacts
lib.save(os.path.join(target_dir, "deploy_lib.o"))
cc.create_shared(os.path.join(target_dir, "deploy_lib.so"),
                [os.path.join(target_dir, "deploy_lib.o")])

with open(os.path.join(target_dir, "deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(os.path.join(target_dir,"deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
```

Now, we need to input the artifacts to create and run the *Graph Runtime* to detect our input cat image

![cat](https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true)

as demostrated in the following Rust snippet

```rust
    let graph = fs::read_to_string("deploy_graph.json")?;
    // load the built module
    let lib = Module::load(&Path::new("deploy_lib.so"))?;
    // get the global TVM graph runtime function
    let runtime_create_fn = Function::get("tvm.graph_runtime.create", true).unwrap();
    let runtime_create_fn_ret = call_packed!(
        runtime_create_fn,
        &graph,
        &lib,
        &ctx.device_type,
        &ctx.device_id
    )?;
    // get graph runtime module
    let graph_runtime_module: Module = runtime_create_fn_ret.try_into()?;
    // get the registered `load_params` from runtime module
    let ref load_param_fn = graph_runtime_module
        .get_function("load_params", false)
        .unwrap();
    // parse parameters and convert to TVMByteArray
    let params: Vec<u8> = fs::read("deploy_param.params")?;
    let barr = TVMByteArray::from(&params);
    // load the parameters
    call_packed!(load_param_fn, &barr)?;
    // get the set_input function
    let ref set_input_fn = graph_runtime_module
        .get_function("set_input", false)
        .unwrap();

    call_packed!(set_input_fn, "data", &input)?;
    // get `run` function from runtime module
    let ref run_fn = graph_runtime_module.get_function("run", false).unwrap();
    // execute the run function. Note that it has no argument
    call_packed!(run_fn,)?;
    // prepare to get the output
    let output_shape = &mut [1, 1000];
    let output = empty(output_shape, TVMContext::cpu(0), TVMType::from("float32"));
    // get the `get_output` function from runtime module
    let ref get_output_fn = graph_runtime_module
        .get_function("get_output", false)
        .unwrap();
    // execute the get output function
    call_packed!(get_output_fn, &0, &output)?;
    // flatten the output as Vec<f32>
    let output = output.to_vec::<f32>()?;
```

and the model correctly predicts the input image as **tiger cat**.

## Installations

Please follow TVM [installations](https://docs.tvm.ai/install/index.html), `export TVM_HOME=/path/to/tvm` and add `libtvm_runtime` to your `LD_LIBRARY_PATH`.

*Note:* To run the end-to-end examples and tests, `tvm`, `nnvm` and `topi` need to be added to your `PYTHONPATH` or it's automatic via an Anaconda environment when it is installed individually.

## Supported TVM Functionalities

### Use TVM to Generate Shared Library

One can use the following Python snippet to generate `add_gpu.so` which add two vectors on GPU.

```python
import os
import tvm
from tvm.contrib import cc

def test_add(target_dir):
    if not tvm.module.enabled("cuda"):
        print(f"skip {__file__} because cuda is not enabled...")
        return
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    fadd_cuda = tvm.build(s, [A, B, C], "cuda", target_host="llvm", name="myadd")

    fadd_cuda.save(os.path.join(target_dir, "add_gpu.o"))
    fadd_cuda.imported_modules[0].save(os.path.join(target_dir, "add_gpu.ptx"))
    cc.create_shared(os.path.join(target_dir, "add_gpu.so"),
            [os.path.join(target_dir, "add_gpu.o")])


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
```

### Run the Generated Shared Library

The following code snippet demonstrates how to load and test the generated shared library (`add_gpu.so`) in Rust.

```rust
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = empty(shape, TVMContext::gpu(0), TVMType::from("float32"));
    arr.copy_from_buffer(data.as_mut_slice());
    let mut ret = empty(shape, TVMContext::gpu(0), TVMType::from("float32"));
    let mut fadd = Module::load(&Path::new("add_gpu.so")).unwrap();
    let fadd_dep = Module::load(&Path::new("add_gpu.ptx")).unwrap();
    assert!(fadd.enabled("gpu"));
    fadd.import_module(fadd_dep);
    fadd.entry();
    function::Builder::from(&mut fadd)
        .arg(&arr)
        .arg(&arr)
        .set_output(&mut ret)?
        .invoke()
        .unwrap();

    assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
}
```

**Note:** it is required to instruct the `rustc` to link to the generated `add_gpu.so` in runtime, for example by
`cargo:rustc-link-search=native=add_gpu`.

See the tests and examples custom `build.rs` for more details.

### Convert and Register a Rust Function as a TVM Packed Function

One can use `register_global_func!` macro to convert and register a Rust
function of type `fn(&[TVMArgValue]) -> Result<TVMRetValue>` to a global TVM **packed function** as follows

```rust
#[macro_use]
extern crate tvm_frontend as tvm;
use std::convert::TryInto;
use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = 0f32;
            let shape = &mut [2];
            for arg in args.iter() {
                let e = empty(shape, TVMContext::cpu(0), TVMType::from("float32"));
                let arg: NDArray = arg.try_into()?;
                let arr = arg.copy_to_ndarray(e).unwrap();
                let rnd: ArrayD<f32> = ArrayD::try_from(&arr).unwrap();
                ret += rnd.scalar_sum();
            }
            let ret_val = TVMRetValue::from(&ret);
            Ok(ret_val)
        }
    }

    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = empty(shape, TVMContext::cpu(0), TVMType::from("float32"));
    arr.copy_from_buffer(data.as_mut_slice());
    let mut registered = function::Builder::default();
    let ret: f64 = registered
        .get_function("sum", true)
        .arg(&arr)
        .arg(&arr)
        .invoke()
        .unwrap()
        .try_into()
        .unwrap();

    assert_eq!(ret, 14f64);
}
```
