Deploy Compiled Modules
=======================
NNVM compiled modules are fully embedded in TVM runtime as long as ```GRAPH_RUNTIME``` option
is enabled in tvm runtime. Check out the [TVM documentation](http://docs.tvmlang.org/) for
how to deploy TVM runtime to your system.

In a nutshell, we will need three items to deploy a compiled module.
Checkout our tutorials on getting started with NNVM compiler for more details.

- The graph json data which contains the execution graph.
- The tvm module library of compiled functions.
- The parameter blobs for stored parameters.

We can then use TVM's runtime API to deploy the compiled module.
Here is an example in python.

```python
import tvm

# tvm module for compiled functions.
loaded_lib = tvm.module.load("deploy.so")
# json graph
loaded_json = open(temp.relpath("deploy.json")).read()
# parameters in binary
loaded_params = bytearray(open(temp.relpath("deploy.params"), "rb").read())

fcreate = tvm.get_global_func("tvm.graph_runtime.create")
ctx = tvm.gpu(0)
gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]
set_input("x", tvm.nd.array(x_np))
gmodule["load_params"](loaded_params)
run()
out = tvm.nd.empty(shape)
get_output(0, out)
print(out.asnumpy())
```

An example in c++.
```cpp
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

int main()
{
    // tvm module for compiled functions
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("deploy.so");

    // json graph
    std::ifstream json_in("deploy.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in("deploy.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_dylib, 1, 0);

    int dtype_code = kFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kCPU;
    int device_id = 0;

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    // load image data saved in binary
    std::ifstream data_fin("cat.bin", std::ios::binary);
    data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* y;
    int out_ndim = 1;
    int64_t out_shape[1] = {1000, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    // get the maximum position in output vector
    auto y_iter = static_cast<float*>(y->data);
    auto max_iter = std::max_element(y_iter, y_iter + 1000);
    auto max_index = std::distance(y_iter, max_iter);
    std::cout << "The maximum position in output vector is: " << max_index << std::endl;

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
```
