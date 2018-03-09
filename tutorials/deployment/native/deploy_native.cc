/*!
 * Native Deploy utility on TVM runtime.
 *
 * Simple utility which takes below args and run the model on target.
 *
 * arg1 : Model Base Name
 *        Expects files with <basename>.so , <basename>.json and <basename>.params
 *
 * arg2 : inputnode
 *        Name of the input node in graph.
 *        Also expects a file <inputnode>.npy which is a numpy saved file.
 *
 * arg3 : Output.npy (Output file name to dump the output)
 *
 * arg4 : Outout shape delimited by spaces.
 *
 * TODO:
 *       Supports only one input and one output.
 *       Assume input and output are float only.
 *       Device type is assumed to be OpenCL. Change below if needed.
 */

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "cnpy.h"

#include <fstream>
#include <iterator>
#include <algorithm>
#include <complex>

#define MODEL_INDEX 1
#define INPUT_INDEX 2
#define OUTPUT_INDEX 3
#define OUTPUT_SHAPE_INDEX 4

int main(int argc, char *argv[])
{

    if (argc < 4) {
        printf ("Usage : ./deploy <model_basename> <input> <output dump> <output shape[0]> .... <output shape[n]>\n");
        exit(1);
    }

    std::string model_arg (argv[MODEL_INDEX]);
    std::string model_lib = model_arg + ".so";
    std::string model_json = model_arg + ".json";
    std::string model_params = model_arg + ".params";

    std::string model_input (argv[INPUT_INDEX]);

    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_lib);

    // json graph
    std::ifstream json_in(model_json, std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in(model_params, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    std::cout << "Module Loaded\n";

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLOpenCL; //kDLCPU;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

    cnpy::NpyArray in_arr = cnpy::npy_load(model_input + ".npy");
    float* loaded_data = in_arr.data<float>();

    DLTensor* x;
    int in_ndim = in_arr.shape.size();
    int64_t *in_shape = (int64_t *) &in_arr.shape[0];
    uint64_t num_samples=1;
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &x);

    // Initlalize the Input
    for (int ii=0; ii < in_ndim; ii++)
        num_samples *= in_arr.shape[ii];

    for (int ii=0; ii < num_samples; ii++)
        (static_cast<float*>(x->data) + x->byte_offset/4)[ii] = loaded_data[ii];

    std::cout << "Set Input\n";
    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input(argv[2], x);

    std::cout << "Load Params\n";
    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    std::cout << "Call Run\n";
    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    // Extract Output
    DLTensor* y;
    int out_ndim = argc - OUTPUT_SHAPE_INDEX;
    int64_t out_shape[out_ndim];
    uint64_t num_out_samples=1;
    std::vector<size_t> out_shape_vec(out_ndim);

    for (int ii=0; ii < out_ndim; ii++) {
        out_shape[ii] = atoi(argv[OUTPUT_SHAPE_INDEX+ii]);
        num_out_samples *= out_shape[ii];
        out_shape_vec[ii] = out_shape[ii];
    }

    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &y);

    std::cout << "Get Output\n";
    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);


    // Write data to npy output file.
    std::vector<float> out_data(num_out_samples);
    for (int ii=0; ii<num_out_samples-10; ii++)
        out_data[ii] = (static_cast<float*>(y->data))[ii];

    cnpy::npy_save(argv[OUTPUT_INDEX], &out_data[0], out_shape_vec, "w");

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
