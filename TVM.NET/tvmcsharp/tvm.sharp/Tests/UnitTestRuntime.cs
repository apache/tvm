using System;
using Xunit;
using TVMRuntime;

namespace Tests
{
    public class UnitTestRuntime
    {
        [Fact]
        public void EmptyRuntimeCreateSuccess()
        {
            Runtime runtime = new Runtime();
            Assert.Equal(runtime.RuntimeHandle, UIntPtr.Zero);
            runtime.DisposeRuntime();
        }

        [Fact]
        public void RuntimeCreateInvalidInputs()
        {
            RuntimeParams runtimeParams = new RuntimeParams();
            runtimeParams.modLibPath = "";
            runtimeParams.modLibFormat = "";
            runtimeParams.paramDictPath = "";
            runtimeParams.graphJsonPath = "";
            runtimeParams.context = new TVMContext(0);

            var exception = Assert.Throws<ArgumentException>(() => new Runtime(runtimeParams));
            Assert.Contains("RuntimeParams.modLibPath", exception.Message);

            runtimeParams.modLibPath = "asset/resnet50tvm.so";
            exception = Assert.Throws<ArgumentException>(() => new Runtime(runtimeParams));
            Assert.Contains("RuntimeParams.graphJsonPath", exception.Message);

            runtimeParams.graphJsonPath = "asset/resnet50tvm-graph.json";
            exception = Assert.Throws<ArgumentException>(() => new Runtime(runtimeParams));
            Assert.Contains("RuntimeParams.paramDictPath", exception.Message);
        }

        [Fact]
        public void RuntimeCreateValidInputSuccess()
        {
            RuntimeParams runtimeParams = new RuntimeParams();
            runtimeParams.modLibPath = "asset/resnet50tvm.so";
            runtimeParams.modLibFormat = "";
            runtimeParams.paramDictPath = "asset/resnet50tvm-params_dict";
            runtimeParams.graphJsonPath = "asset/resnet50tvm-graph.json";
            runtimeParams.context = new TVMContext(0);

            Runtime runtime = new Runtime(runtimeParams);
            Assert.NotEqual(runtime.RuntimeHandle, UIntPtr.Zero);
            runtime.DisposeRuntime();
            Assert.Equal(runtime.RuntimeHandle, UIntPtr.Zero);
        }

        [Fact]
        public void RuntimeProcSuccess()
        {
            RuntimeParams runtimeParams = new RuntimeParams();
            runtimeParams.modLibPath = "asset/resnet50tvm.so";
            runtimeParams.modLibFormat = "";
            runtimeParams.paramDictPath = "asset/resnet50tvm-params_dict";
            runtimeParams.graphJsonPath = "asset/resnet50tvm-graph.json";
            runtimeParams.context = new TVMContext(0);

            Runtime runtime = new Runtime(runtimeParams);
            Assert.NotEqual(runtime.RuntimeHandle, UIntPtr.Zero);

            // Load Params
            runtime.LoadParams();

            // Set Inputs
            int[] shape = { 1, 224, 224, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray input_1 = new NDArray(shape, shape.Length, "float32", ctx);

            unsafe
            {
                runtime.SetInput("input_1", (UIntPtr)(input_1.NDArrayHandle.ToPointer()));
            }

            NDArray output = new NDArray();
            Assert.Equal(output.NDArrayHandle, IntPtr.Zero);

            // Get Output
            runtime.GetOutput(0, ref output);

            Assert.NotEqual(output.NDArrayHandle, IntPtr.Zero);

            runtime.DisposeRuntime();
            Assert.Equal(runtime.RuntimeHandle, UIntPtr.Zero);

            Console.WriteLine("Ndim: " + output.Ndim);
            Console.WriteLine("Size: " + output.Size);

            output.DisposeNDArray();
            Assert.Equal(output.NDArrayHandle, IntPtr.Zero);
        }
    }
}
