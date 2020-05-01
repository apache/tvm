using System;
using Xunit;
using TVMRuntime;

namespace Tests
{
    public class UnitTestRuntime
    {
        [Fact]
        public void RuntimeCreateInvalidInputs()
        {
            RuntimeParams runtimeParams = new RuntimeParams();
            runtimeParams.modLibPath = "";
            runtimeParams.modLibFormat = "";
            runtimeParams.paramDictPath = "";
            runtimeParams.graphJsonPath = "";
            runtimeParams.context = new TVMContext(0);

            var exception = Assert.Throws<ArgumentException>(() => Runtime.Create(runtimeParams));
            Assert.Contains("RuntimeParams.modLibPath", exception.Message);

            runtimeParams.modLibPath = "asset/resnet50tvm.so";
            exception = Assert.Throws<ArgumentException>(() => Runtime.Create(runtimeParams));
            Assert.Contains("RuntimeParams.graphJsonPath", exception.Message);

            runtimeParams.graphJsonPath = "asset/resnet50tvm-graph.json";
            exception = Assert.Throws<ArgumentException>(() => Runtime.Create(runtimeParams));
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

            Runtime runtime = Runtime.Create(runtimeParams);
            Assert.NotEqual(runtime.RuntimeHandle, IntPtr.Zero);
            runtime.DisposeRuntime();
            Assert.Equal(runtime.RuntimeHandle, IntPtr.Zero);
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

            Runtime runtime = Runtime.Create(runtimeParams);
            Assert.NotEqual(runtime.RuntimeHandle, IntPtr.Zero);

            // Load Params
            runtime.LoadParams();

            // Set Inputs
            long[] shape = { 1, 224, 224, 3 };
            TVMContext ctx = new TVMContext(0);
            NDArray input_1 = NDArray.Empty(shape, "float32", ctx);

            runtime.SetInput("input_1", input_1);

            // Run the graph
            runtime.Run();

            // Get Output
            NDArray output = runtime.GetOutput(0);

            Assert.NotEqual(output.NDArrayHandle, IntPtr.Zero);

            // Check Output Data
            Assert.Equal(2, output.Ndim);
            Assert.Equal(new long[] { 1, 1000 }, output.Shape);
            Assert.Equal(1000, output.Size);

            // Release all resources
            runtime.DisposeRuntime();
            Assert.Equal(runtime.RuntimeHandle, IntPtr.Zero);

            output.Dispose();
            Assert.Equal(output.NDArrayHandle, IntPtr.Zero);

            input_1.Dispose();
            Assert.Equal(input_1.NDArrayHandle, IntPtr.Zero);
        }
    }
}
