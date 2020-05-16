using System;
using Xunit;
using TVMRuntime;

namespace Tests
{
    public class UnitTestModule
    {
        [Fact]
        public void ModuleCreateInvalidPath()
        {
            var exception = Assert.Throws<ArgumentException>(() => new Module("Non existing path", ""));
            Assert.Contains("provide valid path", exception.Message);
        }

        [Fact]
        public void ModuleCreateSuccess()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            Assert.NotEqual(module.ModuleHandle, IntPtr.Zero);
            module.DisposeModule();
        }

        [Fact]
        public void ModuleLoadEmbedFuncInvalid()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            IntPtr funcHandle = IntPtr.Zero;
            module.GetModuleEmbededFunc("empty", queryImports: 0, ref funcHandle);
            Assert.Equal(funcHandle, IntPtr.Zero);
            module.DisposeModule();
        }

        [Fact]
        public void ModuleLoadEmbedFuncValid()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            IntPtr funcHandle = IntPtr.Zero;
            module.GetModuleEmbededFunc("addone", 0, ref funcHandle);
            Assert.NotEqual(module.ModuleHandle, IntPtr.Zero);

            // Execute the embedded function in the module
            // NOTE: Here the function f = addone(x) => x + 1
            long[] shape = { 10 };
            float[] data = new float[shape[0]];
            TVMContext ctx = new TVMContext(0);

            NDArray x_nd = NDArray.Empty(shape, "float32", ctx);
            NDArray y_nd = NDArray.Empty(shape, "float32", ctx);

            for (int i = 0; i < shape[0]; ++i)
            {
                data[i] = (float)i;
            }

            x_nd.CopyFrom(data);

            PFManager.RunPackedFunc(funcHandle,
                new object[] { x_nd, y_nd });

            for (int i = 0; i < shape[0]; ++i)
            {
                Assert.Equal(y_nd[i], (float)x_nd[i] + 1);
            }

            module.DisposeModule();
        }
    }
}
