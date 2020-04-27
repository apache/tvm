using System;
using Xunit;
using TVMRuntime;

namespace Tests
{
    public class UnitTestModule
    {
        [Fact]
        public void EmptyModuleCreateSuccess()
        {
            Module module = new Module();
            module.DisposeModule();
        }

        [Fact]
        public void ModuleCreateInvalidPath()
        {
            Module module = new Module("Non existing path", "");
            Assert.Equal(module.ModuleHandle, UIntPtr.Zero);
        }

        [Fact]
        public void ModuleCreateSuccess()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            module.DisposeModule();
        }

        [Fact]
        public void ModuleLoadEmbedFuncInvalid()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            UIntPtr funcHandle = UIntPtr.Zero;
            module.GetModuleEmbededFunc("empty", queryImports: 0, ref funcHandle);
            Assert.Equal(funcHandle, UIntPtr.Zero);
            module.DisposeModule();
        }

        [Fact]
        public void ModuleLoadEmbedFuncValid()
        {
            Module module = new Module("asset/test_addone_dll.so", "");
            UIntPtr funcHandle = UIntPtr.Zero;
            module.GetModuleEmbededFunc("addone", 0, ref funcHandle);
            Assert.NotEqual(module.ModuleHandle, UIntPtr.Zero);

            // Execute the embedded function in the module
            // NOTE: Here the function f = addone(x) => x + 1
            int[] shape = { 10 };
            TVMContext ctx = new TVMContext(0);

            NDArray x_nd = new NDArray(shape, 1, "float32", ctx);
            NDArray y_nd = new NDArray(shape, 1, "float32", ctx);

            for (int i = 0; i < shape[0]; ++i)
            {
                x_nd[i] = (float)i;
            }

            PFManager.RunPackedFunc(funcHandle,
                new object[] { x_nd.NDArrayHandle, y_nd.NDArrayHandle });

            for (int i = 0; i < shape[0]; ++i)
            {
                Assert.Equal(y_nd[i], (float)x_nd[i] + 1);
            }

            module.DisposeModule();
        }
    }
}
