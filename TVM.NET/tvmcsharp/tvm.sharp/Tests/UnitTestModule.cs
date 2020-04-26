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
            Assert.ThrowsAny<Exception>(() => new Module("Non existing path", ""));
        }

        [Fact]
        public void ModuleCreateSuccess()
        {
            Module module = new Module("assets/test_addone_dll.so", "");
            module.DisposeModule();
        }
    }
}
