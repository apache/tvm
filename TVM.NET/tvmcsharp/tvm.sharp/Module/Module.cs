using System;
using static TVMRuntime.UnmanagedModuleWrapper;

namespace TVMRuntime
{
    public class Module
    {
        /// <summary>
        /// The module lib path.
        /// </summary>
        private string mod_lib_path = "";

        /// <summary>
        /// The module lib format.
        /// </summary>
        private string mod_lib_format = "";

        /// <summary>
        /// The module handle.
        /// </summary>
        private UIntPtr module_handle = UIntPtr.Zero;

        /// <summary>
        /// Gets the module handle.
        /// </summary>
        /// <value>The module handle.</value>
        public UIntPtr Module_handle { get => module_handle;}

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        public Module()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="other">Other.</param>
        public Module(Module other)
        {
            module_handle = other.module_handle;
            mod_lib_path = other.mod_lib_path;
            mod_lib_format = other.mod_lib_format;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="path">Path.</param>
        /// <param name="format">Format.</param>
        public Module(string path, string format)
        {
            mod_lib_path = path;
            mod_lib_format = format;

            UnmanagedModuleWrapper.LoadModuleFromFile(mod_lib_path,
                mod_lib_format, ref module_handle);

            Console.WriteLine("Jai hanuman module created");
            Console.WriteLine(module_handle);
        }

        /// <summary>
        /// Imports the module.
        /// </summary>
        /// <param name="dep_mod">Dep mod.</param>
        public void ImportModule(UIntPtr dep_mod)
            => UnmanagedModuleWrapper.ImportModule(module_handle, dep_mod);

        /// <summary>
        /// Gets the module embeded func.
        /// </summary>
        /// <param name="func_name">Func name.</param>
        /// <param name="query_imports">Query imports.</param>
        /// <param name="func_handle">Func handle.</param>
        public void GetModuleEmbededFunc(string func_name,
            int query_imports, ref UIntPtr func_handle)
            => UnmanagedModuleWrapper.GetModuleEmbededFunc(module_handle,
                func_name, query_imports, ref func_handle);

        /// <summary>
        /// Disposes the module.
        /// </summary>
        public void DisposeModule()
        {
            Console.WriteLine("Jai hanuman module destroyed");
            if (!UIntPtr.Zero.Equals(module_handle))
            {
                UnmanagedModuleWrapper.DisposeModule(module_handle);
                module_handle = UIntPtr.Zero;
            }
        }


        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.Module"/> is reclaimed by garbage collection.
        /// </summary>
        ~Module()
        {
            if (!UIntPtr.Zero.Equals(module_handle))
            {
                UnmanagedModuleWrapper.DisposeModule(module_handle);
            }
        }
    }
}
