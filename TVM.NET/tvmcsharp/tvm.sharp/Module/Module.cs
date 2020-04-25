using System;
using static TVMRuntime.UnmanagedModuleWrapper;

namespace TVMRuntime
{
    public class Module
    {
        /// <summary>
        /// The module lib path.
        /// </summary>
        private string modLibPath = "";

        /// <summary>
        /// The module lib format.
        /// </summary>
        private string modLibFormat = "";

        /// <summary>
        /// The module handle.
        /// </summary>
        private UIntPtr moduleHandle = UIntPtr.Zero;

        /// <summary>
        /// Gets the module handle.
        /// </summary>
        /// <value>The module handle.</value>
        public UIntPtr ModuleHandle { get => moduleHandle;}

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
            moduleHandle = other.moduleHandle;
            modLibPath = other.modLibPath;
            modLibFormat = other.modLibFormat;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="path">Path.</param>
        /// <param name="format">Format.</param>
        public Module(string path, string format)
        {
            modLibPath = path;
            modLibFormat = format;

            UnmanagedModuleWrapper.LoadModuleFromFile(modLibPath,
                modLibFormat, ref moduleHandle);

        }

        /// <summary>
        /// Imports the module.
        /// </summary>
        /// <param name="depMod">Dep mod.</param>
        public void ImportModule(UIntPtr depMod)
            => UnmanagedModuleWrapper.ImportModule(moduleHandle, depMod);

        /// <summary>
        /// Gets the module embeded func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="queryImports">Query imports.</param>
        /// <param name="funcHandle">Func handle.</param>
        public void GetModuleEmbededFunc(string funcName,
            int queryImports, ref UIntPtr funcHandle)
            => UnmanagedModuleWrapper.GetModuleEmbededFunc(moduleHandle,
                funcName, queryImports, ref funcHandle);

        /// <summary>
        /// Disposes the module.
        /// </summary>
        public void DisposeModule()
        {
            if (!UIntPtr.Zero.Equals(moduleHandle))
            {
                UnmanagedModuleWrapper.DisposeModule(moduleHandle);
                moduleHandle = UIntPtr.Zero;
            }
        }


        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.Module"/> is reclaimed by garbage collection.
        /// </summary>
        ~Module()
        {
            if (!UIntPtr.Zero.Equals(moduleHandle))
            {
                UnmanagedModuleWrapper.DisposeModule(moduleHandle);
            }
        }
    }
}
