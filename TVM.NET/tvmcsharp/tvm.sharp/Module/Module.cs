using System;
using System.IO;

namespace TVMRuntime
{
    public class Module
    {
        /// <summary>
        /// The module lib path.
        /// </summary>
        private string _modLibPath = "";

        /// <summary>
        /// The module lib format.
        /// </summary>
        private string _modLibFormat = "";

        /// <summary>
        /// The module handle.
        /// </summary>
        private IntPtr _moduleHandle = IntPtr.Zero;

        /// <summary>
        /// Gets the module handle.
        /// </summary>
        /// <value>The module handle.</value>
        public IntPtr ModuleHandle { get => _moduleHandle;}

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="other">Other.</param>
        public Module(Module other)
        {
            _moduleHandle = other._moduleHandle;
            _modLibPath = other._modLibPath;
            _modLibFormat = other._modLibFormat;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="path">Path.</param>
        /// <param name="format">Format.</param>
        public Module(string path, string format)
        {
            if ((!File.Exists(path)))
            {
                throw new System.ArgumentException("Please provide valid path for Module!");
            }
            _modLibPath = path;
            _modLibFormat = format;

            UnmanagedModuleWrapper.LoadModuleFromFile(_modLibPath,
                _modLibFormat, ref _moduleHandle);

        }

        /// <summary>
        /// Imports the module.
        /// </summary>
        /// <param name="depMod">Dep mod.</param>
        public void ImportModule(IntPtr depMod)
            => UnmanagedModuleWrapper.ImportModule(_moduleHandle, depMod);

        /// <summary>
        /// Gets the module embeded func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="queryImports">Query imports.</param>
        /// <param name="funcHandle">Func handle.</param>
        public void GetModuleEmbededFunc(string funcName,
            int queryImports, ref IntPtr funcHandle)
            => UnmanagedModuleWrapper.GetModuleEmbededFunc(_moduleHandle,
                funcName, queryImports, ref funcHandle);

        /// <summary>
        /// Disposes the module.
        /// </summary>
        public void DisposeModule()
        {
            if (!IntPtr.Zero.Equals(_moduleHandle))
            {
                UnmanagedModuleWrapper.DisposeModule(_moduleHandle);
                _moduleHandle = IntPtr.Zero;
            }
        }


        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.Module"/> is reclaimed by garbage collection.
        /// </summary>
        ~Module()
        {
            DisposeModule();
        }
    }
}
