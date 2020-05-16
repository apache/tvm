using System;
using System.IO;
using Native;

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
        public IntPtr ModuleHandle
        { get => _moduleHandle; set
            {
                // Release previous module if any
                DisposeModule();
                _moduleHandle = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.Module"/> class.
        /// </summary>
        /// <param name="moduleHandle">Module handle.</param>
        public Module(IntPtr moduleHandle)
        {
            _moduleHandle = moduleHandle;
        }

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

            int result = NativeImport.TVMModLoadFromFile(_modLibPath, _modLibFormat,
                    ref _moduleHandle);
            Utils.CheckSuccess(0, result);

        }

        /// <summary>
        /// Imports the module.
        /// </summary>
        /// <param name="depMod">Dep mod.</param>
        public void ImportModule(IntPtr depMod)
        {
            int result = NativeImport.TVMModImport(_moduleHandle, depMod);
            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Gets the module embeded func.
        /// </summary>
        /// <param name="funcName">Func name.</param>
        /// <param name="queryImports">Query imports.</param>
        /// <param name="funcHandle">Func handle.</param>
        public void GetModuleEmbededFunc(string funcName,
            int queryImports, ref IntPtr funcHandle)
        {
            int result = NativeImport.TVMModGetFunction(_moduleHandle, funcName, queryImports,
                    ref funcHandle);
            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Loads the module from file.
        /// </summary>
        /// <param name="fileName">File name.</param>
        /// <param name="format">Format.</param>
        /// <param name="modHandle">Module handle.</param>
        private static void LoadModuleFromFile(string fileName,
            string format, ref IntPtr modHandle)
        {
            int result = NativeImport.TVMModLoadFromFile(fileName, format,
                    ref modHandle);
            Utils.CheckSuccess(0, result);
        }

        /// <summary>
        /// Disposes the module.
        /// </summary>
        public void DisposeModule()
        {
            if (!IntPtr.Zero.Equals(_moduleHandle))
            {
                int result = NativeImport.TVMModFree(_moduleHandle);
                Utils.CheckSuccess(0, result);
                _modLibPath = "";
                _modLibFormat = "";
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
