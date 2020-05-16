using System;


namespace TVMRuntime
{
    public sealed class PackedFunction : IDisposable
    {
        IntPtr _funcHandle = IntPtr.Zero;
        string _funcName = "No name";

        public PackedFunction(string funcName)
        {
            PFManager.GetGlobalPackedFunc(funcName, ref _funcHandle);
            _funcName = funcName;
        }

        public PackedFunction(IntPtr funcHandle)
        {
            FuncHandle = funcHandle;
        }

        public IntPtr FuncHandle
        { get => _funcHandle; set
            {
                // To make sure there is no memory leak
                DisposePackedFunction();

                _funcHandle = value;
            }
        }

        public TVMValue Invoke(params object[] inputArgs)
        {
            return PFManager.RunPackedFunc(_funcHandle, inputArgs);
        }

        public void DisposePackedFunction()
        {
            if (!IntPtr.Zero.Equals(_funcHandle))
            {
                PFManager.DisposePackedFunc(_funcHandle);
                _funcHandle = IntPtr.Zero;
            }
        }

        private bool disposedValue = false;

        void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                DisposePackedFunction();
                disposedValue = true;
            }
        }

        ~PackedFunction() 
        {
            Dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
