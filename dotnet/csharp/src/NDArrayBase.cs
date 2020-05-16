using System;
using System.Runtime.InteropServices;
using Native;
using static TVMRuntime.Utils;

namespace TVMRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    /// <summary>
    /// Data type.
    /// </summary>
    internal struct TVMDataTypeNative
    {
        /// <summary>
        /// The code.
        /// </summary>
        public byte code;

        /// <summary>
        /// The bits.
        /// </summary>
        public byte bits;

        /// <summary>
        /// The lanes.
        /// </summary>
        public ushort lanes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TensorContextUnmanaged
    {
        /// <summary>
        /// The type of the device.
        /// </summary>
        public DeviceType deviceType;

        /// <summary>
        /// The device identifier.
        /// </summary>
        public int deviceId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct TVMTensor
    {
        public IntPtr data;
        public TensorContextUnmanaged ctx;
        public int ndim;
        public TVMDataTypeNative dtype;
        public IntPtr shape;
        public IntPtr strides;
        public uint byte_offset;
    }


    public class NDArrayBase
    {
        private IntPtr _arrayHandle = IntPtr.Zero;

        // NOTE:
        // This is an experimental supprot in Csharp runtime.
        // With this feature one NDArray can be accessed with indexer
        // like: NDArray xNdArry = NDArray.Empty({3}, "float32");
        //       float elemAt =  xNdArry[1];
        // 
        // To support above feature a duplicate buffer as below is maintained
        // for a fast access of data(avoids Pinvoke). It is an additional cost 
        //  in terms of memory. However the feature is helpful.
        private byte[] _internalBuffer;

        /// <summary>
        /// Gets the shape.
        /// </summary>
        /// <value>The shape.</value>
        public long[] Shape { get; private set; }

        /// <summary>
        /// Gets the ndim.
        /// </summary>
        /// <value>The ndim.</value>
        public int Ndim { get; private set; } = 0;

        /// <summary>
        /// Gets the size.
        /// </summary>
        /// <value>The size.</value>
        public long Size { get; private set; } = 0;

        public IntPtr NDArrayHandle
        {
            get => _arrayHandle; set
            {
                // To make sure there is no memory leak
                if (!IntPtr.Zero.Equals(_arrayHandle))
                {
                    DisposeNDArray();
                }
                _arrayHandle = value;
                Shape = GetNDArrayShape(_arrayHandle);
                Ndim = GetNDArrayNdim(_arrayHandle);
                Size = 1;
                Array.ForEach(Shape, i => Size *= i);

                TVMDataType dataType = new TVMDataType();
                GetNDArrayDtype(_arrayHandle, ref dataType);
                NumOfBytes = dataType.numOfBytes;

                _internalBuffer = new byte[Size * NumOfBytes];
            }
        }

        private int NumOfBytes { get; set; } = 0;

        protected NDArrayBase()
        {

        }

        /// <summary>
        /// Creates the NDArray.
        /// </summary>
        /// <param name="shape">Shape.</param>
        /// <param name="ndim">Ndim.</param>
        /// <param name="dtypeCode">Dtype code.</param>
        /// <param name="dtypeBits">Dtype bits.</param>
        /// <param name="dtypeLanes">Dtype lanes.</param>
        /// <param name="deviceType">Device type.</param>
        /// <param name="deviceId">Device identifier.</param>
        public void CreateNDArray(long[] shape,
                          int ndim,
                          byte dtypeCode,
                          byte dtypeBits,
                          ushort dtypeLanes,
                          int deviceType,
                          int deviceId)
        {
            int result = NativeImport.TVMArrayAlloc(shape, ndim, dtypeCode, dtypeBits, dtypeLanes,
                    deviceType, deviceId, ref _arrayHandle);

            Utils.CheckSuccess(0, result);

            Shape = shape;
            Ndim = shape.Length;
            Size = 1;
            Array.ForEach(Shape, i => Size *= i);
            NumOfBytes = Utils.GetNumOfBytes(dtypeBits);
            _internalBuffer = new byte[Size * NumOfBytes];
        }

        /// <summary>
        /// Gets the NDArray shape.
        /// </summary>
        /// <returns>The NDArray shape.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        private static long[] GetNDArrayShape(IntPtr arrayHandle)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            int ndim = tensor.ndim;
            long[] shapeOut = new long[ndim];

            Marshal.Copy(tensor.shape, shapeOut, 0, shapeOut.Length);

            return shapeOut;
        }

        /// <summary>
        /// Gets the NDArray ndim.
        /// </summary>
        /// <returns>The NDArray ndim.</returns>
        /// <param name="arrayHandle">Array handle.</param>
        private static int GetNDArrayNdim(IntPtr arrayHandle)
        {
            return Marshal.PtrToStructure<TVMTensor>(arrayHandle).ndim;
        }

        /// <summary>
        /// Gets the NDArray dtype.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="dataType">Data type.</param>
        protected static void GetNDArrayDtype(IntPtr arrayHandle, ref TVMDataType dataType)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            dataType.code = tensor.dtype.code;
            dataType.bits = tensor.dtype.bits;
            dataType.lanes = tensor.dtype.lanes;
            dataType.numOfBytes = (byte)Utils.GetNumOfBytes(dataType.bits);
        }

        /// <summary>
        /// Gets the NDArray context.
        /// </summary>
        /// <param name="arrayHandle">Array handle.</param>
        /// <param name="ctx">Context.</param>
        protected static void GetNDArrayCtx(IntPtr arrayHandle, ref TVMContext ctx)
        {
            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(arrayHandle);
            ctx.deviceId = tensor.ctx.deviceId;
            ctx.deviceType = tensor.ctx.deviceType;
        }

        /// <summary>
        /// Gets the offset.
        /// </summary>
        /// <returns>The offset.</returns>
        /// <param name="numOfBytes">Number of bytes.</param>
        /// <param name="currIndex">Curr index.</param>
        private static int GetOffset(int numOfBytes, int currIndex)
        {
            return numOfBytes * currIndex;
        }

        public object this[int i]
        {
            get { return GetNDArrayElem(i); }
            set { SetNDArrayElem(i, value); }
        }

        /// <summary>
        /// Gets the NDArray element.
        /// </summary>
        /// <returns>The NDArray element.</returns>
        /// <param name="index">Index.</param>
        private object GetNDArrayElem(int index)
        {
            if (index < 0 || index >= Size)
            {
                throw new System.ArgumentOutOfRangeException("Index should be in range [0, " + (Size - 1) + "].");
            }

            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(_arrayHandle);
            byte[] to = _internalBuffer;
            Marshal.Copy(tensor.data, to, 0, to.Length);

            TVMDataType dataType = new TVMDataType();
            GetNDArrayDtype(_arrayHandle, ref dataType);

            if (dataType.IsFloat32())
            {
                return BitConverter.ToSingle(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsFloat64())
            {
                return BitConverter.ToDouble(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsInt32())
            {
                return BitConverter.ToInt32(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsInt64())
            {
                return BitConverter.ToInt64(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsInt16())
            {
                return BitConverter.ToInt16(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsUint16())
            {
                return BitConverter.ToChar(to, GetOffset(dataType.numOfBytes, index));
            }
            else if (dataType.IsInt8())
            {
                return to[GetOffset(dataType.numOfBytes, index)];
            }
            else
            {
                throw new System.ArrayTypeMismatchException("Unknown type");
            }
        }

        /// <summary>
        /// Sets the NDArray element.
        /// </summary>
        /// <param name="index">Index.</param>
        /// <param name="value">Value.</param>
        private void SetNDArrayElem(int index, object value)
        {
            if (index < 0 || index >= Size)
            {
                throw new System.ArgumentOutOfRangeException("Index should be in range [0, " + (Size - 1) + "].");
            }

            TVMTensor tensor = Marshal.PtrToStructure<TVMTensor>(_arrayHandle);
            byte[] to = _internalBuffer;
            Marshal.Copy(tensor.data, to, 0, to.Length);

            TVMDataType dataType = new TVMDataType();
            GetNDArrayDtype(_arrayHandle, ref dataType);

            if (dataType.IsFloat32())
            {
                byte[] inputValue = BitConverter.GetBytes((float)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsFloat64())
            {
                byte[] inputValue = BitConverter.GetBytes((double)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsInt32())
            {
                byte[] inputValue = BitConverter.GetBytes((int)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsInt64())
            {
                byte[] inputValue = BitConverter.GetBytes((long)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsInt16())
            {
                byte[] inputValue = BitConverter.GetBytes((short)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsUint16())
            {
                byte[] inputValue = BitConverter.GetBytes((char)value);
                Buffer.BlockCopy(inputValue, 0, to, GetOffset(dataType.numOfBytes, index), inputValue.Length);
            }
            else if (dataType.IsInt8())
            {
                to[index] = (byte)value;
            }
            else
            {
                throw new System.ArrayTypeMismatchException("Unknown type");
            }

            Marshal.Copy(to, 0, tensor.data, to.Length);
        }

        /// <summary>
        /// Copies the float data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyFloatDataToNDArray(float[] from)
        {
            int total_size = from.Length * NumOfBytes;
            byte[] byteArray = new byte[total_size];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the float data from NDArray.
        /// </summary>
        /// <returns>The float data from NDArray.</returns>
        protected float[] GetFloatDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            float[] toArray = new float[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the double data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyDoubleDataToNDArray(double[] from)
        {
            int totalSize = from.Length * NumOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the double data from NDArray.
        /// </summary>
        /// <returns>The double data from NDArray.</returns>
        protected double[] GetDoubleDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            double[] toArray = new double[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the int data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyIntDataToNDArray(int[] from)
        {
            int totalSize = from.Length * NumOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the int data from NDArray.
        /// </summary>
        /// <returns>The int data from NDArray.</returns>
        protected int[] GetIntDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            int[] toArray = new int[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the long data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyLongDataToNDArray(long[] from)
        {
            int totalSize = from.Length * NumOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the long data from NDArray.
        /// </summary>
        /// <returns>The long data from NDArray.</returns>
        protected long[] GetLongDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            long[] toArray = new long[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the short data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyShortDataToNDArray(short[] from)
        {
            int totalSize = from.Length * NumOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the short data from NDArray.
        /// </summary>
        /// <returns>The short data from NDArray.</returns>
        protected short[] GetShortDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            short[] toArray = new short[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Copies the char data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyCharDataToNDArray(char[] from)
        {
            int totalSize = from.Length * NumOfBytes;
            byte[] byteArray = new byte[totalSize];

            Buffer.BlockCopy(from, 0, byteArray, 0, byteArray.Length);

            CopyByteDataToNDArray(byteArray);
        }

        /// <summary>
        /// Gets the char data from NDArray.
        /// </summary>
        /// <returns>The char data from NDArray.</returns>
        protected char[] GetCharDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);

            char[] toArray = new char[Size];
            Buffer.BlockCopy(fromArray, 0, toArray, 0, fromArray.Length);
            return toArray;
        }

        /// <summary>
        /// Gets the byte data from NDArray.
        /// </summary>
        /// <returns>The byte data from NDArray.</returns>
        protected byte[] GetByteDataFromNDArray()
        {
            long totalSize = Size * NumOfBytes;
            byte[] fromArray = new byte[totalSize];

            GetByteDataFromNDArray(fromArray);
            return fromArray;
        }

        /// <summary>
        /// Copies the byte data to NDArray.
        /// </summary>
        /// <param name="from">From.</param>
        protected void CopyByteDataToNDArray(byte[] from)
        {
            IntPtr unmanagedPointer = Marshal.AllocHGlobal(from.Length);
            try
            {
                Marshal.Copy(from, 0, unmanagedPointer, from.Length);

                NativeImport.TVMArrayCopyFromBytes(_arrayHandle, unmanagedPointer, from.LongLength);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(unmanagedPointer);
            }
        }

        /// <summary>
        /// Gets the byte data from NDArray.
        /// </summary>
        /// <param name="to">To.</param>
        private void GetByteDataFromNDArray(byte[] to)
        {
            IntPtr unmanagedPointer = Marshal.AllocHGlobal(to.Length);
            try
            {
                NativeImport.TVMArrayCopyToBytes(_arrayHandle, unmanagedPointer, to.LongLength);

                Marshal.Copy(unmanagedPointer, to, 0, to.Length);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(unmanagedPointer);
            }
        }

        /// <summary>
        /// Disposes the NDArray.
        /// </summary>
        public void DisposeNDArray()
        {
            if (!IntPtr.Zero.Equals(_arrayHandle))
            {
                int result = NativeImport.TVMArrayFree(_arrayHandle);
                Utils.CheckSuccess(0, result);

                _arrayHandle = IntPtr.Zero;
                _internalBuffer = null;
            }
        }

        ~NDArrayBase()
        {
            DisposeNDArray();
        }
    }
}
