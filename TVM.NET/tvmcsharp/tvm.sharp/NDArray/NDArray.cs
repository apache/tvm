using System;

namespace TVMRuntime
{
    /// <summary>
    /// TVM Data type code.
    /// </summary>
    enum TVMDataTypeCode
    {
        Int = 0,
        UInt = 1,
        Float = 2,
    }

    public struct TVMDataType
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

        /// <summary>
        /// The number of bytes.
        /// </summary>
        public byte numOfBytes;

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.TVMDataType"/> struct.
        /// </summary>
        /// <param name="dtype">Dtype.</param>
        public TVMDataType(string dtype) : this()
        {
            string lowerDtype = dtype.ToLower();
            bits = 0;

            // TODO: support user input for lanes
            lanes = 1;
            if (lowerDtype.StartsWith("i") && lowerDtype.Contains("int"))
            {
                code = (byte)TVMDataTypeCode.Int;
                bits = (byte)Int32.Parse(lowerDtype.Split("int")[1]);
            }
            else if (lowerDtype.StartsWith("u") && lowerDtype.Contains("uint"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = (byte)Int32.Parse(lowerDtype.Split("uint")[1]);
            }
            else if (lowerDtype.StartsWith("f") && lowerDtype.Contains("float"))
            {
                code = (byte)TVMDataTypeCode.Float;
                bits = (byte)Int32.Parse(lowerDtype.Split("float")[1]);
            }
            else if (lowerDtype.StartsWith("b") && lowerDtype.Contains("bool"))
            {
                code = (byte)TVMDataTypeCode.UInt;
                bits = 1;
            }

            if (((bits & (bits - 1)) != 0) || (bits < 8))
            {
                throw new System.ArgumentException(dtype + " not supported!");
            }

            numOfBytes = (byte)(bits / 8);
        }

        // Byte type
        public bool IsInt8()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 8)) return true;
            return false;
        }

        // Char type
        public bool IsUint16()
        {
            if ((code == (byte)TVMDataTypeCode.UInt) && (bits == 16)) return true;
            return false;
        }

        // Short type
        public bool IsInt16()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 16)) return true;
            return false;
        }

        // Int type
        public bool IsInt32()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 32)) return true;
            return false;
        }

        // Long type
        public bool IsInt64()
        {
            if ((code == (byte)TVMDataTypeCode.Int) && (bits == 64)) return true;
            return false;
        }

        // Float type
        public bool IsFloat32()
        {
            if ((code == (byte)TVMDataTypeCode.Float) && (bits == 32)) return true;
            return false;
        }

        // Double type
        public bool IsFloat64()
        {
            if ((code == (byte)TVMDataTypeCode.Float) && (bits == 64)) return true;
            return false;
        }
    }

    public class NDArray : IDisposable
    {
        private IntPtr _arrayHandle = IntPtr.Zero;
        private int _ndim = 0;
        private long[] _shape;
        private long _arraySize = 0;
        private TVMDataType _dataType;

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
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        private NDArray()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="ctx">Context.</param>
        private NDArray(long[] shapeInp,
                          TVMDataType dataType,
                          TVMContext ctx)
        {
            UnmanagedNDArrayWrapper.CreateNDArray(shapeInp, shapeInp.Length, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId, ref _arrayHandle);
            _shape = shapeInp;
            _ndim = shapeInp.Length;
            _arraySize = 1;
            Array.ForEach(_shape, i => _arraySize *= i);
            _dataType = dataType;
            _internalBuffer = new byte[_arraySize * _dataType.numOfBytes];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataTypeStr">Data type string.</param>
        /// <param name="ctx">Context.</param>
        private NDArray(long[] shapeInp,
                          string dataTypeStr,
                          TVMContext ctx)
        {
            TVMDataType dataType = new TVMDataType(dataTypeStr);
            UnmanagedNDArrayWrapper.CreateNDArray(shapeInp, shapeInp.Length, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId, ref _arrayHandle);
            _shape = shapeInp;
            _ndim = shapeInp.Length;
            _arraySize = 1;
            Array.ForEach(_shape, i => _arraySize *= i);
            _dataType = dataType;
            _internalBuffer = new byte[_arraySize * _dataType.numOfBytes];
        }

        /// <summary>
        /// Empty NDArray instance.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        public static NDArray Empty()
        {
            return new NDArray();
        }

        /// <summary>
        /// Empty NDArray instance with the specified shapeInp and dataTypeStr.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataTypeStr">Data type string.</param>
        public static NDArray Empty(long[] shapeInp,
                          string dataTypeStr = "float32")
        {
            TVMContext ctx = new TVMContext(0);
            return new NDArray(shapeInp, dataTypeStr, ctx);
        }

        /// <summary>
        /// Empty NDArray instance with the specified shapeInp and ctx.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="ctx">Context.</param>
        public static NDArray Empty(long[] shapeInp, TVMContext ctx)
        {
            return new NDArray(shapeInp, "float32", ctx);
        }

        /// <summary>
        /// Empty NDArray instance with the specified shapeInp and dataType.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataType">Data type.</param>
        public static NDArray Empty(long[] shapeInp, TVMDataType dataType)
        {
            TVMContext ctx = new TVMContext(0);
            return new NDArray(shapeInp, dataType, ctx);
        }

        /// <summary>
        /// Empty NDArray instance with the specified shapeInp, dataType and ctx.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="ctx">Context.</param>
        public static NDArray Empty(long[] shapeInp, TVMDataType dataType, TVMContext ctx)
        {
            return new NDArray(shapeInp, dataType, ctx);
        }

        /// <summary>
        /// Empty NDArray instance with the specified shapeInp, dataTypeStr and ctx.
        /// </summary>
        /// <returns>The empty NDArray instance.</returns>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="dataTypeStr">Data type string.</param>
        /// <param name="ctx">Context.</param>
        public static NDArray Empty(long[] shapeInp, string dataTypeStr, TVMContext ctx)
        {
            return new NDArray(shapeInp, dataTypeStr, ctx);
        }

        /// <summary>
        /// Gets or sets the NDArray handle.
        /// </summary>
        /// <value>The NDArray handle.</value>
        public IntPtr NDArrayHandle
        {
            get => _arrayHandle; set
            {
                // To make sure there is no memory leak
                if (!IntPtr.Zero.Equals(_arrayHandle))
                {
                    UnmanagedNDArrayWrapper.DisposeNDArray(_arrayHandle);
                }
                _arrayHandle = value;
                _shape = UnmanagedNDArrayWrapper.GetNDArrayShape(_arrayHandle);
                _ndim = UnmanagedNDArrayWrapper.GetNDArrayNdim(_arrayHandle);
                _arraySize = 1;
                Array.ForEach(_shape, i => _arraySize *= i);
                UnmanagedNDArrayWrapper.GetNDArrayDtype(_arrayHandle, ref _dataType);

                _internalBuffer = new byte[_arraySize * _dataType.numOfBytes];
            }
        }

        /// <summary>
        /// Gets the shape.
        /// </summary>
        /// <value>The shape.</value>
        public long[] Shape { get => _shape; }

        /// <summary>
        /// Gets the ndim.
        /// </summary>
        /// <value>The ndim.</value>
        public int Ndim { get => _ndim; }

        /// <summary>
        /// Gets the size.
        /// </summary>
        /// <value>The size.</value>
        public long Size { get => _arraySize; }

        /// <summary>
        /// Gets the internal buffer.
        /// </summary>
        /// <value>The internal buffer.</value>
        internal byte[] InternalBuffer { get => _internalBuffer; }

        /// <summary>
        /// Gets the type of the data.
        /// </summary>
        /// <value>The type of the data.</value>
        public TVMDataType DataType { get => _dataType; }

        /// <summary>
        /// Gets the <see cref="T:TVMRuntime.NDArray"/> with the specified i.
        /// </summary>
        /// <param name="i">The index.</param>
        public object this[int i]
        {
            get { return UnmanagedNDArrayWrapper.GetNDArrayElem(this, i); }
        }

        /// <summary>
        /// Checks the size of the copy.
        /// </summary>
        private void CheckCopySize(int copyLength)
        {
            if (_arraySize != copyLength)
            {
                throw new System.DataMisalignedException("Array size mismatch!");
            }
        }

        /// <summary>
        /// Copies from float data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by float32
        public void CopyFrom(float[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is float type
            if (!_dataType.IsFloat32())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Float type");
            }

            UnmanagedNDArrayWrapper.CopyFloatDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from double data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by float64
        public void CopyFrom(double[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is double type
            if (!_dataType.IsFloat64())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Double type");
            }

            UnmanagedNDArrayWrapper.CopyDoubleDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from int data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by int32
        public void CopyFrom(int[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is int type
            if (!_dataType.IsInt32())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Int type");
            }

            UnmanagedNDArrayWrapper.CopyIntDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from long data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by int64
        public void CopyFrom(long[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is long type
            if (!_dataType.IsInt64())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Long type");
            }

            UnmanagedNDArrayWrapper.CopyLongDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from short data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by int16
        public void CopyFrom(short[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is short type
            if (!_dataType.IsInt16())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Short type");
            }

            UnmanagedNDArrayWrapper.CopyShortDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from char data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by uint16
        public void CopyFrom(char[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is char type
            if (!_dataType.IsUint16())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Char type");
            }

            UnmanagedNDArrayWrapper.CopyCharDataToNDArray(this, from);
        }

        /// <summary>
        /// Copies from byte data.
        /// </summary>
        /// <param name="from">From.</param>
        /// The NDArray type must by int8
        public void CopyFrom(byte[] from)
        {
            // Check whether size matches
            CheckCopySize(from.Length);

            // Check whether the NDArray is byte type
            if (!_dataType.IsInt8())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Byte type");
            }

            UnmanagedNDArrayWrapper.CopyByteDataToNDArray(this, from);
        }

        /// <summary>
        /// Returns NDArray data as float array.
        /// </summary>
        /// <returns>The float array.</returns>
        /// The NDArray type must by float32
        public float [] AsFloatArray()
        {
            // Check whether the NDArray is float type
            if (!_dataType.IsFloat32())
            {
                throw new System.ArrayTypeMismatchException("Do not support Float type");
            }

            return UnmanagedNDArrayWrapper.GetFloatDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as double array.
        /// </summary>
        /// <returns>The double array.</returns>
        /// The NDArray type must by float64
        public double[] AsDoubleArray()
        {
            // Check whether the NDArray is float type
            if (!_dataType.IsFloat64())
            {
                throw new System.ArrayTypeMismatchException("Do not support Double type");
            }

            return UnmanagedNDArrayWrapper.GetDoubleDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as int array.
        /// </summary>
        /// <returns>The int array.</returns>
        /// The NDArray type must by int32
        public int[] AsIntArray()
        {
            // Check whether the NDArray is int type
            if (!_dataType.IsInt32())
            {
                throw new System.ArrayTypeMismatchException("Do not support Int type");
            }

            return UnmanagedNDArrayWrapper.GetIntDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as long array.
        /// </summary>
        /// <returns>The long array.</returns>
        /// The NDArray type must by int64
        public long[] AsLongArray()
        {
            // Check whether the NDArray is long type
            if (!_dataType.IsInt64())
            {
                throw new System.ArrayTypeMismatchException("Do not support Long type");
            }

            return UnmanagedNDArrayWrapper.GetLongDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as short array.
        /// </summary>
        /// <returns>The short array.</returns>
        /// The NDArray type must by int16
        public short[] AsShortArray()
        {
            // Check whether the NDArray is short type
            if (!_dataType.IsInt16())
            {
                throw new System.ArrayTypeMismatchException("Do not support Short type");
            }

            return UnmanagedNDArrayWrapper.GetShortDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as char array.
        /// </summary>
        /// <returns>The char array.</returns>
        /// The NDArray type must by uint16
        public char[] AsCharArray()
        {
            // Check whether the NDArray is char type
            if (!_dataType.IsUint16())
            {
                throw new System.ArrayTypeMismatchException("Do not support Char type");
            }

            return UnmanagedNDArrayWrapper.GetCharDataFromNDArray(this);
        }

        /// <summary>
        /// Returns NDArray data as byte array.
        /// </summary>
        /// <returns>The byte array.</returns>
        /// The NDArray type must by int8
        public byte[] AsByteArray()
        {
            // Check whether the NDArray is byte type
            if (!_dataType.IsInt8())
            {
                throw new System.ArrayTypeMismatchException("Do not support Byte type");
            }

            return UnmanagedNDArrayWrapper.GetByteDataFromNDArray(this);
        }

        private bool _disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (!IntPtr.Zero.Equals(_arrayHandle))
                {
                    UnmanagedNDArrayWrapper.DisposeNDArray(_arrayHandle);
                    _arrayHandle = IntPtr.Zero;
                }
                _internalBuffer = null;

                _disposedValue = true;
            }
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.NDArray"/> is reclaimed by garbage collection.
        /// </summary>
         ~NDArray() {
           // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
           Dispose(false);
         }

        /// <summary>
        /// Releases all resource used by the <see cref="T:TVMRuntime.NDArray"/> object.
        /// </summary>
        /// <remarks>Call <see cref="Dispose"/> when you are finished using the <see cref="T:TVMRuntime.NDArray"/>. The
        /// <see cref="Dispose"/> method leaves the <see cref="T:TVMRuntime.NDArray"/> in an unusable state. After
        /// calling <see cref="Dispose"/>, you must release all references to the <see cref="T:TVMRuntime.NDArray"/> so
        /// the garbage collector can reclaim the memory that the <see cref="T:TVMRuntime.NDArray"/> was occupying.</remarks>
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
