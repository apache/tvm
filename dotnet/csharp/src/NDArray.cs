using System;

namespace TVMRuntime
{
    public class NDArray : NDArrayBase, IDisposable
    {
        private TVMContext _ctx;
        private TVMDataType _dataType;

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
            CreateNDArray(shapeInp, shapeInp.Length, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId);
            DataType = dataType;
            Ctx = ctx;
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
            CreateNDArray(shapeInp, shapeInp.Length, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId);
            DataType = dataType;
            Ctx = ctx;
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
        /// <value>The NDA rray handle.</value>
        public new IntPtr NDArrayHandle
        {
            get => base.NDArrayHandle; set
            {
                base.NDArrayHandle = value;
                GetNDArrayDtype(base.NDArrayHandle, ref _dataType);
                GetNDArrayCtx(base.NDArrayHandle, ref _ctx);
            }
        }

        /// <summary>
        /// Gets or sets the context.
        /// </summary>
        /// <value>The context.</value>
        public TVMContext Ctx { get => _ctx; set => _ctx = value; }

        /// <summary>
        /// Gets or sets the type of the data.
        /// </summary>
        /// <value>The type of the data.</value>
        public TVMDataType DataType { get => _dataType; set => _dataType = value; }

        /// <summary>
        /// Checks the size of the copy.
        /// </summary>
        private void CheckCopySize(int copyLength)
        {
            if (Size != copyLength)
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
            if (!DataType.IsFloat32())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Float type");
            }

            CopyFloatDataToNDArray(from);
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
            if (!DataType.IsFloat64())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Double type");
            }

            CopyDoubleDataToNDArray(from);
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
            if (!DataType.IsInt32())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Int type");
            }

            CopyIntDataToNDArray(from);
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
            if (!DataType.IsInt64())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Long type");
            }

            CopyLongDataToNDArray(from);
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
            if (!DataType.IsInt16())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Short type");
            }

            CopyShortDataToNDArray(from);
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
            if (!DataType.IsUint16())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Char type");
            }

            CopyCharDataToNDArray(from);
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
            if (!DataType.IsInt8())
            {
                throw new System.ArrayTypeMismatchException("Do not accept Byte type");
            }

            CopyByteDataToNDArray(from);
        }

        /// <summary>
        /// Returns NDArray data as float array.
        /// </summary>
        /// <returns>The float array.</returns>
        /// The NDArray type must by float32
        public float[] AsFloatArray()
        {
            // Check whether the NDArray is float type
            if (!DataType.IsFloat32())
            {
                throw new System.ArrayTypeMismatchException("Do not support Float type");
            }

            return GetFloatDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as double array.
        /// </summary>
        /// <returns>The double array.</returns>
        /// The NDArray type must by float64
        public double[] AsDoubleArray()
        {
            // Check whether the NDArray is float type
            if (!DataType.IsFloat64())
            {
                throw new System.ArrayTypeMismatchException("Do not support Double type");
            }

            return GetDoubleDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as int array.
        /// </summary>
        /// <returns>The int array.</returns>
        /// The NDArray type must by int32
        public int[] AsIntArray()
        {
            // Check whether the NDArray is int type
            if (!DataType.IsInt32())
            {
                throw new System.ArrayTypeMismatchException("Do not support Int type");
            }

            return GetIntDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as long array.
        /// </summary>
        /// <returns>The long array.</returns>
        /// The NDArray type must by int64
        public long[] AsLongArray()
        {
            // Check whether the NDArray is long type
            if (!DataType.IsInt64())
            {
                throw new System.ArrayTypeMismatchException("Do not support Long type");
            }

            return GetLongDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as short array.
        /// </summary>
        /// <returns>The short array.</returns>
        /// The NDArray type must by int16
        public short[] AsShortArray()
        {
            // Check whether the NDArray is short type
            if (!DataType.IsInt16())
            {
                throw new System.ArrayTypeMismatchException("Do not support Short type");
            }

            return GetShortDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as char array.
        /// </summary>
        /// <returns>The char array.</returns>
        /// The NDArray type must by uint16
        public char[] AsCharArray()
        {
            // Check whether the NDArray is char type
            if (!DataType.IsUint16())
            {
                throw new System.ArrayTypeMismatchException("Do not support Char type");
            }

            return GetCharDataFromNDArray();
        }

        /// <summary>
        /// Returns NDArray data as byte array.
        /// </summary>
        /// <returns>The byte array.</returns>
        /// The NDArray type must by int8
        public byte[] AsByteArray()
        {
            // Check whether the NDArray is byte type
            if (!DataType.IsInt8())
            {
                throw new System.ArrayTypeMismatchException("Do not support Byte type");
            }

            return GetByteDataFromNDArray();
        }

        private bool _disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                DisposeNDArray();
                _disposedValue = true;
            }
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.NDArray"/> is reclaimed by garbage collection.
        /// </summary>
        ~NDArray()
        {
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
