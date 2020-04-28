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
        /// Initializes a new instance of the <see cref="T:TVMRuntime.TVMDataType"/> struct.
        /// </summary>
        /// <param name="dtype">Dtype.</param>
        public TVMDataType(string dtype) : this()
        {
            string lowerDtype = dtype.ToLower();

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
            else
            {
                Console.WriteLine(dtype + " not supported!");
            }
        }
    }

    public class NDArray
    {
        private IntPtr _arrayHandle = IntPtr.Zero;
        private int _ndim = 0;
        private int[] _shape;
        private int _arraySize = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        public NDArray()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="ndimInp">Ndim inp.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="ctx">Context.</param>
        public NDArray(int[] shapeInp,
                          int ndimInp,
                          TVMDataType dataType,
                          TVMContext ctx)
        {
            UnmanagedNDArrayWrapper.CreateNDArray(shapeInp, ndimInp, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId, ref _arrayHandle);
            _shape = shapeInp;
            _ndim = ndimInp;
            _arraySize = 1;
            Array.ForEach(_shape, i => _arraySize *= i);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.NDArray"/> class.
        /// </summary>
        /// <param name="shapeInp">Shape inp.</param>
        /// <param name="ndimInp">Ndim inp.</param>
        /// <param name="dataTypeStr">Data type string.</param>
        /// <param name="ctx">Context.</param>
        public NDArray(int[] shapeInp,
                          int ndimInp,
                          string dataTypeStr,
                          TVMContext ctx)
        {
            TVMDataType dataType = new TVMDataType(dataTypeStr);
            UnmanagedNDArrayWrapper.CreateNDArray(shapeInp, ndimInp, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.deviceType,
                 ctx.deviceId, ref _arrayHandle);
            _shape = shapeInp;
            _ndim = ndimInp;
            _arraySize = 1;
            Array.ForEach(_shape, i => _arraySize *= i);
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
            }
        }

        /// <summary>
        /// Gets the shape.
        /// </summary>
        /// <value>The shape.</value>
        public int [] Shape { get => _shape; }

        /// <summary>
        /// Gets the ndim.
        /// </summary>
        /// <value>The ndim.</value>
        public int Ndim { get => _ndim; }

        /// <summary>
        /// Gets the size.
        /// </summary>
        /// <value>The size.</value>
        public int Size { get => _arraySize; }

        /// <summary>
        /// Gets or sets the <see cref="T:TVMRuntime.NDArray"/> with the specified i.
        /// </summary>
        /// <param name="i">The index.</param>
        public object this[int i]
        {
            get { return UnmanagedNDArrayWrapper.GetNDArrayElem(_arrayHandle, i, _arraySize); }
            set { UnmanagedNDArrayWrapper.SetNDArrayElem(_arrayHandle, i, value, _arraySize); }
        }

        /// <summary>
        /// Disposes the NDArray.
        /// </summary>
        public void DisposeNDArray()
        {
            if (!IntPtr.Zero.Equals(_arrayHandle))
            {
                UnmanagedNDArrayWrapper.DisposeNDArray(_arrayHandle);
                _arrayHandle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.NDArray"/> is reclaimed by garbage collection.
        /// </summary>
        ~NDArray()
        {
            DisposeNDArray();
        }
    }
}
