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
        UIntPtr arrayHandle = UIntPtr.Zero;
        int ndim = 0;
        int[] shape;
        int arraySize = 0;

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
            UnmanagedNDArrayWrapper.CreateNDArray(shape, ndim, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.device_type,
                 ctx.device_id, ref arrayHandle);
            shape = shapeInp;
            ndim = ndimInp;
            arraySize = 1;
            Array.ForEach(shape, i => arraySize *= i);
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
            UnmanagedNDArrayWrapper.CreateNDArray(shape, ndim, dataType.code,
                dataType.bits, dataType.lanes, (int)ctx.device_type,
                 ctx.device_id, ref arrayHandle);
            shape = shapeInp;
            ndim = ndimInp;
            arraySize = 1;
            Array.ForEach(shape, i => arraySize *= i);
        }

        /// <summary>
        /// Gets the ND Array handle.
        /// </summary>
        /// <value>The NDA rray handle.</value>
        public UIntPtr NDArrayHandle { get => arrayHandle; }

        /// <summary>
        /// Gets or sets the <see cref="T:TVMRuntime.NDArray"/> with the specified i.
        /// </summary>
        /// <param name="i">The index.</param>
        public object this[int i]
        {
            get { return UnmanagedNDArrayWrapper.GetNDArrayElem(arrayHandle, i, arraySize); }
            set { UnmanagedNDArrayWrapper.SetNDArrayElem(arrayHandle, i, value, arraySize); }
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="T:TVMRuntime.NDArray"/> is reclaimed by garbage collection.
        /// </summary>
        ~NDArray()
        {
            UnmanagedNDArrayWrapper.DisposeNDArray(arrayHandle);
        }
    }
}
