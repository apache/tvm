using System;


namespace TVMRuntime
{
    public enum DeviceType
    {
        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief CPU device
        /// </remarks>
        CPU = 1,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief CUDA GPU device
        /// </remarks>
        GPU = 2,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief Pinned CUDA GPU device by cudaMallocHost@note kDLCPUPinned = kDLCPU | kDLGPU
        /// </remarks>
        CPUPinned = 3,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief OpenCL devices.
        /// </remarks>
        OpenCL = 4,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief Vulkan buffer for next generation graphics.
        /// </remarks>
        Vulkan = 5,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief Metal for Apple GPU.
        /// </remarks>
        Metal = 6,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief Verilog simulator buffer
        /// </remarks>
        VPI = 7,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief ROCm GPUs for AMD GPUs
        /// </remarks>
        ROCM = 8,

        /// <summary>
        ///
        /// </summary>
        /// <remarks>
        /// @brief Reserved extension device type,
        /// used for quickly test extension device
        /// The semantics can differ depending on the implementation.
        /// </remarks>
        ExtDev = 9,
    };

    /// <summary>
    /// Tvm context.
    /// </summary>
    public struct TVMContext
    {
        /// <summary>
        /// The type of the device.
        /// </summary>
        public DeviceType deviceType;

        /// <summary>
        /// The device identifier.
        /// </summary>
        public int deviceId;

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.TVMContext"/> struct.
        /// </summary>
        /// <param name="devId">Dev identifier.</param>
        public TVMContext(int devId) : this()
        {
            deviceType = DeviceType.CPU;
            deviceId = devId;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:TVMRuntime.TVMContext"/> struct.
        /// </summary>
        /// <param name="devType">Dev type.</param>
        /// <param name="devId">Dev identifier.</param>
        public TVMContext(DeviceType devType, int devId) : this()
        {
            deviceType = devType;
            deviceId = devId;
        }
    }
}
