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
        public DeviceType device_type;

        /// <summary>
        /// The device identifier.
        /// </summary>
        public int device_id;

        public TVMContext(int dev_id) : this()
        {
            device_type = DeviceType.CPU;
            device_id = dev_id;
        }

        public TVMContext(DeviceType dev_type, int dev_id) : this()
        {
            device_type = dev_type;
            device_id = dev_id;
        }
    }
}
