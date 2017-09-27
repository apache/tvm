# Install OpenCL runtime in nvidia docker.
apt-get install -y --no-install-recommends --force-yes \
        ocl-icd-libopencl1 \
        clinfo && \
        rm -rf /var/lib/apt/lists/*

mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
