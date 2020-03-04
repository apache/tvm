#include <dmlc/logging.h>
#include <vta/hw_spec.h>
#include "intelfocl_device.h"
#include "AOCLUtils/aocl_utils.h"

#define MEM_ALIGNMENT (1024)

#define CL_STATUS_SUCCESS(x) ((x) == CL_SUCCESS)

void cleanup() {}

int IntelFOCLDevice::init(size_t mem_size, std::string aocx_file)
{
    cl_int status;
    cl_device_id device;
    cl_platform_id platform;
    unsigned int argi;
    bool focl_device_avail;
    unsigned int num_devices;
    aocl_utils::scoped_array<cl_device_id> devices;

    platform = aocl_utils::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    CHECK(platform) << "Unable to find Intel(R) FPGA OpenCL platform";
    
    devices.reset(aocl_utils::getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    focl_device_avail = false;
    for ( unsigned int i = 0; i < num_devices; i ++ )
    {
        device = devices[i];
        _context = clCreateContext(NULL, 1, &device, &aocl_utils::oclContextCallback, NULL, &status);
        if ( CL_STATUS_SUCCESS(status) )
        {
            focl_device_avail = true;
            LOG(INFO) << "Using device: " << aocl_utils::getDeviceName(device);
            break;
        }
    }
    CHECK(focl_device_avail) << "No FPGA device available";
    num_devices = 1;

    LOG(INFO) << "Using AOCX: " << aocx_file;
    _program = aocl_utils::createProgramFromBinary(_context, aocx_file.c_str(), &device, num_devices);
    status = clBuildProgram(_program, 0, NULL, "", NULL, NULL);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to build program";

    for ( unsigned int i = 0; i < KERNEL_PROFILE; i++ )
    {
        _kernels[i] = clCreateKernel(_program, kernel_names[i].c_str(), &status);
        CHECK(CL_STATUS_SUCCESS(status)) << "Failed to create kernel";
        _queues[i] = clCreateCommandQueue(_context, device, 0, &status);
        CHECK(CL_STATUS_SUCCESS(status)) << "Failed to create command queue";
    }

    _mem = clCreateBuffer(_context, CL_MEM_READ_WRITE, mem_size, NULL, &status);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to create buffer mem";
    mem_chunk_t init_chunk = {.offset = 0, .size = mem_size, .occupied = false};
    _mem_chunks.push_back(init_chunk);

    argi = 1;
    status = clSetKernelArg(_kernels[KERNEL_FETCH], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    argi = 0;
    status = clSetKernelArg(_kernels[KERNEL_COMPUTE], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    status = clSetKernelArg(_kernels[KERNEL_COMPUTE], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    status = clSetKernelArg(_kernels[KERNEL_COMPUTE], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    status = clSetKernelArg(_kernels[KERNEL_COMPUTE], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    status = clSetKernelArg(_kernels[KERNEL_COMPUTE], argi++, sizeof(cl_mem), &_mem);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;

    return 0;
}

ifocl_mem_off_t IntelFOCLDevice::alloc(size_t size)
{
    auto iter = _mem_chunks.begin();
    size_t aligned_size = ((size + MEM_ALIGNMENT - 1) / MEM_ALIGNMENT) * MEM_ALIGNMENT;

    while ( iter != _mem_chunks.end() && (iter->occupied || (iter->size < aligned_size)) )
    {
        iter++;
    }

    if ( iter == _mem_chunks.end() ) return IFOCL_MEM_OFF_ERR;

    iter->occupied = true;
    if ( iter->size != aligned_size )
    {
        mem_chunk_t rem = {iter->offset + aligned_size, iter->size - aligned_size, false};
        iter->size = aligned_size;
        _mem_chunks.insert(std::next(iter), rem);
    }

    return iter->offset;
}

void IntelFOCLDevice::free(ifocl_mem_off_t offset)
{
    auto iter = _mem_chunks.begin();
    while ( iter != _mem_chunks.end() && iter->offset < offset ) iter++;

    if ( iter == _mem_chunks.end() || iter->offset != offset || !iter->occupied )
    {
        return;
    }

    iter->occupied = false;
    if ( iter != _mem_chunks.begin() && !std::prev(iter)->occupied ) iter--;

    while ( std::next(iter) != _mem_chunks.end() && !std::next(iter)->occupied )
    {
        iter->size += std::next(iter)->size;
        _mem_chunks.erase(std::next(iter));
    }
}


void IntelFOCLDevice::write_mem(ifocl_mem_off_t offset, const void *buf, size_t nbyte)
{
    cl_int status = clEnqueueWriteBuffer(_queues[0], _mem, CL_TRUE, offset, nbyte, buf, 0, NULL, NULL);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to enqueue write buffer";
}

void IntelFOCLDevice::read_mem(ifocl_mem_off_t offset, void *buf, size_t nbyte)
{
    cl_int status = clEnqueueReadBuffer(_queues[0], _mem, CL_TRUE, offset, nbyte, buf, 0, NULL, NULL);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to enqueue read buffer";
};

int IntelFOCLDevice::execute_instructions(ifocl_mem_off_t offset, size_t count)
{
    cl_int status;
    unsigned int argi;
    unsigned int insn_offset = offset / VTA_INS_ELEM_BYTES;
    unsigned int insn_count = count;
    const size_t global_work_size = 1;

    argi = 0;
    status = clSetKernelArg(_kernels[KERNEL_FETCH], argi, sizeof(unsigned int), &insn_count);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;
    argi = 2;
    status = clSetKernelArg(_kernels[KERNEL_FETCH], argi, sizeof(unsigned int), &insn_offset);
    CHECK(CL_STATUS_SUCCESS(status)) << "Failed to set argument " << argi;

    for ( unsigned int i = 0; i < KERNEL_PROFILE; i++ )
    {
        status = clEnqueueNDRangeKernel(_queues[i], _kernels[i], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        CHECK(CL_STATUS_SUCCESS(status)) << "Failed to enqueue kernel";
    }

    for ( unsigned int i = 0; i < KERNEL_PROFILE; i++ )
    {
        status = clFinish(_queues[i]);
        CHECK(CL_STATUS_SUCCESS(status)) << "Failed to clFinish";
    }

    return 0;
};

void IntelFOCLDevice::deinit()
{
    for ( unsigned int i = 0; i < NUM_OCL_KERNELS; i++ )
    {
        clReleaseKernel(_kernels[i]);
        clReleaseCommandQueue(_queues[i]);
    }

    clReleaseMemObject(_mem);

    clReleaseProgram(_program);

    clReleaseContext(_context);
}

IntelFOCLDevice::~IntelFOCLDevice()
{
    deinit();
}
