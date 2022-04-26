import warnings

from .._ffi.base import string_types
from .._ffi.runtime_ctypes import Device


# function exposures


def device(dev_type, dev_id=0):
    """Construct a TVM device with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev: tvm.runtime.Device
        The corresponding device.

    Examples
    --------
    Device can be used to create reflection of device by
    string representation of the device type.

    .. code-block:: python

      assert tvm.device("cpu", 1) == tvm.cpu(1)
      assert tvm.device("cuda", 0) == tvm.cuda(0)
    """
    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type not in Device.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = Device.STR2MASK[dev_type]
    return Device(dev_type, dev_id)


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(1, dev_id)


def cuda(dev_id=0):
    """Construct a CUDA GPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(2, dev_id)


def gpu(dev_id=0):
    """Construct a CUDA GPU device

        deprecated:: 0.9.0
        Use :py:func:`tvm.cuda` instead.

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    warnings.warn(
        "Please use tvm.cuda() instead of tvm.gpu(). tvm.gpu() is going to be deprecated in 0.9.0",
    )
    return Device(2, dev_id)


def rocm(dev_id=0):
    """Construct a ROCM device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(10, dev_id)


def opencl(dev_id=0):
    """Construct a OpenCL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(4, dev_id)


def metal(dev_id=0):
    """Construct a metal device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(8, dev_id)


def vpi(dev_id=0):
    """Construct a VPI simulated device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(9, dev_id)


def vulkan(dev_id=0):
    """Construct a Vulkan device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(7, dev_id)


def ext_dev(dev_id=0):
    """Construct a extension device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device

    Note
    ----
    This API is reserved for quick testing of new
    device by plugin device API as ext_dev.
    """
    return Device(12, dev_id)


def hexagon(dev_id=0):
    """Construct a Hexagon device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(14, dev_id)


def webgpu(dev_id=0):
    """Construct a webgpu device.

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return Device(15, dev_id)


cl = opencl
mtl = metal