"""uTVM module for bare-metal backends.

uTVM (or the micro backend) enables provides support for bare-metal devices.
Its targets currently include a host-emulated device which is used for testing,
and JTAG-based openocd device which allows actual interfacing with microdevices.
"""

from ..contrib import binutil
from .base import Session, cross_compiler, create_micro_lib
