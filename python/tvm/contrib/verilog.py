"""Verilog simulator modules."""
from __future__ import absolute_import

import subprocess
import sys
import os
import ctypes

from .. import _api_internal
from .._ffi.base import string_types
from .._ffi.node import NodeBase, register_node
from .._ffi.function import register_func
from . import util

@register_node
class VPISession(NodeBase):
    """Verilog session"""
    def __init__(self, handle):
        super(VPISession, self).__init__(handle)
        self.proc = None
        self.execpath = None
        self.yield_callbacks = []

    def __del__(self):
        self.proc.kill()
        try:
            super(VPISession, self).__del__()
        except AttributeError:
            pass

    def arg(self, index):
        """Get handle passed to host session.

        Parameters
        ----------
        index : int
            The index value.

        Returns
        -------
        handle : VPIHandle
            The handle
        """
        return _api_internal._vpi_SessGetArg(self, index)

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("have to be string types")
        return _api_internal._vpi_SessGetHandleByName(self, name)

    def __getattr__(self, name):
        return _api_internal._vpi_SessGetHandleByName(self, name)

    def yield_until_next_cycle(self):
        """Yield until next posedge"""
        for f in self.yield_callbacks:
            f()
        return _api_internal._vpi_SessYield(self)

    def shutdown(self):
        """Shutdown the simulator"""
        return _api_internal._vpi_SessShutdown(self)


@register_node
class VPIHandle(NodeBase):
    """Handle to a verilog variable."""
    def __init__(self, handle):
        super(VPIHandle, self).__init__(handle)
        self._name = None
        self._size = None

    def get_int(self):
        """Get integer value from handle.

        Returns
        -------
        value : int
        """
        return _api_internal._vpi_HandleGetInt(self)

    def put_int(self, value):
        """Put integer value to handle.

        Parameters
        ----------
        value : int
            The value to put
        """
        return _api_internal._vpi_HandlePutInt(self, value)

    @property
    def name(self):
        if self._name is None:
            self._name = _api_internal._vpi_HandleGetName(self)
        return self._name

    @property
    def size(self):
        if self._size is None:
            self._size = _api_internal._vpi_HandleGetSize(self)
        return self._size

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("have to be string types")
        return _api_internal._vpi_HandleGetHandleByName(self, name)

    def __getattr__(self, name):
        return _api_internal._vpi_HandleGetHandleByName(self, name)


def _find_vpi_path():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../../../lib/')
    vpi_path = [curr_path, api_path]
    vpi_path = [os.path.join(p, 'tvm_vpi.vpi') for p in vpi_path]
    vpi_found = [p for p in vpi_path if os.path.exists(p) and os.path.isfile(p)]
    if vpi_found:
        return os.path.dirname(vpi_found[0])
    else:
        raise ValueError("Cannot find tvm_vpi.vpi, make sure you did `make verilog`")

def search_path():
    """Get the search directory."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    ver_path = [os.path.join(curr_path, '../../../verilog/')]
    ver_path += [os.path.join(curr_path, '../../../tests/verilog/unittest/')]
    ver_path += [os.path.join(curr_path, '../../../tests/verilog/integration/')]
    return ver_path


def find_file(file_name):
    """Find file in the search directories.

    Parameters
    ----------
    file_name : str
        The file name

    Return
    ------
    file_name : str
        The absolute path to the file, raise Error if cannot find it.
    """
    ver_path = search_path()
    flist = [os.path.join(p, file_name) for p in ver_path]
    found = [p for p in flist if os.path.exists(p) and os.path.isfile(p)]
    if not found:
        raise ValueError("Cannot find %s in %s" % (file_name, flist))
    return found[0]


def compile_file(file_name, file_target, options=None):
    """Compile verilog via iverilog

    Parameters
    ----------
    file_name : str or list of str
        The cuda code.

    file_target : str
        The target file.
    """
    cmd = ["iverilog"]
    for path in search_path():
        cmd += ["-I%s" % path]

    cmd += ["-o", file_target]
    if options:
        cmd += options

    if isinstance(file_name, string_types):
        file_name = [file_name]
    cmd += file_name
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        raise ValueError("Compilation error:\n%s" % out)


def session(file_names, codes=None):
    """Create a new iverilog session by compile the file.

    Parameters
    ----------
    file_names : str or list of str
        The name of the file

    codes : str or list of str
        The code in str.

    Returns
    -------
    sess : VPISession
        The created session.
    """
    if isinstance(file_names, string_types):
        file_names = [file_names]
    path = util.tempdir()

    if codes:
        if isinstance(codes, (list, tuple)):
            codes = '\n'.join(codes)
        fcode = path.relpath("temp_code.v")
        with open(fcode, "w") as out_file:
            out_file.write(codes)
        file_names.append(fcode)

    for name in file_names:
        if not os.path.exists(name):
            raise ValueError("Cannot find file %s" % name)

    target = path.relpath(os.path.basename(file_names[0].rsplit(".", 1)[0]))
    compile_file(file_names, target)
    vpi_path = _find_vpi_path()

    cmd = ["vvp"]
    cmd += ["-M", vpi_path]
    cmd += ["-m", "tvm_vpi"]
    cmd += [target]
    env = os.environ.copy()

    read_device, write_host = os.pipe()
    read_host, write_device = os.pipe()

    if sys.platform == "win32":
        import msvcrt
        env['TVM_DREAD_PIPE'] = str(msvcrt.get_osfhandle(read_device))
        env['TVM_DWRITE_PIPE'] = str(msvcrt.get_osfhandle(write_device))
        read_host = msvcrt.get_osfhandle(read_host)
        write_host = msvcrt.get_osfhandle(write_host)
    else:
        env['TVM_DREAD_PIPE'] = str(read_device)
        env['TVM_DWRITE_PIPE'] = str(write_device)

    env['TVM_HREAD_PIPE'] = str(read_host)
    env['TVM_HWRITE_PIPE'] = str(write_host)

    try:
        # close_fds does not work well for all python3
        # Use pass_fds instead.
        # pylint: disable=unexpected-keyword-arg
        pass_fds = (read_device, write_device, read_host, write_host)
        proc = subprocess.Popen(cmd, pass_fds=pass_fds, env=env)
    except TypeError:
        # This is effective for python2
        proc = subprocess.Popen(cmd, close_fds=False, env=env)

    # close device side pipe
    os.close(read_device)
    os.close(write_device)

    sess = _api_internal._vpi_SessMake(read_host, write_host)
    sess.proc = proc
    sess.execpath = path
    return sess


@register_func
def tvm_callback_verilog_simulator(code, *args):
    """Callback by TVM runtime to invoke verilog simulator

    Parameters
    ----------
    code : str
        The verilog code to be simulated

    args : list
        Additional arguments to be set.
    """
    libs = [
        find_file("tvm_vpi_mmap.v")
    ]
    sess = session(libs, code)
    for i, value in enumerate(args):
        vpi_h = sess.main["tvm_arg%d" % i]
        if isinstance(value, ctypes.c_void_p):
            int_value = int(value.value)
        elif isinstance(value, int):
            int_value = value
        else:
            raise ValueError(
                "Do not know how to handle value type %s" % type(value))
        vpi_h.put_int(int_value)

    rst = sess.main.rst
    done = sess.main.done
    # start driving
    rst.put_int(1)
    sess.yield_until_next_cycle()
    rst.put_int(0)
    sess.yield_until_next_cycle()
    while not done.get_int():
        sess.yield_until_next_cycle()
    sess.yield_until_next_cycle()
    sess.shutdown()
