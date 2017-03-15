"""Information about nnvm."""
from __future__ import absolute_import

import subprocess
import sys
import os

from .. import _api_internal
from .._base import string_types
from .._ctypes._node import NodeBase, register_node
from . import testing

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
        super(VPISession, self).__del__()

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

    def yield_until_posedge(self):
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
    ver_path += [os.path.join(curr_path, '../../../tests/verilog/')]
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
    if len(found):
        return found[0]
    else:
        raise ValueError("Cannot find %s in %s" % (file_name, flist))


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


def session(file_name):
    """Create a new iverilog session by compile the file.

    Parameters
    ----------
    file_name : str or list of str
        The name of the file

    Returns
    -------
    sess : VPISession
        The created session.
    """
    if isinstance(file_name, string_types):
        file_name = [file_name]

    for name in file_name:
        if not os.path.exists(name):
            raise ValueError("Cannot find file %s" % name)

    path = testing.tempdir()
    target = path.relpath(os.path.basename(file_name[0].rsplit(".", 1)[0]))
    compile_file(file_name, target)
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
