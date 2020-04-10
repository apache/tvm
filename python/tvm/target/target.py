# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Target data structure."""
import warnings
import tvm._ffi

from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object
class Target(Object):
    """Target device information, use through TVM API.

    Note
    ----
    Do not use class constructor, you can create target using the following functions

    - :py:func:`tvm.target.create` create target from string
    - :py:func:`tvm.target.arm_cpu` create arm_cpu target
    - :py:func:`tvm.target.cuda` create CUDA target
    - :py:func:`tvm.target.rocm` create ROCM target
    - :py:func:`tvm.target.mali` create Mali target
    - :py:func:`tvm.target.intel_graphics` create Intel Graphics target
    """
    def __new__(cls):
        # Always override new to enable class
        obj = Object.__new__(cls)
        obj._keys = None
        obj._options = None
        obj._libs = None
        return obj

    @property
    def keys(self):
        if not self._keys:
            self._keys = [str(k) for k in self.keys_array]
        return self._keys

    @property
    def options(self):
        if not self._options:
            self._options = [str(o) for o in self.options_array]
        return self._options

    @property
    def libs(self):
        if not self._libs:
            self._libs = [str(l) for l in self.libs_array]
        return self._libs

    @property
    def model(self):
        for opt in self.options_array:
            if opt.startswith('-model='):
                return opt[7:]
        return 'unknown'

    @property
    def mcpu(self):
        """Returns the mcpu from the target if it exists."""
        mcpu = ''
        if self.options is not None:
            for opt in self.options:
                if 'mcpu' in opt:
                    mcpu = opt.split('=')[1]
        return mcpu

    def __enter__(self):
        _ffi_api.EnterTargetScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_api.ExitTargetScope(self)

    @staticmethod
    def current(allow_none=True):
        """Returns the current target.

        Parameters
        ----------
        allow_none : bool
            Whether allow the current target to be none

        Raises
        ------
        ValueError if current target is not set.
        """
        return _ffi_api.GetCurrentTarget(allow_none)


def _merge_opts(opts, new_opts):
    """Helper function to merge options"""
    if isinstance(new_opts, str):
        new_opts = new_opts.split()
    if new_opts:
        opt_set = set(opts)
        new_opts = [opt for opt in new_opts if opt not in opt_set]
        return opts + new_opts
    return opts


def cuda(model='unknown', options=None):
    """Returns a cuda target.

    Parameters
    ----------
    model: str
        The model of cuda device (e.g. 1080ti)
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(['-model=%s' % model], options)
    return _ffi_api.TargetCreate("cuda", *opts)


def rocm(model='unknown', options=None):
    """Returns a ROCM target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(["-model=%s" % model], options)
    return _ffi_api.TargetCreate("rocm", *opts)


def mali(model='unknown', options=None):
    """Returns a ARM Mali GPU target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=mali", '-model=%s' % model]
    opts = _merge_opts(opts, options)
    return _ffi_api.TargetCreate("opencl", *opts)


def intel_graphics(model='unknown', options=None):
    """Returns an Intel Graphics target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=intel_graphics", '-model=%s' % model]
    opts = _merge_opts(opts, options)
    return _ffi_api.TargetCreate("opencl", *opts)


def opengl(model='unknown', options=None):
    """Returns a OpenGL target.

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(["-model=%s" % model], options)
    return _ffi_api.TargetCreate("opengl", *opts)


def arm_cpu(model='unknown', options=None):
    """Returns a ARM CPU target.
    This function will also download pre-tuned op parameters when there is none.

    Parameters
    ----------
    model: str
        SoC name or phone name of the arm board.
    options : str or list of str
        Additional options
    """
    trans_table = {
        "pixel2":    ["-model=snapdragon835", "-target=arm64-linux-android -mattr=+neon"],
        "mate10":    ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "mate10pro": ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "p20":       ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "p20pro":    ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "rasp3b":    ["-model=bcm2837", "-target=armv7l-linux-gnueabihf -mattr=+neon"],
        "rasp4b":    ["-model=bcm2711", "-target=arm-linux-gnueabihf -mattr=+neon"],
        "rk3399":    ["-model=rk3399", "-target=aarch64-linux-gnu -mattr=+neon"],
        "pynq":      ["-model=pynq", "-target=armv7a-linux-eabi -mattr=+neon"],
        "ultra96":   ["-model=ultra96", "-target=aarch64-linux-gnu -mattr=+neon"],
    }
    pre_defined_opt = trans_table.get(model, ["-model=%s" % model])

    opts = ["-device=arm_cpu"] + pre_defined_opt
    opts = _merge_opts(opts, options)
    return _ffi_api.TargetCreate("llvm", *opts)


def rasp(options=None):
    """Return a Raspberry 3b target.

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    warnings.warn('tvm.target.rasp() is going to be deprecated. '
                  'Please use tvm.target.arm_cpu("rasp3b")')
    return arm_cpu('rasp3b', options)


def vta(model='unknown', options=None):
    opts = ["-device=vta", '-keys=cpu', '-model=%s' % model]
    opts = _merge_opts(opts, options)
    ret = _ffi_api.TargetCreate("ext_dev", *opts)
    return ret


def bifrost(model='unknown', options=None):
    """Return an ARM Mali GPU target (Bifrost architecture).

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    opts = ["-device=bifrost", '-model=%s' % model]
    opts = _merge_opts(opts, options)
    return _ffi_api.TargetCreate("opencl", *opts)


def hexagon(cpu_ver='v66', sim_args=None, hvx=128):
    """Returns a Hexagon target.

    Parameters
    ----------
    cpu_ver : str
        CPU version used for code generation. Not all allowed cpu str
        will be valid, LLVM will throw an error.
    sim_args : str or list of str
        User defined sim arguments. CPU version defaults to cpu_ver.
        Otherwise, separate versions are used for codegen and sim. Not
        all allowed cpu strings will be valid, simulator will throw an
        error if invalid. Does not affect codegen.
    hvx : int
        Size of hvx register. Value of 0 indicates disabled hvx.
    """
    # Example compiler arguments
    # llvm -target=hexagon -mcpu=hexagonv66 -mattr=+hvxv66,+hvx-length128b

    # Check for valid codegen cpu
    valid_hex = ['v60', 'v62', 'v65', 'v66', 'v67', 'v67t']
    try:
        cpu_ver = cpu_ver[cpu_ver.index('v'):].lower()
        assert 3 <= len(cpu_ver) <= 4
    except:
        msg = '{} is not a valid Hexagon version\nvalid versions include {}'
        raise ValueError(msg.format(cpu_ver, valid_hex)) from None

    assert hvx in [0, 64, 128]

    # Target string
    def create_target(cpu_ver):
        target = ' -target=hexagon'
        mcpu = ' -mcpu=hexagon' + cpu_ver
        mattr = ''
        # HVX enable
        if hvx:
            mattr = ' -mattr=+hvx' + cpu_ver + ',+hvx-length' + str(hvx) + 'b'
        return 'llvm' + target + mcpu + mattr

    # Simulator string
    def create_sim(cpu_ver, sim_args):
        def validate_hvx_length(codegen_hvx, sim_args):
            if sim_args and '--hvx_length' in sim_args:
                # If --hvx_length was specified, check HVX length of sim
                # vs codegen
                i = sim_args.index('hvx_length') + len('hvx_length') + 1
                sim_hvx = sim_args[i:i+3]
                if sim_hvx != str(codegen_hvx):
                    print('WARNING: sim hvx {} and codegen hvx {} mismatch!' \
                          .format(sim_hvx, codegen_hvx))
            elif codegen_hvx != 0:
                # If --hvx_length was not given, add it if HVX is enabled
                sim_args = sim_args + ' ' if isinstance(sim_args, str) else ''
                sim_args += '--hvx_length ' + str(codegen_hvx)
            return sim_args or ''

        if not sim_args:
            return cpu_ver + ' ' + validate_hvx_length(hvx, sim_args)

        sim_cpu = cpu_ver + ' '

        # Add user defined args
        if isinstance(sim_args, list):
            sim_args = ' '.join(sim_args)

        # Check for supplied sim cpu version
        if 'v6' in sim_args:
            sim_cpu = ''

            # Regex match for allowed cpus
            valid_cpu_str_regex = r'(?P<pre>--.*\s)?(--m)?' +                 \
                r'(?P<base_version>v6[25678])(?P<sub_version>[a-z])?' +       \
                r'(?P<l2_size>_[0-9]+)?(?P<rev>_rev[0-9])?\s?(?P<post>--.*)?'
            m = re.match(valid_cpu_str_regex, sim_args.lower())
            if not m:
                raise ValueError(
                    'Invalid simulator argument string "{}"'.format(sim_args))

            # Parse options into correct order
            cpu_attr = {x: str(m.groupdict()[x] or '') for x in m.groupdict()}
            sim_args = cpu_attr['base_version'] +  \
                       cpu_attr['sub_version']  +  \
                       cpu_attr['l2_size'] +       \
                       cpu_attr['rev'] + ' ' +     \
                       cpu_attr['pre'] + cpu_attr['post']

        return sim_cpu + ' ' + validate_hvx_length(hvx, sim_args)

    # Sim args
    os.environ['HEXAGON_SIM_ARGS'] = create_sim(cpu_ver, sim_args)

    target_str = create_target(cpu_ver)
    args_list = target_str.split()
    return _ffi_api.TargetCreate("hexagon", *args_list)


def create(target_str):
    """Get a target given target string.

    Parameters
    ----------
    target_str : str
        The target string.

    Returns
    -------
    target : Target
        The target object

    Note
    ----
    See the note on :py:mod:`tvm.target` on target string format.
    """
    if isinstance(target_str, Target):
        return target_str
    if not isinstance(target_str, str):
        raise ValueError("target_str has to be string type")

    return _ffi_api.TargetFromString(target_str)
