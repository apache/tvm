"""Util to compile with NVCC"""
import os
import sys
import tempfile
import subprocess

def compile_source(code, target="cubin", options=None):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target : str
        The target format

    options : str
        The additional options

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    temp_dir = tempfile.mkdtemp()
    if target not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target must be in cubin, ptx, fatbin")
    path_code = os.path.join(temp_dir, "my_kernel.cu")
    path_target = os.path.join(temp_dir, "my_kernel.%s" % target)

    with open(path_code, "w") as out_file:
        out_file.write(code)

    cmd = ["nvcc"]
    cmd += ["--%s" % target, "-O3"]
    cmd += ["-o", path_target]
    if options:
        cmd += options
    cmd += [path_code]
    args = ' '.join(cmd)

    proc = subprocess.Popen(
        args, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(out)
        sys.stderr.flush()
        cubin = None
    else:
        cubin = bytearray(open(path_target, "rb").read())
    os.remove(path_code)
    if os.path.exists(path_target):
        os.remove(path_target)
    os.rmdir(temp_dir)
    return cubin
