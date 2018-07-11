"""Utilith for CUDA backend"""

def parse_cc(compute_capability):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_capability : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_cc = compute_capability.split('.')
    if len(split_cc) == 2 and split_cc[0].isdigit and split_cc[1].isdigit:
        major = int(split_cc[0])
        minor = int(split_cc[1])
        return major, minor

    raise RuntimeError("the compute capability string is unsupported format: " + cc)


def have_fp16(compute_capability):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_capability: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, minor = parse_cc(compute_capability)
    # fp 16 support in reference to:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions
    if major == 5 and minor == 3:
        return True
    # NOTE: exclude compute capability 6.1 device although it is actually available
    #       to compute fp16, because 6.1 only have low-rate fp16 performance.
    if major == 6 and minor != 1:
        return True
    if major == 7:
        return True

    return False

def have_int8(compute_capability):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_capability : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, minor = parse_cc(compute_capability)
    if major == 6 and minor == 1:
        return True

    return False

def have_tensorcore(compute_capability):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_capability : str
        compute capability of a GPU (e.g. "7.0")
    """
    major, _ = parse_cc(compute_capability)
    if major == 7:
        return True

    return False
