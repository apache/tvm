""" The definiton of SearchTask """
import os
import numpy as np

from tvm.runtime import ndarray
from tvm.runtime._ffi_node_api import LoadJSON, SaveJSON


# The map stores special registered buffer for measurement.
# This can be used for sparse workloads when we cannot use random tensors for measurment.
# {
#     "workload_key_0": {
#         "task_input_0": Tensor(...),
#         "task_input_1": Tensor(...)
#     },
#     "workload_key_1": {
#         "task_input_2": Tensor(...),
#         "task_input_3": Tensor(...)
#     },
#     ...
# }
TASK_INPUT_BUFFER_TABLE = {}


def _save_buffer_to_file(buffer_name, buffer_data):
    """Save the current Tensor buffer to a numpy file.

    File name will be: {buffer_name}.{buffer_shape}_{buffer_data_type}.npy
    """
    np_data = buffer_data.numpy()

    buffer_name += "."
    for i in np_data.shape:
        buffer_name += "%d_" % (i)
    buffer_name += "%s" % (np_data.dtype)
    buffer_name += ".npy"

    np_data.tofile(buffer_name, " ")


def _try_load_buffer_from_file(buffer_name):
    """Try to load buffer from a numpy file, if not found, return None.

    File name has a same format as `_save_buffer_to_file`.
    """
    filelist = os.listdir()

    for file in filelist:
        if file.startswith(buffer_name + "."):
            meta_info = file.split(".")[-2].split("_")
            shape = [int(i) for i in meta_info[:-1]]
            dtype = meta_info[-1]
            buffer_data = np.fromfile(file, dtype=dtype, sep=" ")
            buffer_data = buffer_data.reshape(shape)
            return ndarray.array(buffer_data)

    return None


def register_task_input_buffer(
    workload_key,
    input_name,
    input_data,
    overwrite=False,
    save_to_file=False,
):
    """Register special buffer for measurement.

    Parameters
    ----------
    workload_key : str
        The workload key of the SearchTask.

    input_name : str
        The name of input buffer.

    input_data : tvm.nd.NDArray
        The input Tensor data.

    overwrite : bool = False
        Whether to overwrite the data if a name has already registered.

    save_to_file : bool = False
        Whether to save the data to a local file as well. This can be reused to resume the last
        tuning process.

    Returns
    -------
    tvm.nd.NDArray
        The actual registered Tensor data of this input_name. With `overwrite` set to False, will
        return the original one if the name has already registered before.
    """
    global TASK_INPUT_BUFFER_TABLE

    if workload_key not in TASK_INPUT_BUFFER_TABLE:
        TASK_INPUT_BUFFER_TABLE[workload_key] = {}
    input_table = TASK_INPUT_BUFFER_TABLE[workload_key]

    if not overwrite:
        if input_name not in input_table.keys():
            # Try to load buffer data from local file
            tensor_from_file = _try_load_buffer_from_file(input_name)
            if tensor_from_file:
                input_table[input_name] = tensor_from_file
        elif input_name in input_table.keys():
            raise RuntimeError(
                "Tensor %s exists in TASK_INPUT_BUFFER_TABLE, %s"
                % (input_name, "set overwrite to True or this Tensor will not be registered")
            )

    input_table[input_name] = input_data
    if save_to_file:
        _save_buffer_to_file(input_name, input_data)
    return input_data


def get_task_input_buffer(workload_key, input_name):
    """Get special buffer for measurement.

    The buffers are registered by `register_task_input_buffer`.

    Parameters
    ----------
    workload_key : str
        The workload key of the SearchTask.

    input_name : str
        The name of input buffer.

    Returns
    -------
    tvm.nd.NDArray
        The registered input buffer.
    """
    global TASK_INPUT_BUFFER_TABLE

    if workload_key not in TASK_INPUT_BUFFER_TABLE:
        TASK_INPUT_BUFFER_TABLE[workload_key] = {}
    input_table = TASK_INPUT_BUFFER_TABLE[workload_key]

    if input_name not in input_table:
        # Try to load buffer data from local file
        tensor_from_file = _try_load_buffer_from_file(input_name)
        if tensor_from_file:
            input_table[input_name] = tensor_from_file

    # Then check for the default table, the input names extracted from a relay model will be
    # stored here for we're not able to get the workload_key at that time
    if input_name not in input_table:
        input_table = TASK_INPUT_BUFFER_TABLE["default"]

    if input_name in input_table:
        return input_table[input_name]

    raise ValueError(
        "%s not found in TASK_INPUT_BUFFER_TABLE, " % (input_name)
        + "should provide with `SearchTask(..., task_inputs={...})`"
    )


def serialize_task_input_buffer(workload_key):
    """
    Serialize a task input buffer entry.

    This is used when the start method of multiprocessing is spawn.
    We need to serialize the task input buffer table it in the new processes.

    Parameters
    ----------
    workload_key : str
        The workload key

    Returns
    -------
    data: Tuple
        The serialized pickable data
    """
    sname = workload_key

    # the return value of get_task_input_buffer is tvm.ndarray
    # convert it to np.array to make it picklable,
    global TASK_INPUT_BUFFER_TABLE

    if workload_key not in TASK_INPUT_BUFFER_TABLE:
        TASK_INPUT_BUFFER_TABLE[workload_key] = {}
    svalue = TASK_INPUT_BUFFER_TABLE[workload_key]
    if not callable(svalue):
        # pylint: disable=assignment-from-no-return
        svalue = SaveJSON(svalue)

    return sname, svalue


def deserialize_task_input_buffer(data):
    """
    Deserialize a task input buffer entry.
    This should be used along with :code:`serialize_task_input_buffer_table`

    Parameters
    ----------
    data: Tuple
        The return value of :code:`serialize_task_input_buffer_table`
    """
    global TASK_INPUT_BUFFER_TABLE

    name, value = data
    # pylint: disable=assignment-from-no-return
    if not callable(value):
        value = LoadJSON(value)
        TASK_INPUT_BUFFER_TABLE[name] = value
