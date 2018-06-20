"""Classes and functions to handle data-dump of TVM Debugger."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

METADATA_FILE_PREFIX = "_tvmdbg_"
DEVICE_TAG = "device_"
HASH_TAG = "hash"

class InconvertibleTensorProto(object):
    """Represents a TensorProto that cannot be converted to np.ndarray."""

    def __init__(self, tensor_proto, initialized=True):
        """Constructor.

        Args:
          tensor_proto: the `TensorProto` object that cannot be represented as a
            `np.ndarray` object.
          initialized: (`bool`) whether the Tensor is initialized.
        """
        self._tensor_proto = tensor_proto
        self._initialized = initialized

    def __str__(self):
        output = "" if self._initialized else "Uninitialized tensor:\n"
        output += str(self._tensor_proto)
        return output

def load_tensor_from_event_file(event_file_path):
    """Load a tensor from an event file.

    Assumes that the event file contains a `Event` protobuf and the `Event`
    protobuf contains a `Tensor` value.

    Args:
      event_file_path: (`str`) path to the event file.

    Returns:
      The tensor value loaded from the event file, as a `numpy.ndarray`. For
      uninitialized Tensors, returns `None`. For Tensors of data types that
      cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return
      `None`.
    """
    content = np.load(event_file_path)
    return content

def _get_tensor_name(node_name, output_slot):
    """Get tensor name given node name and output slot index.

    Args:
      node_name: Name of the node that outputs the tensor, as a string.
      output_slot: Output slot index of the tensor, as an integer.

    Returns:
      Name of the tensor, as a string.
    """
    return "%s:%d" % (node_name, output_slot)


def _get_tensor_watch_key(node_name, output_slot, debug_op):
    """Get the string representation of a debug watch on a tensor.

    Args:
      node_name: Name of the node by which the watched tensor is produced, as a
          string.
      output_slot: Output slot index of the tensor, as an integer.
      debug_op: Name of the debug op that is used to watch the tensor, as a
          string.

    Returns:
      A string representing the debug watch on the tensor (i.e., the "watch
          key").
    """
    return "%s:%s" % (_get_tensor_name(node_name, output_slot), debug_op)

def device_path_to_device_name(device_dir):
    """Parse device name from device path.

    Args:
      device_dir: (str) a directory name for the device.

    Returns:
      (str) parsed device name.
    """
    path_items = os.path.basename(device_dir)[
        len(METADATA_FILE_PREFIX) + len(DEVICE_TAG):].split(",")
    return "/".join([
        path_item.replace("device_", "device:").replace("_", ":", 1)
        for path_item in path_items])

class DebugTensorDatum(object):
    """A single tensor dumped by TVM Debugger (tvmdbg).

    Contains metadata about the dumped tensor, including `timestamp`,
    `node_name`, `output_slot`, `debug_op`, and path to the dump file
    (`file_path`).

    This type does not hold the generally space-expensive tensor value (numpy
    array). Instead, it points to the file from which the tensor value can be
    loaded (with the `get_tensor` method) if needed.
    """

    def __init__(self, dump_root, debug_dump_rel_path):
        """`DebugTensorDatum` constructor.

        Args:
          dump_root: (`str`) Debug dump root directory. This path should not include
            the path component that represents the device name (see also below).
          debug_dump_rel_path: (`str`) Path to a debug dump file, relative to the
            `dump_root`. The first item of this relative path is assumed to be
            a path representing the name of the device that the Tensor belongs to.
            See `device_path_to_device_name` for more details on the device path.
            For example, suppose the debug dump root
            directory is `/tmp/tvmdbg_1` and the dump file is at
            `/tmp/tvmdbg_1/<device_path>/>ns_1/node_a_0_DebugIdentity_123456789`,
            then the value of the debug_dump_rel_path should be
            `<device_path>/ns_1/node_a_0_DebugIdenity_1234456789`.

        Raises:
          ValueError: If the base file name of the dump file does not conform to
            the dump file naming pattern:
            `node_name`_`output_slot`_`debug_op`_`timestamp`
        """
        path_components = os.path.normpath(debug_dump_rel_path).split(os.sep)
        self._device_name = device_path_to_device_name(path_components[0])
        base = path_components[-1]
        if base.count("_") < 3:
            raise ValueError(
                "Dump file path does not conform to the naming pattern: %s" % base)

        self._extended_timestamp = base.split("_")[-1]
        # It may include an index suffix at the end if file path collision happened
        # due to identical timestamps.
        if "-" in self._extended_timestamp:
            self._timestamp = int(
                self._extended_timestamp[:self._extended_timestamp.find("-")])
        else:
            self._timestamp = int(self._extended_timestamp)

        self._debug_op = base.split("_")[-2]
        self._output_slot = int(base.split("_")[-3])

        node_base_name = "_".join(base.split("_")[:-3])
        self._node_name = "/".join(path_components[1:-1] + [node_base_name])

        self._file_path = os.path.join(dump_root, debug_dump_rel_path)
        self._dump_size_bytes = (os.stat(self._file_path).st_size if
                                 os.path.exists(self._file_path) else None)

    def __str__(self):
        return "{DebugTensorDatum (%s) %s:%d @ %s @ %d}" % (self.device_name,
                                                            self.node_name,
                                                            self.output_slot,
                                                            self.debug_op,
                                                            self.timestamp)

    def __repr__(self):
        return self.__str__()

    def get_tensor(self):
        """Get tensor from the dump (`Event`) file.

        Returns:
          The tensor loaded from the dump (`Event`) file.
        """
        return load_tensor_from_event_file(self.file_path)

    @property
    def timestamp(self):
        """Timestamp of when this tensor value was dumped.

        Returns:
          (`int`) The timestamp in microseconds.
        """
        return self._timestamp

    @property
    def extended_timestamp(self):
        """Extended timestamp, possibly with an index suffix.

        The index suffix, e.g., "-1", is for disambiguating multiple dumps of the
        same tensor with the same timestamp, which can occur if the dumping events
        are spaced by shorter than the temporal resolution of the timestamps.

        Returns:
          (`str`) The extended timestamp.
        """
        return self._extended_timestamp

    @property
    def debug_op(self):
        """Name of the debug op.

        Returns:
          (`str`) debug op name (e.g., `DebugIdentity`).
        """
        return self._debug_op

    @property
    def device_name(self):
        """Name of the device that the tensor belongs to.

        Returns:
          (`str`) device name.
        """
        return self._device_name

    @property
    def node_name(self):
        """Name of the node from which the tensor value was dumped.

        Returns:
          (`str`) name of the node watched by the debug op.
        """
        return self._node_name

    @property
    def output_slot(self):
        """Output slot index from which the tensor value was dumped.

        Returns:
          (`int`) output slot index watched by the debug op.
        """
        return self._output_slot

    @property
    def tensor_name(self):
        """Name of the tensor watched by the debug op.

        Returns:
          (`str`) `Tensor` name, in the form of `node_name`:`output_slot`
        """
        return _get_tensor_name(self.node_name, self.output_slot)

    @property
    def watch_key(self):
        """Watch key identities a debug watch on a tensor.

        Returns:
          (`str`) A watch key, in the form of `tensor_name`:`debug_op`.
        """
        return _get_tensor_watch_key(self.node_name, self.output_slot,
                                     self.debug_op)

    @property
    def file_path(self):
        """Path to the file which stores the value of the dumped tensor."""
        return self._file_path

    @property
    def dump_size_bytes(self):
        """Size of the dump file.

        Unit: byte.

        Returns:
          If the dump file exists, size of the dump file, in bytes.
          If the dump file does not exist, None.
        """
        return self._dump_size_bytes
