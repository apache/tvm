VTA Configuration
=================

The VTA stack incorporates both a hardware accelerator stack and
a TVM based software stack.
VTA incorporates flexibility out of the box: by modifying the
``vta/config/vta_config.json`` high-level configuration file,
the user can change the shape of the tensor intrinsic,
clock frequency, pipelining, data type width, and on-chip buffer sizes.

Parameters Overview
-------------------

We explain the parameters listed in the ``vta_config.json`` file in the table
below.

+-----------------------+------------+--------------------------------------------------------+
| Attribute             | Format     | Description                                            |
+=======================+============+========================================================+
| ``TARGET``            | String     | The TVM device target.                                 |
+-----------------------+------------+--------------------------------------------------------+
| ``HW_TARGET``         | Int        | FPGA frequency in MHz.                                 |
+-----------------------+------------+--------------------------------------------------------+
| ``HW_CLK_TARGET``     | Int        | FPGA clock period in ns target for HLS tool.           |
+-----------------------+------------+--------------------------------------------------------+
| ``HW_VER``            | String     | VTA hardware version number.                           |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_INP_WIDTH``     | Int (log2) | Input data type signed integer width.                  |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_WGT_WIDTH``     | Int (log2) | Weight data type signed integer width.                 |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_ACC_WIDTH``     | Int (log2) | Accumulator data type signed integer width.            |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_OUT_WIDTH``     | Int (log2) | Output data type signed integer width.                 |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_BATCH``         | Int (log2) | VTA matrix multiply intrinsic output dimension 0.      |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_BLOCK_IN``      | Int (log2) | VTA matrix multiply reduction dimension.               |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_BLOCK_OUT``     | Int (log2) | VTA matrix multiply intrinsic output dimension 1.      |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_UOP_BUFF_SIZE`` | Int (log2) | Micro-op on-chip buffer in Bytes.                      |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_INP_BUFF_SIZE`` | Int (log2) | Input on-chip buffer in Bytes.                         |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_WGT_BUFF_SIZE`` | Int (log2) | Weight on-chip buffer in Bytes.                        |
+-----------------------+------------+--------------------------------------------------------+
| ``LOG_ACC_BUFF_SIZE`` | Int (log2) | Accumulator on-chip buffer in Bytes.                   |
+-----------------------+------------+--------------------------------------------------------+


 .. note::

    When a parameter name is preceded with ``LOG``, it means that it describes a value that can only be expressed a power of two.
    For that reason we describe these parameters by their log2 value.
    For instance, to describe an integer width of 8-bits for the input data types, we set the ``LOG_INP_WIDTH`` to be 3, which is the log2 of 8.
    Similarly, to descibe a 64kB micro-op buffer, we would set ``LOG_UOP_BUFF_SIZE`` to be 16.

We provide additional detail below regarding each parameter:

 - ``TARGET``: Can be set to ``"pynq"`` or ``"sim"``.
 - ``HW_TARGET``: In pynq mode, can be set to ``100``, ``142``, ``167``, or ``200`` MHz.
 - ``HW_CLK_TARGET``: The lower the target, the more pipeline stages HLS will insert to achieve timing closure during place and route (this can also slightly decrease performance).
 - ``HW_VER``: Hardware version which increments everytime the VTA hardware design changes. This parameter is used to uniquely idenfity hardware bitstreams.
 - ``LOG_OUT_WIDTH``: We recommend matching ``LOG_OUT_WIDTH`` to ``LOG_INP_WIDTH``.
 - ``LOG_BATCH``: Equivalent to A in multiplication of shape (A, B) x (B, C), or typically, the batch dimension.
 - ``LOG_BATCH``: Equivalent to A in multiplication of shape (A, B) x (B, C), or typically, the batch dimension.
 - ``LOG_BLOCK_IN``: Equivalent to B in multiplication of shape (A, B) x (B, C), or typically, the input channel dimension.
 - ``LOG_BLOCK_OUT``: Equivalent to C in multiplication of shape (A, B) x (B, C), or typically, the output channel dimension.

