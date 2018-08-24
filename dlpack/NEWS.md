DLPack Change Log
=================

This file records the changes in DLPack in reverse chronological order.

## v0.2
- New device types
  - kDLMetal for Apple Metal device
  - kDLVPI for verilog simulator memory
  - kDLROCM for AMD GPUs
- Add prefix DL to all enum constant values
  - This requires dependent frameworks to upgrade their reference to these constant
  - The ABI is compatible, as it is only change of constant name.
- Add DLManagedTensor structure for borrowing tensors

## v0.1
- Finalize DLTensor structure
