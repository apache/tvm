# TOPI: TVM Operator Inventory

TOPI is the operator collection library for TVM intended at sharing the effort of crafting
and optimizing tvm generated kernels. The goal:

- Provide sugars for operator declaration
- Give common primitives for fused op creation.
- Provide commonly used schedules under each architectures

## Organization
- [include](include) C++ library, header only
- [python](python) python library
- [recipe](recipe) Recipe collections containing useful operator examples.

## Guidelines
- Use numpy-style naming convention for known ops
- Seperate operator declaration from schedule when possible.
  - This can be inconvenient but enables more general scheduling across ops.
  - We can always recover the tensors from its outputs by traversing the tree.
- Deliberately assert the requirements
  - Some kernels have requirements on shape and data layout, assert them
- Data layout aware, if not specified in argument or in function, assume NCHW by default.


## Testcase
- Add testcases to testout the schedule and dataflow in the TOPI workflow
- Only do correctness testing without attaching compiler flags and only run it once.

## Performance Tuning Workflow
Since TVM is work in progress, some optimization might not be perfect.
One quick way I find useful is to do codegen plus manual modification.
The workflow is:

- Generate the GPU kernels, write them into a file, say ```perf/matexp_generated.cu```
- Copy the generated file into another one, say ```perf/matexp_manual.cu```,
  do modifications according to your intuition.
- Set use_manual flag in the script to continue the codegen workflow as normal, but piggy back the manual written code instead.
- Observe the performance difference.
- If the performance improves, mark the manual code and think of optimization pass
  to generate the desired target code.
