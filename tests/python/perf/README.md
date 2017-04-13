# Perf Examples for TVM
This folder contains perf examples of tvm under various settings.

## GPU Perf Workflow
Since TVM is work in progress, some optimization might not be perfect.
One quick way I find useful is to do codegen plus manual modification.
The workflow is:

- Generate the GPU kernels, write them into a file, say ```cuda/matexp_generated.cu```
- Copy the generated file into another one, say ```cuda/matexp_manual.cu```,
  do modifications according to your intuition.
- Set use_manual flag in the script to continue the codegen workflow as normal, but piggy back the manual written code instead.
- Observe the performance difference.
- If the performance improves, mark the manual code and think of optimization pass
  to generate the desired target code.
