# TVM Tutorial

## TVM Inference End2End
    - Torch/Tflite model execution 
    - relay.testing models
    - Cross-Compilation / RPC

## TVM Passes  / Pass Infrastructure

    - Relay / TE / TIR
      - PrintIR Instrumentation Example
    - Passes / Pass Debugging

## Scheduling Language

    - Manually Schedule Convolution for CPU
      - TE Schedule
      - TIR Schedule
    
    - Extending TVM to use new Schedules

## Autoscheduling
    - AutoTVM
      - Autotuning Network 
      - Extending manual schedule with knobs
    - AutoScheduler
      - AutoScheduler back to back example
      - Auto-Generated Tuning spaces visualization? 
    - MetaScheduler? (Optional)


## Quantization (Christoph)
    - Quantization in TVM 
    - Framework Pre-Quantized Models

    - Inference Options:
      - Dense
      - Bit-Serial


## MicroTVM (Christoph)
    - Micro-TVM Inference Example
      - Host-Driven
      - AOT-Compiler

    - Internals:
      - Model Library Format
      - Project API-Server

    - Memory Planning?

## Backend / BYOC
  - External C-Code generation example
  - Graph Pattern Matching
  - 
  - BYOC & MicroTVM
