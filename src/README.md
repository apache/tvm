# Code Organization

Header files in include are public APIs that share across modules.
There can be internal header files within each module that sit in src.

## Modules
- common: Internal common utilities.
- api: API function registration
- lang: The definition of DSL related data structure
- arithmetic: Arithmetic expression and set simplification
- op: The detail implementations about each operation(compute, scan, placeholder)
- schedule: The operations on the schedule graph before converting to IR.
- pass: The optimization pass on the IR structure
- codegen: The code generator.
- runtime: Minimum runtime related codes
- contrib: Contrib extension libraries
