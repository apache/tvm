Making your hardware accelerator TVM-ready with UMA 
=============================================
This tutorial will give you step-by-step guidance how to use UMA to
make your hardware accelerator TVM-ready.
While there is no one-fits-all solution for this problem, UMA targets to provide a stable and Python-only
API to integrate a number of hardware accelerator classes into TVM.

In this tutorial you will get to know the UMA API in three use cases of increasing complexity.
We call the accelerators in these cases **Vanilla**, **Strawberry** and **Chocolate**. 

Prerequisites
---

```
git clone https://github.com/apache/tvm.git
pip install 
```

Vanilla
===
**Vanilla** is a simple accelerator consisting of a MAC array, that can ONLY process Conv2D layers.
All other layers are executed on a CPU, that also orchestrates **Vanilla**.

For this purpose **Vanilla** has a C interface `vanilla_conv2dnchw`, that accepts pointers to input data *if_map*,
*weights* and *result* data, as well as the parameters of `Conv2D`: `oc`, `iw`, `ih`, `ic`, `kh`, `kw`.
```c
int vanilla_conv2dnchw(float* ifmap, float*  weights, float*  result, int oc, int iw, int ih, int ic, int kh, int kw);
```

The script `uma_cli` creates you code skeletons with API-calls into the UMA-API for your accelerator.

```
cd tvm/python/tvm/relay/backend/contrib/uma
python uma_cli.py --add-accelerator vanilla_accelerator --template vanilla
```
The option `--template vanilla` adds all the additional files required for this tutorial.

```
$ ls tvm/python/tvm/relay/backend/contrib/uma/vanilla_accelerator

backend.py
codegen.py
passes.py
patterns.py
strategies.py
```

Step 1: Vanilla backend
---
This snippet is a full backed for **Vanilla**:
```python
class VanillaAccelerator(UMABackend):
    """UMA backend for VanillaAccelerator."""

    def __init__(self):
        super().__init__()

        self._register_pattern("conv2d", conv2d_pattern())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaAcceleratorConv2DPass())
        self._register_codegen(fmt="c", includes=gen_includes)

    @property
    def target_name(self):
        return "vanilla_accelerator"
```
It is found in `tvm/python/tvm/relay/backend/contrib/uma/vanilla_accelerator/backend.py`.

Step 2: Define offloaded patterns
---

To specify that `Conv2D` is offloaded to **Vanilla**, we describe it as Relay dataflow pattern in 
`patterns.py` 
 [[DFPattern]](https://tvm.apache.org/docs/reference/langref/relay_pattern.html) 
:
```python
def conv2d_pattern():
    pattern = is_op("nn.conv2d")(wildcard(), wildcard())
    pattern = pattern.has_attr({"strides": [1, 1]})
    return pattern
```

To map Conv2D operations from Tensorflow input files to **Vanilla**'s 
low level function call, we are using the TIR pass 
*VanillaAcceleratorConv2DPass* (that will be discussed later in this tutorial).

Step 3: Modify Codegen
---
```
self._register_codegen(fmt="c", includes=gen_includes)
```

We tell TVM to create C code using ``fmt="c"`` via 
`self._register_codegen`. Since we specified `Conv2D` layers to be called via our 
own implementation `vanilla_conv2dnchw(...)`, the TVM generated C code also require an
`#include` statement.

This is done by providing the include-string like this:
```python
# in vanilla_accelerator/backend.py
self._register_codegen(fmt="c", includes=gen_includes)

# in vanilla_accelerator/codegen.py
def gen_includes() -> str:
    return "#include \"conv2dnchw.cpp\""
```        


Step 4: Build the NN
---
Now we are going to generate C code for an MNIST-12 NN using.
For this, run `vanilla_accelerator/run.py`.
This creates the directory `build` that contains the generated data in the model library format (MLF).
```
$cd build/
$ ls -1
codegen
lib.tar
metadata.json
parameters
runtime
src
```
To evaluate the generated C code go to 
```
$ cd codegen/host/src/
$ ls -1
default_lib0.c
default_lib1.c
default_lib2.c
default_lib3.c
```

Run a 

More
---
Did this tutorial **not** fit to your accelerator? Please add your requirements to the UMA thread in
the TVM discuss forum: [Link](https://discuss.tvm.apache.org/t/rfc-uma-universal-modular-accelerator-interface/12039).
We are eager to extend this tutorial to provide guidance on making further classes of AI hardware
accelerators TVM-ready using the UMA interface.

References
---
[UMA-RFC] [UMA: Universal Modular Accelerator Interface](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0060_UMA_Unified_Modular_Accelerator_Interface.md), TVM RFC, June 2022.

[DFPattern] [Pattern Matching in Relay](https://tvm.apache.org/docs/reference/langref/relay_pattern.html) 
