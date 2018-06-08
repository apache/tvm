# Hybrid Frontend Developer Guide

This hybrid frontend is aimed at:
1. Building IR in a more intuitive way
2. Writing preliminary versions of some idioms that yet have not been supported by

## Features

### Software emulation

This feature supports both software emulation and compilation of the code.

To define a function, you need to use `tvm.hybrid.script` decorator to indicate this is a hybrid function:
````Python
@tvm.hybrid.script
def outer_product(a, b, c):
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            c[i, j] = a[i] * b[j]
a = numpy.random.rand(100)
b = numpy.random.rand(99)
c = numpy.zeros((100, 99))
outer_product(a, b)
````
This decorator will help you to import [key words](#keywords) required spontaneously when software emulation.
Every element in the argument list is either a python variable or `numpy` tensor.

### Backend Compilation

The current parse interface looks like:
````Python
a = tvm.placeholder((100, ), name='a')
b = tvm.placeholder((99, ), name='b')
c = tvm.placeholder((100, 99), name='c')
tvm.hybrid.parse(outer_product, [a, b, c]) # return an ir root of this function
````
**TODO**: If we pass these tvm tensors to this function, it returns a op node:
````Python
a = tvm.placeholder((100, ), name='a')
b = tvm.placeholder((99, ), name='b')
c = tvm.placeholder((100, 99), name='c')
op = outer_product(a, b, c) # return the corresponding op node
````
#### Scheduling

**Under construction, not truly supported yet.**

Follow up the example above, you can use some tvm like interfaces to manipulate the structure of IR:
````Python
sch = tvm.create_schedule(op)
jo, ji = sch.split(j, 4)
sch.vectorize(ji)
````
`split`, `reorder`, and loop_annotation will be supported!

### Attributes
So far, ONLY tensors' `shape` attribute is supported!

### Loops

In HalideIR, loops have in total 4 types: `serail`, `unrolled`, `parallel`, and `vectorized`.

Here we use `range`, `serial`, `unroll`, `parallel`, and `vectorize`, these **5** keywords to annotate the types of for loops.

**NOTE**: In HalideIR those are enums, they are in passive form. Here we use active form to annotate loops, because they are ready to run.

**NOTE**: Unlike what that is in HalideIR, in `loop_type(a, b)`, `a` is the starting point and `b` is the trip count of iterations. Here `loop_type(a, b)` indicates `[a, b)`.

### Variables

Because there is no variables in `HalideIR`, all the mutatable variables will be lowered to an array with size 1.
It takes the first store of a variable as its declaration.
**NOTE**: Unlike conventional Python, the declared array can only be used in the scope level it is declared.
````Python
for i in range(5):
    sum = 0
    for j in range(5):
    	sum += a[i, j] #do something with sum
    b[i] = sum #you can still use sum in this level
#you can NEVER use some here, even though it is allowed in conventional Python
a[0] = sum
````
### Conditional Statement and Expression

````Python
if condition:
    # do something
a = b if condition else c
````
However, NO `True` and `False` keyword supported yet.

### Math intrinsics
So far, these math intrinsics, `log`, `exp`, `sigmoid`, `tanh`, `power`, and `popcount`, are supported. No import is required, just use it!
### Array allocation
**TODO**: Use a function call `allocation(shape, type, share/local)` to declare an array buffer. The basic usage is roughly the same as variables
### Thread bind
You can also do loop-thread bind by writing code like this:
````Python
for tx in bind("threadIdx.x", 100):
    a[tx] = b[tx]
````
## Appendix

### <a name="keywords"> Keywords </a>
- Statement keywords: `for`, `in`, `if`, `else`
- For keywords: `serial`, `range`, `unroll`, `parallel`, `vectorize`, `bind`
- Math keywords: `log`, `exp`, `sigmoid`, `tanh`, `power`, `popcount`
