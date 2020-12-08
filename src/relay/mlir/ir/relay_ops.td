#ifndef RELAY_OPS
#define RELAY_OPS

include "mlir/IR/OpBase.td"

#ifndef SHAPE_INFERENCE_INTERFACE
include "mlir/Interfaces/SideEffectInterfaces.td"
#endif // SHAPE_INFERENCE_INTERFACE

//===----------------------------------------------------------------------===//
// Relay dialect definitions
//===----------------------------------------------------------------------===//

def Relay_Dialect : Dialect {
  let name = "mrelay";

  let description = [{
The Relay dialect.

This dialect maps to Relay operations.
  }];

  let cppNamespace = "mrelay";
}

//===----------------------------------------------------------------------===//
// Relay op definitions
//===----------------------------------------------------------------------===//

class Relay_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Relay_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// tvm.relay.op.add
//===----------------------------------------------------------------------===//

def Relay_AddOp: Relay_Op<"add", [NoSideEffect]> {
    let summary = "Addition with numpy-style broadcasting.";

    let description = [{
Parameters:
    lhs: The left hand side input data
    rhs: The right hand side input data

Returns:
    Result: The computed result
    }];

    let arguments = (ins
                AnyTensor:$lhs,
                AnyTensor:$rhs
    );

    let results = (outs
                AnyTensor:$res
    );

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// tvm.relay.op.reshape
//===----------------------------------------------------------------------===//

def Relay_ReshapeOp: Relay_Op<"Reshape", [NoSideEffect]> {
    let summary = "Reshape the input array.";

    let description = [{
To give user more convenience in without doing manual shape inference, some
dimensions of the shape can take special values from the set
{0, -1, -2, -3, -4}. The significance of each is explained below:

`0` copy this dimension from the input to the output shape.

```
data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)
```

`-1` infers the dimension of the output shape by using the remainder of the
input dimensions keeping the size of the new array same as that of the input
array. At most one dimension of shape can be -1.

```
data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)
```

`-2` copy all/remainder of the input dimensions to the output shape.

```
data.shape = (2,3,4), newshape = (-2,), result.shape = (2,3,4)
data.shape = (2,3,4), newshape = (2,-2), result.shape = (2,3,4)
data.shape = (2,3,4), newshape = (-2,1,1), result.shape = (2,3,4,1,1)
```

`-3` use the product of two consecutive dimensions of the input shape as the
output dimension.

```
data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)
data.shape = (2,3,4), newshape = (-3,-2), result.shape = (6,4)
```

`-4` split one dimension of the input into two dimensions passed subsequent to
-4 in shape (can contain -1).

```
data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape = (1,2,3,4)
data.shape = (2,3,4), newshape = (2,-4,-1,3,-2), result.shape = (2,1,3,4)
```

Parameters:
    data – The input data to the operator.
    newshape – The new shape. Should be compatible with the original shape.

Returns
    res – The reshaped result.
    }];

    let arguments = (ins
                AnyTensor:$data,
                AnyTensor:$newshape
    );

    let results = (outs
                AnyTensor:$res
    );

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// tvm.relay.op.strided_slice
//===----------------------------------------------------------------------===//

def Relay_StridedSliceOp : Relay_Op<"strided_slice", [NoSideEffect]> {
    let summary = "Strided slice of an array.";

    let description = [{
Parameters:
    data – The source array to be sliced.
    begin – The indices to begin with in the slicing.
    end  – Indices indicating end of the slice.
    strides – Specifies the stride values, it can be negative in that case, the
        input tensor will be reversed in that particular axis.

Returns
    res – The computed result.


Examples::

```
  x = [[  1.,   4.,   7.,  10.],
       [  2.,   5.,   8.,  11.],
       [  3.,   6.,   9.,  12.]]

  strided_slice(x, begin=[0, 1], end=[2, 4], stride=[1, 1]) = [[ 4.,  7.,  10.],
                                                               [ 5.,  8.,  11.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  strided_slice(x, begin=[0, 0], end=[2, 2]) = [[[ 1.,  2.],
                                                 [ 3.,  4.]],

                                                [[ 5.,  6.],
                                                 [ 7.,  8.]]]
```
    }];

    let arguments = (ins
                AnyTensor:$data,
                AnyTensor:$begin,
                AnyTensor:$end,
                AnyTensor:$strides
    );

    let results = (outs
                AnyTensor:$res
    );

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// tvm.relay.op.const
//===----------------------------------------------------------------------===//

def Relay_ConstOp : Relay_Op<"const", [NoSideEffect]> {
    let summary = "Create a constant value.";

    let description = [{
Parameters
	value – The constant value.
	dtype – The data type of the value.

Returns
    res – The computed result.
    }];

    let arguments = (ins
                ElementsAttr:$value
    );

    let results = (outs
                AnyTensor:$res
    );

    let builders = [
      OpBuilderDAG<(ins "Attribute":$value)>];

    let hasFolder = 1;
}

#endif
