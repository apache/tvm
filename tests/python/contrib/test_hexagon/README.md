Documents manual TE schedule to illustrate Hexagon operator slicing.

# High Level Notes

* Using float32 (for now) so that tests will pass on CPU
* Using global storage scope (for now) which means "cache" reads and writes from global, to global
* TIR is pending changes from the work-in-progress layout RFC
* TIR has been hand-edited for context and clarity
  * Added C-style comments
  * Changed variable names
  * Added spacing and line breaks
* Naming conventions
  * Using input (instead of activation)
  * Using kernel (instead of weight, filter)
  * Using `k` to denote channel-out and `c` or `rc` (reduction channel) to denote channel-in
  * Using `rh` and `rw` to denote kernel height and width

# Calling Convention

TODO: Map this packed string to parameters
conv2d_packed_filter-1-1-0-float32-1-1-64-64-64-llvm

# Baseline conv2d

This is a baseline 1x1 conv2d schedule for Hexagon.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[conv2d_packed_filter-1-1-0-float32-1-1-64-64-64-llvm]"

## Parameters

| Parameter | Value       |
| --------- | ----------- |
| Batch     | 1           |
| Kernel    | 1x1         |
| Spatial   | 64x64       |
| Input Ch  | 64          |
| Output Ch | 64          |
| Stride    | 1           |
| Padding   | 0           |
| Layout    | NHWC8h8w32c |

## Assumptions

* Microkernels will compute "full depth" in channel-out (k) dimension.  
  * The compute schedule (see TIR below) 
    * Places the outer channel-out loop over `ko` inside the outer width loop over `wo` 
    * Encodes the assumption that Hexagon microkernels will compute "full depth" in the channel-out (k) dimension

## To Do

* Adjust compute schedule and add kernel cache read once Hexagon microkernel semantics are understood

## Annotated TIR

```
primfn(input_handle: handle, kernel_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 2, 8, 8, 32], []), // NHWC8h8w32c
             kernel_buffer: Buffer(kernel_pointer: Pointer(float32), float32, [2, 2, 1, 1, 8, 32, 4], []), // OIHW8i32o4i
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])} // NHWC (pending layout RFC)
  buffer_map = {input_handle: input_buffer, kernel_handle: kernel_buffer, output_handle: output_buffer} {

  allocate(input.cache: Pointer(global float32), float32, [32768]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [32768]), storage_scope = global;

  for (ho.outer: int32, 0, 8) {
    // cache read
    // NHWC -> NHWC8h8w32c (pending layout RFC)
    for (wo: int32, 0, 8) {
      for (co: int32, 0, 2) {
        for (hi: int32, 0, 8) {
          for (wi: int32, 0, 8) {
            for (ci: int32, 0, 32) {
              input.cache[(((((wo*4096) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] = 
                (float32*)input_pointer[((((((ho.outer*32768) + (hi*4096)) + (wo*512)) + (wi*64)) + (co*32)) + ci)]
            }
          }
        }
      }
    }

    // compute
    for (wo.c: int32, 0, 8) {
      for (ko.c: int32, 0, 2) {
        
        // init output cache
        for (hi.c.init: int32, 0, 8) {
          for (wi.c.init: int32, 0, 8) {
            for (ki.c.init: int32, 0, 32) {
              output.cache[(((((wo.c*4096) + (ko.c*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
            }
          }
        }

        // convolution
        for (rc.outer: int32, 0, 2) {
          for (hi.c: int32, 0, 8) {
            for (wi.c: int32, 0, 8) {
              for (ki.c: int32, 0, 32) {
                for (rc.inner: int32, 0, 32) {
                  output.cache[(((((wo.c*4096) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] = 
                  (
                    (float32*)output.cache[(((((wo.c*4096) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] + 
                    (
                      (float32*)input.cache[(((((wo.c*4096) + (rc.outer*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner)] *
                      (float32*)kernel_pointer[(((((ko.c*2048) + (rc.outer*1024)) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                    )
                  )
                }
              }
            }
          }
        } // end rc.outer
      } // end ko.c
    } // end wo.c

    // cache write
    for (wo: int32, 0, 8) {
      for (ko: int32, 0, 2) {
        for (hi: int32, 0, 8) {
          for (wi: int32, 0, 8) {
            for (ki: int32, 0, 32) {
              output_pointer[((((((ho.outer*32768) + (wo*4096)) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)] = 
                (float32*)output.cache[(((((wo*4096) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)]
            }
          }
        }
      }
    }
  }
}
```

# Split on Height - "Full Output Slice"

Adds a new parameter `h_split` which creates a loop split on the height `h` dimension.  The cache reads and writes are moved to the outer of the two loops created by that split - the loop over `ho.outer`.  This increases cache usage by a factor equivalent to `h_split`.  The compute is still "full width" and "full depth" in the channel-out dimension and now over multiple slices in the height `h` dimension.  

The key changes in TIR versus the baseline are ...

1) Increased cache allocations:

```
  allocate(input.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
```

2) The loop split on the `h` dimension:

```
  for (ho.outer: int32, 0, 4) {
    for (ho.inner: int32, 0, 2) {
```

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[conv2d_packed_filter-1-1-0-float32-2-1-64-64-64-llvm]"

## Parameters

| Parameter | Value       |
| --------- | ----------- |
| Batch     | 1           |
| Kernel    | 1x1         |
| Spatial   | 64x64       |
| Input Ch  | 64          |
| Output Ch | 64          |
| Stride    | 1           |
| Padding   | 0           |
| Layout    | NHWC8h8w32c |
| h_split   | 2           |

## Assumptions

Same as baseline

## To Do

Same as baseline

## Annotated TIR

```
primfn(input_handle: handle, kernel_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 2, 8, 8, 32], []),
             kernel_buffer: Buffer(kernel_pointer: Pointer(float32), float32, [2, 2, 1, 1, 8, 32, 4], []),
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])}
  buffer_map = {input_handle: input_buffer, kernel_handle: kernel_buffer, output_handle: output_buffer} {
  
  // increased cache usage due to h_split parameter
  allocate(input.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;

  // loop split ho.outer vs. ho.inner based on h_split parameter
  for (ho.outer: int32, 0, 4) {
    for (ho.inner: int32, 0, 2) {
      for (wo: int32, 0, 8) {
        for (co: int32, 0, 2) {
          for (hi: int32, 0, 8) {
            for (wi: int32, 0, 8) {
              for (ci: int32, 0, 32) {
                input.cache[((((((ho.inner*32768) + (wo*4096)) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] = 
                  (float32*)input_pointer[(((((((ho.outer*65536) + (ho.inner*32768)) + (hi*4096)) + (wo*512)) + (wi*64)) + (co*32)) + ci)]
              }
            }
          }
        }
      }
    }
    for (ho.c.inner: int32, 0, 2) {
      for (wo.c: int32, 0, 8) {
        for (ko.c: int32, 0, 2) {
          for (hi.c.init: int32, 0, 8) {
            for (wi.c.init: int32, 0, 8) {
              for (ki.c.init: int32, 0, 32) {
                output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
              }
            }
          }
          for (rc.outer: int32, 0, 2) {
            for (hi.c: int32, 0, 8) {
              for (wi.c: int32, 0, 8) {
                for (ki.c: int32, 0, 32) {
                  for (rc.inner: int32, 0, 32) {
                    output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] = 
                    (
                      (float32*)output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] + 
                      (
                        (float32*)input.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (rc.outer*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner)] *
                        (float32*)kernel_pointer[(((((ko.c*2048) + (rc.outer*1024)) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                      )
                    )
                  }
                }
              }
            }
          }
        }
      }
    }
    for (ho.inner: int32, 0, 2) {
      for (wo: int32, 0, 8) {
        for (ko: int32, 0, 2) {
          for (hi: int32, 0, 8) {
            for (wi: int32, 0, 8) {
              for (ki: int32, 0, 32) {
                output_pointer[(((((((ho.outer*65536) + (ho.inner*32768)) + (wo*4096)) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)] = 
                  (float32*)output.cache[((((((ho.inner*32768) + (wo*4096)) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)]
              }
            }
          }
        }
      }
    }
  }
}
```

# 3x3 conv2d (no padding)

Change from a 1x1 kernel to a 3x3 kernel.  The implication of this change is that `h_split + 1` rather than just `h_split` "full width" slices of the input are required to compute the output.  This is due to the fact that the 3x3 kernel will "fall off the bottom" of the input and thus the vertically adjacent "full width" slice must be prefetched into the input cache.

The key changes in TIR versus the above are...

1) Increased input cache size to hold the vertically adjacent slice

```
  allocate(input.cache: Pointer(global float32), float32, [98304]), storage_scope = global;
```

2) Loop over `ho.inner` upper bound increased from `h_split` = 2 to `h_split + 1` = 3

```
  for (ho.outer: int32, 0, 4) {
    for (ho.inner: int32, 0, 3) {
      if (((ho.outer*2) + ho.inner) < 8) {
```

The `if` statement above indicates NOT to prefetch the vertically adjacent slice at the "bottom" of the input since it does not exist.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[conv2d_packed_filter-3-1-0-float32-2-1-64-64-64-llvm]"

## Parameters

| Parameter | Value       |
| --------- | ----------- |
| Batch     | 1           |
| Kernel    | 3x3         |
| Spatial   | 64x64       |
| Input Ch  | 64          |
| Output Ch | 64          |
| Stride    | 1           |
| Padding   | 0           |
| Layout    | NHWC8h8w32c |
| h_split   | 2           |

## Assumptions

Same as above

## To Do

Same as above, and ...

There may be some opportunity to optimize cache reuse in this case.  Consider the loops over `ho.outer` and `ho.inner` and the index calculation `ho.outer * 64k + ho.inner * 32k` into the input pointer:

| ho.outer | ho.inner | ho.outer * 64k + ho.inner * 32k       |
| -------- | -------- | ------------------------------------- |
| 0        | 0        | 0                                     |
| 0        | 1        | 32k                                   |
| 0        | 2        | 64k (vertical adjacent slice loop 0)  |
| 1        | 0        | 64k                                   |
| 1        | 1        | 96k                                   |
| 1        | 2        | 128k (vertical adjacent slice loop 1) |
| 2        | 0        | 128k                                  |
| 2        | 1        | 160k                                  |
| 2        | 2        | 192k (vertical adjacent slice loop 2) |
| 3        | 0        | 192k                                  |
| 3        | 1        | 224k                                  |
| 3        | 2        | (No vertical adjacent slice loop 3)   |

Noe that the vertically adjacent slice in loop N (i.e. the loop where `ho.outer` = N) is reused in loop N + 1.

## Annotated TIR

```
primfn(input_handle: handle, kernel_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 2, 8, 8, 32], []),
             kernel_buffer: Buffer(kernel_pointer: Pointer(float32), float32, [2, 2, 3, 3, 8, 32, 4], []),
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])}
  buffer_map = {input_handle: input_buffer, kernel_handle: kernel_buffer, output_handle: output_buffer} {

  // increased input cache size to hold vertically adjacent slice
  allocate(input.cache: Pointer(global float32), float32, [98304]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
  for (ho.outer: int32, 0, 4) {

    // iterate over h_split + 1 = 3 input slices
    for (ho.inner: int32, 0, 3) {

      // don't prefetch the vertically adjacent slice at the "bottom" of the input
      if (((ho.outer*2) + ho.inner) < 8) {
        for (wo: int32, 0, 8) {
          for (co: int32, 0, 2) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ci: int32, 0, 32) {
                  input.cache[((((((ho.inner*32768) + (wo*4096)) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] = 
                    (float32*)input_pointer[(((((((ho.outer*65536) + (ho.inner*32768)) + (hi*4096)) + (wo*512)) + (wi*64)) + (co*32)) + ci)]
                }
              }
            }
          }
        }
      }
    }
    for (ho.c.inner: int32, 0, 2) {
      for (wo.c: int32, 0, 8) {
        for (ko.c: int32, 0, 2) {
          for (hi.c.init: int32, 0, 8) {
            for (wi.c.init: int32, 0, 8) {
              for (ki.c.init: int32, 0, 32) {
                output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
              }
            }
          }
          for (rc.outer: int32, 0, 2) {
            for (hi.c: int32, 0, 8) {
              for (wi.c: int32, 0, 8) {
                for (rh: int32, 0, 3) {
                  for (rw: int32, 0, 3) {
                    for (ki.c: int32, 0, 32) {
                      for (rc.inner: int32, 0, 32) {
                        output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] = 
                        (
                          (float32*)output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] + 
                          (
                            (float32*)input.cache[((((((((floordiv((hi.c + rh), 8)*32768) + (ho.c.inner*32768)) + (floordiv((wi.c + rw), 8)*4096)) + (wo.c*4096)) + (rc.outer*2048)) + (floormod((hi.c + rh), 8)*256)) + (floormod((wi.c + rw), 8)*32)) + rc.inner)] *
                            (float32*)kernel_pointer[(((((((ko.c*18432) + (rc.outer*9216)) + (rh*3072)) + (rw*1024)) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                          )
                        )
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (ho.inner: int32, 0, 2) {
      for (wo: int32, 0, 8) {
        for (ko: int32, 0, 2) {
          for (hi: int32, 0, 8) {
            for (wi: int32, 0, 8) {
              for (ki: int32, 0, 32) {
                output_pointer[(((((((ho.outer*65536) + (ho.inner*32768)) + (wo*4096)) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)] = 
                  (float32*)output.cache[((((((ho.inner*32768) + (wo*4096)) + (ko*2048)) + (hi*256)) + (wi*32)) + ki)]
              }
            }
          }
        }
      }
    }
  }
}
```