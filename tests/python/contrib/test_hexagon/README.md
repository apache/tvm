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
  * Using `k` to denote channel-out and `c` to denote channel-in

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

The key changes in TIR versus the baseline are 

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