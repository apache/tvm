<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Hexagon conv2d schedules

# Baseline conv2d

This is a baseline 1x1 conv2d schedule for Hexagon.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[1-64-64-0-1-1-128-1-1-float32-llvm]"

## Parameters

| Parameter | Value |
| --------- | ----- |
| Batch     | 1     |
| Spatial   | 64x64 |
| Input Ch  | 64    |
| Padding   | 0     |
| Stride    | 1     |
| Filter    | 1x1   |
| Output Ch | 128   |

## Assumptions

* n/a

## To Do

* n/a

## Annotated TIR

```
primfn(input_handle: handle, filter_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []), // NHWC8h8w32c
             filter_buffer: Buffer(filter_pointer: Pointer(float32), float32, [4, 2, 1, 1, 8, 32, 4], []), // OIHW8i32o4i
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])} // NHWC (pending RFC)
  buffer_map = {input_handle: input_buffer, filter_handle: filter_buffer, output_handle: output_buffer} {
  allocate(input.cache: Pointer(global float32), float32, [32768]), storage_scope = global;
  allocate(filter.cache: Pointer(global float32), float32, [2048]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [16384]), storage_scope = global;

  for (ko.outer: int32, 0, 4) {
    for (ho.outer: int32, 0, 8) {

      // input cache read
      // NHWC -> NHWC8h8w32c (pending RFC)
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

      // filter cache read
      for (co: int32, 0, 2) {
        for (ci8: int32, 0, 8) {
          for (ki: int32, 0, 32) {
            for (ci4: int32, 0, 4) {
              filter.cache[((((co*1024) + (ci8*128)) + (ki*4)) + ci4)] =
                (float32*)filter_pointer[(((((ko.outer*2048) + (co*1024)) + (ci8*128)) + (ki*4)) + ci4)]
            }
          }
        }
      }

      // compute
      for (wo.c: int32, 0, 8) {

        // init output cache
        for (hi.c.init: int32, 0, 8) {
          for (wi.c.init: int32, 0, 8) {
            for (ki.c.init: int32, 0, 32) {
              output.cache[((((wo.c*2048) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
            }
          }
        }

        // convolution
        for (rc.outer: int32, 0, 2) {
          for (hi.c: int32, 0, 8) {
            for (wi.c: int32, 0, 8) {
              for (ki.c: int32, 0, 32) {
                for (rc.inner: int32, 0, 32) {
                  output.cache[((((wo.c*2048) + (hi.c*256)) + (wi.c*32)) + ki.c)] =
                  (
                    (float32*)output.cache[((((wo.c*2048) + (hi.c*256)) + (wi.c*32)) + ki.c)] +
                    (
                      (float32*)input.cache[(((((wo.c*4096) + (rc.outer*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner)] *
                      (float32*)filter.cache[((((rc.outer*1024) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                    )
                  )
                }
              }
            }
          }
        }
      } // end wo.c

      // cache write
      for (wo: int32, 0, 8) {
        for (hi: int32, 0, 8) {
          for (wi: int32, 0, 8) {
            for (ki: int32, 0, 32) {
              output_pointer[((((((ho.outer*65536) + (wo*8192)) + (ko.outer*2048)) + (hi*256)) + (wi*32)) + ki)] =
                (float32*)output.cache[((((wo*2048) + (hi*256)) + (wi*32)) + ki)]
            }
          }
        }
      }
    } // end ho.outer
  } // end ko.outer
}
```

# Split on Channel Out and Height - "Full Output Slice"

Adds new parameters `k_split` and `h_split` which creates a loop split on the outer channel out `ko` and height `ho` loops creating `outer` and `inner` loops for each split.  The cache reads and writes are computed at `ho.outer` which means that cache allocation grow in proportion to `k_split` and `h_split` factors.

The key changes in TIR versus the above are...

1) Increased cache allocations:

```
  // input cache grows by factor of h_split = 2
  allocate(input.cache: Pointer(global float32), float32, [65536]), storage_scope = global;

  // filter cache grows by factor of k_split = 2
  allocate(filter.cache: Pointer(global float32), float32, [4096]), storage_scope = global;

  // output cache grows by factor of h_split * k_split = 4
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
```

2) Outer loop splits using k_split and h_split factors

```
  // ko.outer = outer loop split on ko using k_split factor
  for (ko.outer: int32, 0, 2) {
    // ho.outer = outer loop split on ho using h_split factor
    for (ho.outer: int32, 0, 4) {
```

3) Inner loop splits in both cache read / write and compute schedules.  This is taken from the compute schedule e.g.
```
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
```

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[1-64-64-0-1-1-128-2-2-float32-llvm]"

## Parameters

| Parameter | Value |
| --------- | ----- |
| Batch     | 1     |
| Spatial   | 64x64 |
| Input Ch  | 64    |
| Padding   | 0     |
| Stride    | 1     |
| Filter    | 1x1   |
| Output Ch | 128   |
| k_split   | 2     |
| h_split   | 2     |

## Assumptions

* n/a

## To Do

* n/a

## Annotated TIR

```
primfn(input_handle: handle, filter_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []), // NHWC8h8w32c
             filter_buffer: Buffer(filter_pointer: Pointer(float32), float32, [4, 2, 1, 1, 8, 32, 4], []), // OIHW8i32o4i
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])} // NHWC (pending RFC)
  buffer_map = {input_handle: input_buffer, filter_handle: filter_buffer, output_handle: output_buffer} {

  // input cache grows by factor of h_split = 2
  allocate(input.cache: Pointer(global float32), float32, [65536]), storage_scope = global;

  // filter cache grows by factor of k_split = 2
  allocate(filter.cache: Pointer(global float32), float32, [4096]), storage_scope = global;

  // output cache grows by factor of h_split * k_split = 4
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;

  // ko.outer = outer loop split on ko using k_split factor
  for (ko.outer: int32, 0, 2) {
    // ho.outer = outer loop split on ho using h_split factor
    for (ho.outer: int32, 0, 4) {

      // input cache read
      // NHWC -> NHWC8h8w32c (pending RFC)
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
      } // end ho.inner

      // filter cache read
      for (ko.inner: int32, 0, 2) {
        for (co: int32, 0, 2) {
          for (ci8: int32, 0, 8) {
            for (ki: int32, 0, 32) {
              for (ci4: int32, 0, 4) {
                filter.cache[(((((ko.inner*2048) + (co*1024)) + (ci8*128)) + (ki*4)) + ci4)] =
                  (float32*)filter_pointer[((((((ko.outer*4096) + (ko.inner*2048)) + (co*1024)) + (ci8*128)) + (ki*4)) + ci4)]
              }
            }
          }
        }
      } // end ko.inner

      // compute
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {

            // init output cache
            for (hi.c.init: int32, 0, 8) {
              for (wi.c.init: int32, 0, 8) {
                for (ki.c.init: int32, 0, 32) {
                  output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
                }
              }
            }

            // convolution
            for (rc.outer: int32, 0, 2) {
              for (hi.c: int32, 0, 8) {
                for (wi.c: int32, 0, 8) {
                  for (ki.c: int32, 0, 32) {
                    for (rc.inner: int32, 0, 32) {
                      output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] =
                      (
                        (float32*)output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] +
                        (
                          (float32*)input.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (rc.outer*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner)] *
                          (float32*)filter.cache[(((((ko.c.inner*2048) + (rc.outer*1024)) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                        )
                      )
                    }
                  }
                }
              }
            }
          } // end wo.c
        } // end ho.c.inner
      } // end ko.c.inner

      // cache write
      for (ko.inner: int32, 0, 2) {
        for (ho.inner: int32, 0, 2) {
          for (wo: int32, 0, 8) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  output_pointer[((((((((ho.outer*131072) + (ho.inner*65536)) + (wo*8192)) + (ko.outer*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] =
                    (float32*)output.cache[((((((ho.inner*32768) + (wo*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)]
                }
              }
            }
          }
        } // end ho.inner
      } // end ko.inner
    } // end ho.outer
  } // end ko.outer
}
```

# 3x3 conv2d (no padding)

Change from a 1x1 filter to a 3x3 filter.  The implication of this change is that `h_split + 1` rather than just `h_split` "full width" slices of the input are required to compute the output.  This is due to the fact that the 3x3 filter will "fall off the bottom" of the input and thus the vertically adjacent "full width" slice must be prefetched into the input cache.

The key changes in TIR versus the above are...

1) Increased input cache size to hold the vertically adjacent slice

```
  // input cache grows to hold vertically adjacent slice
  allocate(input.cache: Pointer(global float32), float32, [98304]), storage_scope = global;
```

2) Loop over `ho.inner` upper bound increased from `h_split` = 2 to `h_split + 1` = 3

```
  for (ho.outer: int32, 0, 4) {
    for (ho.inner: int32, 0, 3) {
      if (((ho.outer*2) + ho.inner) < 8) {
```

The `if` statement above indicates NOT to prefetch the vertically adjacent slice at the "bottom" of the input since it does not exist.


3) Increased filter cache size to hold 3x3 filter

```
  // filter cache grows to hold larger 3x3 filter
  allocate(filter.cache: Pointer(global float32), float32, [36864]), storage_scope = global;
```

4) Loops over `rh` and `rw` the kernel spatial dimensions:
```
          for (rh: int32, 0, 3) {
            for (rw: int32, 0, 3) {
```

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_blocked.py::TestConv2dPackedFilter::test_conv2d[1-64-64-0-1-3-128-2-2-float32-llvm]"

## Parameters

| Parameter | Value |
| --------- | ----- |
| Batch     | 1     |
| Spatial   | 64x64 |
| Input Ch  | 64    |
| Padding   | 0     |
| Stride    | 1     |
| Filter    | 1x1   |
| Output Ch | 128   |
| k_split   | 2     |
| h_split   | 2     |

## Assumptions

* n/a

## To Do

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
primfn(input_handle: handle, filter_handle: handle, output_handle: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "default_function", "tir.noalias": True, "target": meta[Target][0]}
  buffers = {output_buffer: Buffer(output_pointer: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []), // NHWC8h8w32c
             filter_buffer: Buffer(filter_pointer: Pointer(float32), float32, [4, 2, 3, 3, 8, 32, 4], []), // OIHW8i32o4i
             input_buffer: Buffer(input_pointer: Pointer(float32), float32, [1, 64, 64, 64], [])} // NHWC (pending RFC)
  buffer_map = {input_handle: input_buffer, filter_handle: filter_buffer, output_handle: output_buffer} {
  // input cache grows to hold vertically adjacent slice
  allocate(input.cache: Pointer(global float32), float32, [98304]), storage_scope = global;
  // filter cache grows to hold larger 3x3 filter
  allocate(filter.cache: Pointer(global float32), float32, [36864]), storage_scope = global;
  allocate(output.cache: Pointer(global float32), float32, [65536]), storage_scope = global;
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {
      // input cache read
      // NHWC -> NHWC8h8w32c (pending RFC)
      for (ho.inner: int32, 0, 3) {
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
      // filter cache read
      for (ko.inner: int32, 0, 2) {
        for (co: int32, 0, 2) {
          for (rh: int32, 0, 3) {
            for (rw: int32, 0, 3) {
              for (ci8: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  for (ci4: int32, 0, 4) {
                    filter.cache[(((((((ko.inner*18432) + (co*9216)) + (rh*3072)) + (rw*1024)) + (ci8*128)) + (ki*4)) + ci4)] =
                      (float32*)filter_pointer[((((((((ko.outer*36864) + (ko.inner*18432)) + (co*9216)) + (rh*3072)) + (rw*1024)) + (ci8*128)) + (ki*4)) + ci4)]
                  }
                }
              }
            } // end rw
          } // end rh
        }
      }
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {
            for (hi.c.init: int32, 0, 8) {
              for (wi.c.init: int32, 0, 8) {
                for (ki.c.init: int32, 0, 32) {
                  output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
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
                          output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] =
                          (
                            (float32*)output.cache[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] +
                            (
                              (float32*)input.cache[((((((((floordiv((hi.c + rh), 8)*32768) + (ho.c.inner*32768)) + (floordiv((wi.c + rw), 8)*4096)) + (wo.c*4096)) + (rc.outer*2048)) + (floormod((hi.c + rh), 8)*256)) + (floormod((wi.c + rw), 8)*32)) + rc.inner)] *
                              (float32*)filter.cache[(((((((ko.c.inner*18432) + (rc.outer*9216)) + (rh*3072)) + (rw*1024)) + (floordiv(rc.inner, 4)*128)) + (ki.c*4)) + floormod(rc.inner, 4))]
                            )
                          )
                        }
                      }
                    } // end rw
                  } // end rh
                }
              }
            }
          } // end wo.c
        } // end ho.c.inner
      } // end ko.c.inner
      for (ko.inner: int32, 0, 2) {
        for (ho.inner: int32, 0, 2) {
          for (wo: int32, 0, 8) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  output_pointer[((((((((ho.outer*131072) + (ho.inner*65536)) + (wo*8192)) + (ko.outer*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] =
                    (float32*)output.cache[((((((ho.inner*32768) + (wo*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)]
                }
              }
            }
          }
        } // end ho.inner
      } // end ko.inner
    } // end ho.outer
  } // end ko.outer
}```
