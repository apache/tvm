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

Hexagon conv2d -> conv2d schedules

# Baseline conv2d -> conv2d

This is a baseline 1x1 conv2d -> 1x1 conv2d schedule for Hexagon.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_conv2d.py::TestConv2dConv2dPackedFilter::test_conv2d[1-64-128-0-1-1-128-1-1-128-1-1-float32-llvm]"

## Parameters

| Parameter                | Value |
| ------------------------ | ----- |
| Batch                    | 1     |
| Input Size               | 64x64 |
| Input Channel            | 128   |
| Conv2d #1 Pad            | 0     |
| Conv2d #1 Stride         | 1     |
| Conv2d #1 Kernel Size    | 1     |
| Conv2d #1 Output Channel | 128   |
| Conv2d #2 Stride         | 1     |
| Conv2d #2 Kernel Size    | 1     |
| Conv2d #2 Output Channel | 128   |
| k_split                  | 1     |
| h_split                  | 1     |

## Constants

| Constant           | Value |
| ------------------ | ----- |
| Conv2d #2 Pad      | 0     |
| Conv2d #1 Dilation | 1     |
| Conv2d #2 Dilation | 1     |

## Shapes and Layouts

| Tensor       | Type     | Layout      | Shape                  |
| ------------ | -------- | ----------- | ---------------------- |
| Input        | Logical  | NHWC        | [1, 64, 64, 128]       |
| Padded Input | Logical  | NHWC        | [1, 64, 64, 128]       |
| Packed Input | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] |
| Filter 1     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] |
| Filter 2     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] |
| Output       | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] |

Logically both filters are OIHW [128, 128, 1, 1] with the assumption that they are pre-packed into physical layouts for this test.

## Schedule

This is the conv2d compute schedule:

```
  for (ko.outer: int32, 0, 4) {
    for (ho.outer: int32, 0, 8) {
      // caches computed here
      for (wo.c: int32, 0, 8) {
        for (rc.outer_1: int32, 0, 4) {
          for (hi.c: int32, 0, 8) {
            for (wi.c: int32, 0, 8) {
              for (ki.c: int32, 0, 32) {
                for (rc.inner_1: int32, 0, 32) {
```

Note that conv2d #1 has an independent loop over channel out `ko` dimension.  This is because the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d.  Hence, we must compute over all output channels of conv2d #1 before we compute conv2d #2.

## Cache Usage

*Input Cache*

We compute over the WC8h8w32c portion of the input so we need 8 * 4 * 8 * 8 * 32 = 64k for the input cache.

```
  allocate(packed_input.global: Pointer(global float32), float32, [65536]), storage_scope = global;
```

*Filter Cache*

We compute over the IHW8i32o4i portion of each filter so we need 4 * 1 * 1 * 8 * 32 * 4 = 4k filter cache.

```
  allocate(packed_filter.global: Pointer(global float32), float32, [4096]), storage_scope = global;
```

Note that there is just one cache which is reused for conv2d / filter #1 and conv2d / filter #2.

*Output Cache*

The output cache is the same size as the input cache given the "square" nature of thie computation --- 1x1 kernels with same channel in / out.  There is a temporary allocation to store the results of conv2d #1:

```
  allocate(temp_output: Pointer(global float32), float32, [65536]), storage_scope = global;
```

Note that the input cache is reused to store the results of conv2d #2.

## Assumptions

* n/a

## To Do

* n/a

## Annotated TIR

```
primfn(placeholder_3: handle, placeholder_4: handle, placeholder_5: handle, output_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {output: Buffer(output_2: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []),             // nhw8h8w32c
             placeholder_2: Buffer(placeholder_6: Pointer(float32), float32, [4, 4, 1, 1, 8, 32, 4], []), // oihw8i32o4i
             placeholder_1: Buffer(placeholder_7: Pointer(float32), float32, [4, 4, 1, 1, 8, 32, 4], []), // oihw8i32o4i
             placeholder: Buffer(placeholder_8: Pointer(float32), float32, [1, 64, 64, 128], [])}         // nhwc
  buffer_map = {placeholder_3: placeholder, placeholder_4: placeholder_1, placeholder_5: placeholder_2, output_1: output} {
  allocate(packed_input.global: Pointer(global float32), float32, [65536]), storage_scope = global;
  allocate(temp_output: Pointer(global float32), float32, [65536]), storage_scope = global;
  allocate(packed_filter.global: Pointer(global float32), float32, [4096]), storage_scope = global;
  for (ko.outer: int32, 0, 4) {
    for (ho.outer: int32, 0, 8) {

      // input cache read
      for (wo: int32, 0, 8) {
        for (co: int32, 0, 4) {
          for (hi: int32, 0, 8) {
            for (wi: int32, 0, 8) {
              for (ci: int32, 0, 32) {
                packed_input.global[(((((wo*8192) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] = 
                  (float32*)placeholder_8[((((((ho.outer*65536) + (hi*8192)) + (wo*1024)) + (wi*128)) + (co*32)) + ci)]
              }
            }
          }
        }
      }

      // NOTE: compute over all output channels of conv2d #1 before computing conv2d #2
      for (ko.outer_1: int32, 0, 4) {

        // filter #1 cache read
        for (co: int32, 0, 4) {
          for (cio: int32, 0, 8) {
            for (ki: int32, 0, 32) {
              for (cii: int32, 0, 4) {
                packed_filter.global[((((co*1024) + (cio*128)) + (ki*4)) + cii)] = 
                  (float32*)placeholder_7[(((((ko.outer_1*4096) + (co*1024)) + (cio*128)) + (ki*4)) + cii)]
              }
            }
          }
        }

        // conv2d #1
        for (wo: int32, 0, 8) {

          // init temp output to zero
          for (hi.init: int32, 0, 8) {
            for (wi.init: int32, 0, 8) {
              for (ki.init: int32, 0, 32) {
                temp_output[(((((wo*8192) + (ko.outer_1*2048)) + (hi.init*256)) + (wi.init*32)) + ki.init)] = 0f32
              }
            }
          }

          // compute
          for (rc.outer: int32, 0, 4) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  for (rc.inner: int32, 0, 32) {
                    temp_output[(((((wo*8192) + (ko.outer_1*2048)) + (hi*256)) + (wi*32)) + ki)] = 
                    (
                      (float32*)temp_output[(((((wo*8192) + (ko.outer_1*2048)) + (hi*256)) + (wi*32)) + ki)] + 
                      (
                        (float32*)packed_input.global[(((((wo*8192) + (rc.outer*2048)) + (hi*256)) + (wi*32)) + rc.inner)] *
                        (float32*)packed_filter.global[((((rc.outer*1024) + (floordiv(rc.inner, 4)*128)) + (ki*4)) + floormod(rc.inner, 4))]
                      )
                    )
                  }
                }
              }
            }
          }
        }
      }

      // filter #2 cache read
      // NOTE: reusing same filter cache
      for (co: int32, 0, 4) {
        for (cio: int32, 0, 8) {
          for (ki: int32, 0, 32) {
            for (cii: int32, 0, 4) {
              packed_filter.global[((((co*1024) + (cio*128)) + (ki*4)) + cii)] = 
                (float32*)placeholder_6[(((((ko.outer*4096) + (co*1024)) + (cio*128)) + (ki*4)) + cii)]
            }
          }
        }
      }

      // conv2d #2
      for (wo.c: int32, 0, 8) {

        // init output cache to zero
        // NOTE: reusing the input cache as the output cache
        for (hi.c.init: int32, 0, 8) {
          for (wi.c.init: int32, 0, 8) {
            for (ki.c.init: int32, 0, 32) {
              packed_input.global[((((wo.c*2048) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
            }
          }
        }

        // compute
        for (rc.outer_1: int32, 0, 4) {
          for (hi.c: int32, 0, 8) {
            for (wi.c: int32, 0, 8) {
              for (ki.c: int32, 0, 32) {
                for (rc.inner_1: int32, 0, 32) {
                  packed_input.global[((((wo.c*2048) + (hi.c*256)) + (wi.c*32)) + ki.c)] = 
                  (
                    (float32*)packed_input.global[((((wo.c*2048) + (hi.c*256)) + (wi.c*32)) + ki.c)] + 
                    (
                      (float32*)temp_output[(((((wo.c*8192) + (rc.outer_1*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner_1)] *
                      (float32*)packed_filter.global[((((rc.outer_1*1024) + (floordiv(rc.inner_1, 4)*128)) + (ki.c*4)) + floormod(rc.inner_1, 4))]
                    )
                  )
                }
              }
            }
          }
        }
      }

      // write back output cache
      for (wo_1: int32, 0, 8) {
        for (hi_1: int32, 0, 8) {
          for (wi_1: int32, 0, 8) {
            for (ki_1: int32, 0, 32) {
              output_2[((((((ho.outer*65536) + (wo_1*8192)) + (ko.outer*2048)) + (hi_1*256)) + (wi_1*32)) + ki_1)] = 
                (float32*)packed_input.global[((((wo_1*2048) + (hi_1*256)) + (wi_1*32)) + ki_1)]
            }
          }
        }
      }
    }
  }
}
```

# Split on Channel Out and Height

Uses parameters `k_split` and `h_split` which creates a loop split on the outer channel out `ko` and height `ho` loops creating `outer` and `inner` loops for each split.  The cache reads and writes are computed at `ho.outer` which means that cache allocation grow in proportion to `k_split` and `h_split` factors.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_conv2d.py::TestConv2dConv2dPackedFilter::test_conv2d[1-64-128-0-1-1-128-1-1-128-2-2-float32-llvm]"

## Parameters

| Parameter                | Value |
| ------------------------ | ----- |
| Batch                    | 1     |
| Input Size               | 64x64 |
| Input Channel            | 128   |
| Conv2d #1 Pad            | 0     |
| Conv2d #1 Stride         | 1     |
| Conv2d #1 Kernel Size    | 1     |
| Conv2d #1 Output Channel | 128   |
| Conv2d #2 Stride         | 1     |
| Conv2d #2 Kernel Size    | 1     |
| Conv2d #2 Output Channel | 128   |
| k_split                  | 2 ^   |
| h_split                  | 2 ^   |

^ Changes from above

## Constants

| Constant           | Value |
| ------------------ | ----- |
| Conv2d #2 Pad      | 0     |
| Conv2d #1 Dilation | 1     |
| Conv2d #2 Dilation | 1     |

## Shapes and Layouts

| Tensor       | Type     | Layout      | Shape                  |
| ------------ | -------- | ----------- | ---------------------- |
| Input        | Logical  | NHWC        | [1, 64, 64, 128]       |
| Padded Input | Logical  | NHWC        | [1, 64, 64, 128]       |
| Packed Input | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] |
| Filter 1     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] |
| Filter 2     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] |
| Output       | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] |

Logically both filters are OIHW [128, 128, 1, 1] with the assumption that they are pre-packed into physical layouts for this test.

## Schedule

This is the conv2d compute schedule:

```
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {
      // caches computed here
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {
            for (rc.outer_1: int32, 0, 4) {
              for (hi.c: int32, 0, 8) {
                for (wi.c: int32, 0, 8) {
                  for (ki.c: int32, 0, 32) {
                    for (rc.inner_1: int32, 0, 32) {
```

The major change here versus above is the presence of `inner` loops for both channel out `ko` and height `ho` dimensions created from the `k_split` and `h_split` schedule parameters respectively:


```
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
```

The major effect of this change is increased cache usage given that caches are computed at the `ho.outer` level of the loop schedule.  This is documented in the next section.

(Same as above) Note that conv2d #1 has an independent loop over channel out `ko` dimension.  This is because the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d.  Hence, we must compute over all output channels of conv2d #1 before we compute conv2d #2.


## Cache Usage

*Input Cache*

The input cache grows by a factor of `h_split = 2` compared with above:

```
  allocate(packed_input.global: Pointer(global float32), float32, [131072]), storage_scope = global;
```

*Filter Cache*

The filter cache grows by a factor of `k_split = 2` compared with above:

```
  allocate(packed_filter.global: Pointer(global float32), float32, [8192]), storage_scope = global;
```

*Output Cache*

The output cache is the same size as the input cache given the "square" nature of thie computation --- 1x1 kernels with same channel in / out.  There is a temporary allocation to store the results of conv2d #1:

```
  allocate(temp_output: Pointer(global float32), float32, [131072]), storage_scope = global;
```

(Same as above) Note that the input cache is reused to store the results of conv2d #2.

## Assumptions

* n/a

## To Do

* n/a

## Annotated TIR

```
primfn(placeholder_3: handle, placeholder_4: handle, placeholder_5: handle, output_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {output: Buffer(output_2: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []),             // nhw8h8w32c
             placeholder_2: Buffer(placeholder_6: Pointer(float32), float32, [4, 4, 1, 1, 8, 32, 4], []), // oihw8i32o4i
             placeholder_1: Buffer(placeholder_7: Pointer(float32), float32, [4, 4, 1, 1, 8, 32, 4], []), // oihw8i32o4i
             placeholder: Buffer(placeholder_8: Pointer(float32), float32, [1, 64, 64, 128], [])}         // nhwc
  buffer_map = {placeholder_3: placeholder, placeholder_4: placeholder_1, placeholder_5: placeholder_2, output_1: output} {
  allocate(packed_input.global: Pointer(global float32), float32, [131072]), storage_scope = global;
  allocate(temp_output: Pointer(global float32), float32, [131072]), storage_scope = global;
  allocate(packed_filter.global: Pointer(global float32), float32, [8192]), storage_scope = global;
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {

      // input cache read
      for (ho.inner: int32, 0, 2) {
        for (wo: int32, 0, 8) {
          for (co: int32, 0, 4) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ci: int32, 0, 32) {
                  packed_input.global[((((((ho.inner*65536) + (wo*8192)) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] = 
                    (float32*)placeholder_8[(((((((ho.outer*131072) + (ho.inner*65536)) + (hi*8192)) + (wo*1024)) + (wi*128)) + (co*32)) + ci)]
                }
              }
            }
          }
        }
      }

      // NOTE: compute over all output channels of conv2d #1 before computing conv2d #2
      for (ko.outer_1: int32, 0, 2) {
        for (ko.inner: int32, 0, 2) {
          // filter #1 cache read
          for (co: int32, 0, 4) {
            for (cio: int32, 0, 8) {
              for (ki: int32, 0, 32) {
                for (cii: int32, 0, 4) {
                  packed_filter.global[(((((ko.inner*4096) + (co*1024)) + (cio*128)) + (ki*4)) + cii)] = 
                    (float32*)placeholder_7[((((((ko.outer_1*8192) + (ko.inner*4096)) + (co*1024)) + (cio*128)) + (ki*4)) + cii)]
                }
              }
            }
          }
        }

        // conv2d #1
        for (ko.inner: int32, 0, 2) {
          for (ho.inner: int32, 0, 2) {
            for (wo: int32, 0, 8) {

              // init temp output to zero
              for (hi.init: int32, 0, 8) {
                for (wi.init: int32, 0, 8) {
                  for (ki.init: int32, 0, 32) {
                    temp_output[(((((((ho.inner*65536) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi.init*256)) + (wi.init*32)) + ki.init)] = 0f32
                  }
                }
              }

              // compute
              for (rc.outer: int32, 0, 4) {
                for (hi: int32, 0, 8) {
                  for (wi: int32, 0, 8) {
                    for (ki: int32, 0, 32) {
                      for (rc.inner: int32, 0, 32) {
                        temp_output[(((((((ho.inner*65536) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] = 
                        (
                          (float32*)temp_output[(((((((ho.inner*65536) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] + 
                          (
                            (float32*)packed_input.global[((((((ho.inner*65536) + (wo*8192)) + (rc.outer*2048)) + (hi*256)) + (wi*32)) + rc.inner)] *
                            (float32*)packed_filter.global[(((((ko.inner*4096) + (rc.outer*1024)) + (floordiv(rc.inner, 4)*128)) + (ki*4)) + floormod(rc.inner, 4))]
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

      // filter #2 cache read
      // NOTE: reusing same filter cache
      for (ko.inner: int32, 0, 2) {
        for (co: int32, 0, 4) {
          for (cio: int32, 0, 8) {
            for (ki: int32, 0, 32) {
              for (cii: int32, 0, 4) {
                packed_filter.global[(((((ko.inner*4096) + (co*1024)) + (cio*128)) + (ki*4)) + cii)] = 
                  (float32*)placeholder_6[((((((ko.outer*8192) + (ko.inner*4096)) + (co*1024)) + (cio*128)) + (ki*4)) + cii)]
              }
            }
          }
        }
      }

      // conv2d #2
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {

            // init output cache to zero
            // NOTE: reusing the input cache as the output cache
            for (hi.c.init: int32, 0, 8) {
              for (wi.c.init: int32, 0, 8) {
                for (ki.c.init: int32, 0, 32) {
                  packed_input.global[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c.init*256)) + (wi.c.init*32)) + ki.c.init)] = 0f32
                }
              }
            }

            // compute
            for (rc.outer_1: int32, 0, 4) {
              for (hi.c: int32, 0, 8) {
                for (wi.c: int32, 0, 8) {
                  for (ki.c: int32, 0, 32) {
                    for (rc.inner_1: int32, 0, 32) {
                      packed_input.global[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] = 
                      (
                        (float32*)packed_input.global[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] + 
                        (
                          (float32*)temp_output[((((((ho.c.inner*65536) + (wo.c*8192)) + (rc.outer_1*2048)) + (hi.c*256)) + (wi.c*32)) + rc.inner_1)] *
                          (float32*)packed_filter.global[(((((ko.c.inner*4096) + (rc.outer_1*1024)) + (floordiv(rc.inner_1, 4)*128)) + (ki.c*4)) + floormod(rc.inner_1, 4))]
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

      // write back output cache
      for (ko.inner_1: int32, 0, 2) {
        for (ho.inner_1: int32, 0, 2) {
          for (wo_1: int32, 0, 8) {
            for (hi_1: int32, 0, 8) {
              for (wi_1: int32, 0, 8) {
                for (ki_1: int32, 0, 32) {
                  output_2[((((((((ho.outer*131072) + (ho.inner_1*65536)) + (wo_1*8192)) + (ko.outer*4096)) + (ko.inner_1*2048)) + (hi_1*256)) + (wi_1*32)) + ki_1)] = 
                    (float32*)packed_input.global[((((((ho.inner_1*32768) + (wo_1*4096)) + (ko.inner_1*2048)) + (hi_1*256)) + (wi_1*32)) + ki_1)]
                }
              }
            }
          }
        }
      }
    }
  }
}
```