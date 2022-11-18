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

The input is provided and padded in logical layout and then packed into its physical layout prior to compute.  Logical layout / shape information is provided as a reference for physical tensors.

| Tensor       | Type     | Layout      | Shape                  | Logical Layout | Logical Shape    |
| ------------ | -------- | ----------- | ---------------------- | -------------- | ---------------- |
| Input        | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Padded Input | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Packed Input | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |
| Filter 1     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] | OIHW           | [128, 128, 1, 1] |
| Temp Output  | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |
| Filter 2     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] | OIHW           | [128, 128, 1, 1] |
| Output       | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |

## Schedule

This is the conv2d compute schedule:

```
  for (ko.outer: int32, 0, 4) {
    for (ho.outer: int32, 0, 8) {

      // input cache read

      for (ko.outer_1: int32, 0, 4) {

        // filter #1 cache read

        // conv2d #1
        for (wo: int32, 0, 8) {
          for (rc.outer: int32, 0, 4) {
            for (hi: int32, 0, 8) {
              for (wi: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  for (rc.inner: int32, 0, 32) {
      } // end ko.outer_1

      // filter #2 cache read

      // conv2d #2
      for (wo.c: int32, 0, 8) {
        for (rc.outer_1: int32, 0, 4) {
          for (hi.c: int32, 0, 8) {
            for (wi.c: int32, 0, 8) {
              for (ki.c: int32, 0, 32) {
                for (rc.inner_1: int32, 0, 32) {

      // write back output cache

    } // end ho.outer
  } // end ko.outer
```

Note that conv2d #1 has an independent loop over the channel out `ko.outer_1` dimension.  This is because the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d so we must compute over all output channels of conv2d #1 before we compute conv2d #2.

```
      for (ko.outer_1: int32, 0, 2) {
```

## Cache Usage

*Input Cache*

We compute over the WC8h8w32c portion of the input so we need 8 * 4 * 8 * 8 * 32 = 64kb for the input cache.

```
  allocate(packed_input.global: Pointer(global float32), float32, [65536]), storage_scope = global;
```

*Filter Cache*

We compute over the IHW8i32o4i portion of each filter so we need 4 * 1 * 1 * 8 * 32 * 4 = 4kb filter cache.

```
  allocate(packed_filter.global: Pointer(global float32), float32, [4096]), storage_scope = global;
```

Note that there is just one cache which is reused for conv2d / filter #1 and conv2d / filter #2.

*Output Cache*

We compute over the WK8h832k portion of the output where `k` denotes the output channel.  The output cache is computed for each `ko.outer` which means it should be W * 8h * 8w * 32k = 8 * 8 * 8 * 32 = 16kb.  And, in fact, this is the case for a single conv2d case.   But, as already noted, for this conv2d -> conv2d case "the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d so we must compute over all output channels of conv2d #1 before we compute conv2d #2".  This means that the output cache must grow accordingly to K * W * 8h * 8w * 32k = 4 * 8 * 8 * 8 * 32 = 64kb.  There is a temporary allocation to store the results of conv2d #1:

```
  allocate(temp_output: Pointer(global float32), float32, [65536]), storage_scope = global;
```

Note that the input cache is reused to store the results of conv2d #2.

## Assumptions

* n/a

## To Do

* Reuse of the input cache to store the results of conv2d #2 could be problematic for async copy. e.g.

```
slice 0: global -> load -> cache0 -> conv2d_0 -> cache1 -> conv2d_1 -> cache0 -> store -> global
slice 1: global -> load -> cache0 -> conv2d_0 -> cache1 -> conv2d_1 -> cache0 -> store -> global
```

In this case the store from slice 0: cache0 -> store -> global
can potentially block the load in slice 1: global -> load -> cache0

StorageRewrite is responsible for planning these caches, we'll need to understand how to avoid this for the async case.

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

The input is provided and padded in logical layout and then packed into its physical layout prior to compute.  Logical layout / shape information is provided as a reference for physical tensors.

| Tensor       | Type     | Layout      | Shape                  | Logical Layout | Logical Shape    |
| ------------ | -------- | ----------- | ---------------------- | -------------- | ---------------- |
| Input        | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Padded Input | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Packed Input | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |
| Filter 1     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] | OIHW           | [128, 128, 1, 1] |
| Temp Output  | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |
| Filter 2     | Physical | OIHW8i32o4i | [4, 4, 1, 1, 8, 32, 4] | OIHW           | [128, 128, 1, 1] |
| Output       | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |

## Schedule

This is the conv2d compute schedule:

```
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {

      // input cache read
      for (ho.inner: int32, 0, 2) {
        ...
      }

      for (ko.outer_1: int32, 0, 2) {

        // filter #1 cache read
        for (ko.inner: int32, 0, 2) {
          ...
        }

        // conv2d #1
        for (ko.inner: int32, 0, 2) {
          for (ho.inner: int32, 0, 2) {
            for (wo: int32, 0, 8) {
              for (rc.outer: int32, 0, 4) {
                for (hi: int32, 0, 8) {
                  for (wi: int32, 0, 8) {
                    for (ki: int32, 0, 32) {
                      for (rc.inner: int32, 0, 32) {
      } // end ko.outer_1

      // filter #2 cache read
      for (ko.inner: int32, 0, 2) {
        ...
      }

      // conv2d #2
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {
            for (rc.outer_1: int32, 0, 4) {
              for (hi.c: int32, 0, 8) {
                for (wi.c: int32, 0, 8) {
                  for (ki.c: int32, 0, 32) {
                    for (rc.inner_1: int32, 0, 32) {

      // write back output cache

    } // end ho.outer
  } // end ko.outer
```

The major change here versus above is the presence of `inner` loops for both channel out `ko` and height `ho` dimensions created from the `k_split` and `h_split` schedule parameters respectively, for example:


```
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
```

The effect of this change is increased cache usage given where the caches are computed in the schedule.  Specifically, the input cache is now computed over `ho.inner` and the filter caches are computed over `ko.inner` which will grow the size of the cache.  Details below.

(Same as above) Note that conv2d #1 has an independent loop over the channel out `ko.outer_1` dimension.  This is because the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d so we must compute over all output channels of conv2d #1 before we compute conv2d #2.

```
      for (ko.outer_1: int32, 0, 2) {
```

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

(Same as above) Note that there is just one cache which is reused for conv2d / filter #1 and conv2d / filter #2.

*Output Cache*

The output cache grows by a factor of `k_split = 2` compared with above:

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

        // filter #1 cache read
        for (ko.inner: int32, 0, 2) {
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

# 3x3 conv2d -> conv2d (no padding)

Change from a 1x1 filter to a 3x3 filter.

## Command

pytest -sv "tests/python/contrib/test_hexagon/test_conv2d_conv2d.py::TestConv2dConv2dPackedFilter::test_conv2d[1-64-128-0-1-3-128-1-3-128-2-2-float32-llvm]"

## Parameters

| Parameter                | Value |
| ------------------------ | ----- |
| Batch                    | 1     |
| Input Size               | 64x64 |
| Input Channel            | 128   |
| Conv2d #1 Pad            | 0     |
| Conv2d #1 Stride         | 1     |
| Conv2d #1 Kernel Size    | 3 ^   |
| Conv2d #1 Output Channel | 128   |
| Conv2d #2 Stride         | 1     |
| Conv2d #2 Kernel Size    | 3 ^   |
| Conv2d #2 Output Channel | 128   |
| k_split                  | 2     |
| h_split                  | 2     |

^ Changes from above

## Constants

| Constant           | Value |
| ------------------ | ----- |
| Conv2d #2 Pad      | 0     |
| Conv2d #1 Dilation | 1     |
| Conv2d #2 Dilation | 1     |

## Shapes and Layouts

The input is provided and padded in logical layout and then packed into its physical layout prior to compute.  Logical layout / shape information is provided as a reference for physical tensors.

| Tensor       | Type     | Layout      | Shape                  | Logical Layout | Logical Shape    |
| ------------ | -------- | ----------- | ---------------------- | -------------- | ---------------- |
| Input        | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Padded Input | Logical  | NHWC        | [1, 64, 64, 128]       |                |                  |
| Packed Input | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 64, 64, 128] |
| Filter 1     | Physical | OIHW8i32o4i | [4, 4, 3, 3, 8, 32, 4] | OIHW           | [128, 128, 3, 3] |
| Temp Output  | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 62, 62, 128] |
| Filter 2     | Physical | OIHW8i32o4i | [4, 4, 3, 3, 8, 32, 4] | OIHW           | [128, 128, 3, 3] |
| Output       | Physical | NHWC8h8w32c | [1, 8, 8, 4, 8, 8, 32] | NHWC           | [1, 60, 60, 128] |

## Schedule

This is the conv2d compute schedule:

```
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {

      for (ko.outer_1: int32, 0, 2) {
        for (ho.outer_1: int32, 0, 2) {

          // input cache read
          for (ho.inner: int32, 0, 3) {
            if ((((ho.outer_1*2) + (ho.outer*2)) + ho.inner) < 8) {
              ...
            }
          }

          // filter #1 cache read
          for (ko.inner: int32, 0, 2) {
            ...
          }

          // conv2d #1
          for (ko.inner: int32, 0, 2) {
            for (ho.inner: int32, 0, 2) {
              for (wo: int32, 0, 8) {
                if (((ho.outer_1*2) + ho.inner) < 3) {
                  if ((((ho.outer_1*2) + (ho.outer*2)) + ho.inner) < 8) {
                    for (rc.outer: int32, 0, 4) {
                      for (hi: int32, 0, 8) {
                        for (wi: int32, 0, 8) {
                          for (rh: int32, 0, 3) {
                            for (rw: int32, 0, 3) {
                              for (ki: int32, 0, 32) {
                                for (rc.inner: int32, 0, 32) {
        } // end ho.outer_1
      } // end ko.outer_1

      // filter #2 cache read
      for (ko.inner: int32, 0, 2) {
        ...
      }

      // conv2d #2
      for (ko.c.inner: int32, 0, 2) {
        for (ho.c.inner: int32, 0, 2) {
          for (wo.c: int32, 0, 8) {
            for (rc.outer_1: int32, 0, 4) {
              for (hi.c: int32, 0, 8) {
                for (wi.c: int32, 0, 8) {
                  for (rh_1: int32, 0, 3) {
                    for (rw_1: int32, 0, 3) {
                      for (ki.c: int32, 0, 32) {
                        for (rc.inner_1: int32, 0, 32) {

      // write back output cache

    } // end ho.outer
  } // end ko.outer
```

There are two major changes here:

1) The first change is the farily obvious presence of the kernel height `rh` and width `rw` iterators, for example:

```
                  for (rh_1: int32, 0, 3) {
                    for (rw_1: int32, 0, 3) {
```

The effect of this change is to grow the filter cache by the size of the kernel.  Details below.

2) The second change is a bit more tricky.  Remember that we want to produce `h_split` (2) "full width" and "full channel depth" slices from each conv2d.  Given the 3x3 kernel size there are several changes to the schedule regarding the handling of the height dimension.

First, notice that in order to produce `h_split` (2) "full width" and "full channel depth" slices for conv2d #1 we will need `h_split + 1` (3) "full width" and "full channel depth" slices of the input.  This is because a 3x3 kernel (as opposed to a 1x1 kernel) creates a many-to-one relationship between the spatial coordinates of the input relative to the output.  To illustrate, the 3x3 kernel will "fall off the bottom" of the 2nd input slice requiring values from the vertically adjacent 3rd input slice in order to produce the 2nd full output slice.  Hence, we have the following input cache read over `h_split + 1` (3) input slices:

```
          for (ho.inner: int32, 0, 3) {
            if ((((ho.outer_1*2) + (ho.outer*2)) + ho.inner) < 8) {
```

The `if` statement above indicates NOT to prefetch the vertically adjacent slice at the "bottom" of the input since it does not exist.

Second, notice that conv2d #1 must produce sufficient output in the height dimension before conv2d #2 can proceed.  This is similar to the requirement that conv2d #1 in regard to the channel out dimension, but also different because we do not require *all* output in the height dimenson only *sufficient* output in the height dimension.  How much output in the height dimension is required?  The intuitive guess might be `h_split + 1` (3) slices but that is wrong and the reason is that the output spatial coordinates are "shrinking" relative to the input coordinate space due to lack of padding.  Hence 2 output slices from conv2d #1 are sufficient as intput to calculate 2 output slices from conv2d #2 and we get the following independent loop over `ho.outer_1` for conv2d #1:

```
        for (ho.outer_1: int32, 0, 2) {
```

There are similar `if` statements in the conv2d compute schedule to prevent computing off the "bottom" of the input and output.

(Same as above) Note that conv2d #1 has an independent loop over the channel out `ko.outer_1` dimension.  This is because the output channels of conv2d #1 are the input channels to conv2d #2 and we compute over all input channels for each conv2d so we must compute over all output channels of conv2d #1 before we compute conv2d #2.

```
      for (ko.outer_1: int32, 0, 2) {
```

## Cache Usage

*Input Cache*

The input cache grows to hold the vertically adjacent slice:

```
  allocate(packed_input.global: Pointer(global float32), float32, [196608]), storage_scope = global;
```

*Filter Cache*

The filter cache grows to hold the 3x3 filter filter:

```
  allocate(packed_filter.global: Pointer(global float32), float32, [73728]), storage_scope = global;
```

(Same as above) Note that there is just one cache which is reused for conv2d / filter #1 and conv2d / filter #2.

*Output Cache*

The output cache scales with the input cache:

```
  allocate(temp_output: Pointer(global float32), float32, [196608]), storage_scope = global;
```

(Same as above) Note that the input cache is reused to store the results of conv2d #2.

## Assumptions

* n/a

## To Do

* There may be some opportunity to optimized cache reuse in this case as the vertically adjacent input slice from a previous input cache read will be reloaded as in a subsequent input cache read

## Annotated TIR

```
primfn(placeholder_3: handle, placeholder_4: handle, placeholder_5: handle, output_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {output: Buffer(output_2: Pointer(float32), float32, [1, 8, 8, 4, 8, 8, 32], []),             // nhw8h8w32c
             placeholder_2: Buffer(placeholder_6: Pointer(float32), float32, [4, 4, 3, 3, 8, 32, 4], []), // oihw8i32o4i
             placeholder_1: Buffer(placeholder_7: Pointer(float32), float32, [4, 4, 3, 3, 8, 32, 4], []), // oihw8i32o4i
             placeholder: Buffer(placeholder_8: Pointer(float32), float32, [1, 64, 64, 128], [])}         // nhwc
  buffer_map = {placeholder_3: placeholder, placeholder_4: placeholder_1, placeholder_5: placeholder_2, output_1: output} {
  allocate(packed_input.global: Pointer(global float32), float32, [196608]), storage_scope = global;
  allocate(temp_output: Pointer(global float32), float32, [196608]), storage_scope = global;
  allocate(packed_filter.global: Pointer(global float32), float32, [73728]), storage_scope = global;
  for (ko.outer: int32, 0, 2) {
    for (ho.outer: int32, 0, 4) {
      // NOTE: compute over all output channels of conv2d #1 before computing conv2d #2
      for (ko.outer_1: int32, 0, 2) {
        // NOTE: compute enough height of conv2d #1 before computing conv2d #2
        for (ho.outer_1: int32, 0, 2) {

          // input cache read
          for (ho.inner: int32, 0, 3) {
            if ((((ho.outer_1*2) + (ho.outer*2)) + ho.inner) < 8) {
              for (wo: int32, 0, 8) {
                for (co: int32, 0, 4) {
                  for (hi: int32, 0, 8) {
                    for (wi: int32, 0, 8) {
                      for (ci: int32, 0, 32) {
                        packed_input.global[((((((ho.inner*65536) + (wo*8192)) + (co*2048)) + (hi*256)) + (wi*32)) + ci)] =
                          (float32*)placeholder_8[((((((((ho.outer_1*131072) + (ho.outer*131072)) + (ho.inner*65536)) + (hi*8192)) + (wo*1024)) + (wi*128)) + (co*32)) + ci)]
                      }
                    }
                  }
                }
              }
            }
          }

          // filter #1 cache read
          for (ko.inner: int32, 0, 2) {
            for (co: int32, 0, 4) {
              for (rh: int32, 0, 3) {
                for (rw: int32, 0, 3) {
                  for (cio: int32, 0, 8) {
                    for (ki: int32, 0, 32) {
                      for (cii: int32, 0, 4) {
                        packed_filter.global[(((((((ko.inner*36864) + (co*9216)) + (rh*3072)) + (rw*1024)) + (cio*128)) + (ki*4)) + cii)] =
                          (float32*)placeholder_7[((((((((ko.outer_1*73728) + (ko.inner*36864)) + (co*9216)) + (rh*3072)) + (rw*1024)) + (cio*128)) + (ki*4)) + cii)]
                      }
                    }
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
                if (((ho.outer_1*2) + ho.inner) < 3) {
                  for (hi.init: int32, 0, 8) {
                    for (wi.init: int32, 0, 8) {
                      for (ki.init: int32, 0, 32) {
                        temp_output[((((((((ho.outer_1*131072) + (ho.inner*65536)) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi.init*256)) + (wi.init*32)) + ki.init)] = 0f32
                      }
                    }
                  }
                }

                // compute
                if (((ho.outer_1*2) + ho.inner) < 3) {
                  if ((((ho.outer_1*2) + (ho.outer*2)) + ho.inner) < 8) {
                    for (rc.outer: int32, 0, 4) {
                      for (hi: int32, 0, 8) {
                        for (wi: int32, 0, 8) {
                          for (rh: int32, 0, 3) {
                            for (rw: int32, 0, 3) {
                              for (ki: int32, 0, 32) {
                                for (rc.inner: int32, 0, 32) {
                                  temp_output[((((((((ho.outer_1*131072) + (ho.inner*65536)) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] =
                                  (
                                    (float32*)temp_output[((((((((ho.outer_1*131072) + (ho.inner*65536)) + (wo*8192)) + (ko.outer_1*4096)) + (ko.inner*2048)) + (hi*256)) + (wi*32)) + ki)] +
                                    (
                                      (float32*)packed_input.global[((((((((floordiv((hi + rh), 8)*65536) + (ho.inner*65536)) + (floordiv((wi + rw), 8)*8192)) + (wo*8192)) + (rc.outer*2048)) + (floormod((hi + rh), 8)*256)) + (floormod((wi + rw), 8)*32)) + rc.inner)] *
                                      (float32*)packed_filter.global[(((((((ko.inner*36864) + (rc.outer*9216)) + (rh*3072)) + (rw*1024)) + (floordiv(rc.inner, 4)*128)) + (ki*4)) + floormod(rc.inner, 4))]
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
            }
          }
        }
      }

      // filter #2 cache read
      // NOTE: reusing same filter cache
      for (ko.inner: int32, 0, 2) {
        for (co: int32, 0, 4) {
          for (rh: int32, 0, 3) {
            for (rw: int32, 0, 3) {
              for (cio: int32, 0, 8) {
                for (ki: int32, 0, 32) {
                  for (cii: int32, 0, 4) {
                    packed_filter.global[(((((((ko.inner*36864) + (co*9216)) + (rh*3072)) + (rw*1024)) + (cio*128)) + (ki*4)) + cii)] =
                      (float32*)placeholder_6[((((((((ko.outer*73728) + (ko.inner*36864)) + (co*9216)) + (rh*3072)) + (rw*1024)) + (cio*128)) + (ki*4)) + cii)]
                  }
                }
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
                  for (rh_1: int32, 0, 3) {
                    for (rw_1: int32, 0, 3) {
                      for (ki.c: int32, 0, 32) {
                        for (rc.inner_1: int32, 0, 32) {
                          packed_input.global[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] =
                          (
                            (float32*)packed_input.global[((((((ho.c.inner*32768) + (wo.c*4096)) + (ko.c.inner*2048)) + (hi.c*256)) + (wi.c*32)) + ki.c)] +
                            (
                              (float32*)temp_output[((((((((floordiv((hi.c + rh_1), 8)*65536) + (ho.c.inner*65536)) + (floordiv((wi.c + rw_1), 8)*8192)) + (wo.c*8192)) + (rc.outer_1*2048)) + (floormod((hi.c + rh_1), 8)*256)) + (floormod((wi.c + rw_1), 8)*32)) + rc.inner_1)] *
                              (float32*)packed_filter.global[(((((((ko.c.inner*36864) + (rc.outer_1*9216)) + (rh_1*3072)) + (rw_1*1024)) + (floordiv(rc.inner_1, 4)*128)) + (ki.c*4)) + floormod(rc.inner_1, 4))]
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
