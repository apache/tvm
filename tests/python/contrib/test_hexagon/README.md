Documents manual TE schedule to illustrate Hexagon operator slicing.

High Level Notes:
* Using float32 (for now) so that tests will pass on CPU
* Using global storage scope (for now) which means "cache" reads and writes from global, to global
* TIR is pending changes from the work-in-progress layout RFC
  (https://github.com/apache/tvm-rfcs/pull/39)
* TIR has been hand-edited for context and clarity
  * Added C-style comments
  * Changed variable names
  * Added spacing and line breaks
* Naming conventions
  * Using input (instead of activation)
  * Using filter (instead of weight, kernel)
  * Using `k` to denote channel-out and `c` or `rc` (reduction channel) to denote channel-in
  * Using `rh` and `rw` (reduction height / width) to denote filter height and width

[Conv2d](test_conv2d_blocked.md)

[Conv2d -> Conv2d](test_conv2d_conv2d.md)