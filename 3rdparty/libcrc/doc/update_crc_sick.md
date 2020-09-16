# Libcrc API Reference

### `update_crc_sick( crc, c, prev_byte );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`crc`**|`uint16_t`|The CRC value calculated from the byte stream upto but not including the current byte|
|**`c`**|`unsigned char`|The next byte from the byte stream to be used in the CRC calculation|
|**`prev_byte`**|`unsigned char`|The previous byte from the byte stream|

### Return Value

| Type | Description |
| :--- | :--- |
|**`uint16_t`**|The new CRC value of the byte stream including the current byte|

### Description

The function `update_crc_sick()` can be used to calculate the CRC value in a stream of bytes where it is not possible to first buffer the stream completely to calculate the CRC when all data is received. The parameters are the previous CRC value and the current byte which must be used to calculate the new CRC value.

In order for this function to work properly, the CRC value must be initialized before the first call to `update_crc_sick()`. In order for this function to work properly the initalization value `CRC_START_SICK` must be used.

Please note that every byte of the input stream must be passed twice to the function. First as parameter **`c`** and the secont time as parameter **`prev_time`**. For the first byte in the stream where no previous byte is known, you may use the value **`0`** for parameter `prev_byte`.

Please also note that when all processing is completed, that the low and high order bytes of the CRC value must be swapped to get the final CRC value.

If you need an example to implement `update_crc_sick()` properly in your code, you can look at the implementation of `crc_sick()` in the source file [`crcsick.c`](../src/crcsick.c) as this function contains the full algorithm including initialization and byte swap.

### See Also

* [`crc_sick();`](crc_sick.md)
* [CRC start values](crc_start.md)
