# Libcrc API Reference

### `update_crc_ccitt( crc, c );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`crc`**|`uint16_t`|The CRC value calculated from the byte stream upto but not including the current byte|
|**`c`**|`unsigned char`|The next byte from the byte stream to be used in the CRC calculation|

### Return Value

| Type | Description |
| :--- | :--- |
|**`uint16_t`**|The new CRC value of the byte stream including the current byte|

### Description

The function `update_crc_ccitt()` can be used to calculate the CRC value in a stream of bytes where it is not possible to first buffer the stream completely to calculate the CRC when all data is received. The parameters are the previous CRC value and the current byte which must be used to calculate the new CRC value.

In order for this function to work properly, the CRC value must be initialized before the first call to `update_crc_ccitt()`. The most common initialization values are `CRC_START_CCITT_XMODEM`, `CRC_START_CCITT_1D0F` and `CRC_START_CCITT_FFFF` to perform the CRC calculation according to the Xmodem protocol, and two other common CCITT implementations.

### See Also

* [`crc_ccitt_1d0f();`](crc_ccitt_1d0f.md)
* [`crc_ccitt_ffff();`](crc_ccitt_ffff.md)
* [`crc_xmodem();`](crc_xmodem.md)
* [CRC start values](crc_start.md)
