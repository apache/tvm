# Libcrc API Reference

### `update_crc_kermit( crc, c );`

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

The function `update_crc_kermit()` can be used to calculate the CRC value in a stream of bytes where it is not possible to first buffer the stream completely to calculate the CRC when all data is received. The parameters are the previous CRC value and the current byte which must be used to calculate the new CRC value.

In order for this function to work properly, the CRC value must be initialized before the first call to `update_crc_kermit()`. The most common initialization value is `CRC_START_KERMIT` to perform the Kermit CRC calculation.

### See Also

* [`crc_kermit();`](crc_kermit.md)
* [CRC start values](crc_start.md)
