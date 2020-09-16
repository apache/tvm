# Libcrc API Reference

### `update_crc_8( crc, c );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`crc`**|`uint8_t`|The CRC value calculated from the byte stream upto but not including the current byte|
|**`c`**|`unsigned char`|The next byte from the byte stream to be used in the CRC calculation|

### Return Value

| Type | Description |
| :--- | :--- |
|**`uint8_t`**|The new CRC value of the byte stream including the current byte|

### Description

The function `update_crc_8()` can be used to calculate the CRC value in a stream of bytes where it is not possible to first buffer the stream completely to calculate the CRC when all data is received. The parameters are the previous CRC value and the current byte which must be used to calculate the new CRC value.

In order for this function to work properly, the CRC value must be initialized before the first call to `update_crc_8()`. The most common initialization values is `CRC_START_8` to perform the CRC-8 CRC calculation as it is used by the Sensirion SHTxx temperature and humidity sensors.

### See Also

* [`crc_8();`](crc_8.md)
* [CRC start values](crc_start.md)
